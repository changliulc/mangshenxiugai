#
# # -*- coding: utf-8 -*-
# """
# CH3 Closed-loop pipeline (REALPACK)
#
# This script is designed for thesis Chapter 3:
# - Edge baselines: Softmax (LogReg), Linear SVM, Decision Tree (control)
# - Offline baseline: 1D CNN on linear fixed-length resampling
# - DTW enhancements:
#     (1) DTW-ClsMin + CNN (single-template routing, constant 4 channels)
#     (2) DTW-MultiTemplate(K) + CNN (final method; K selected by validation)
# - Fairness control:
#     Replication Control (channel replication to match MultiTemplate input channels)
#
# Outputs:
# - latex_out/figures/*.png : confusion matrices, ablation plots, motivation plots
# - latex_out/tables/*.csv  : raw tables
# - latex_out/tables/*_rows.tex : LaTeX row snippets for \input
# - results/summary.json : machine-readable summary
#
# Usage:
#     python run_ch3_thesis_pipeline.py --mat PATH_TO_MAT --out_dir OUTDIR --seed 42
#
# Notes:
# - Uses the DTW implementations from dtw_cnn_handoff (numba accelerated).
# - Patches numba/coverage compatibility for some environments.
# """
# import os
# import json
# import time
# import argparse
# from pathlib import Path
# import numpy as np
#
# # ---- numba/coverage compatibility patch (safe no-op on normal envs)
# try:
#     import coverage
#     from types import SimpleNamespace
#     if not hasattr(coverage, "types"):
#         coverage.types = SimpleNamespace()
#     for _name in [
#         "Tracer", "TTraceData", "TTraceFn",
#         "TShouldTraceFn", "TShouldStartContextFn",
#         "TWarnFn", "TFileDisposition",
#     ]:
#         if not hasattr(coverage.types, _name):
#             setattr(coverage.types, _name, object)
# except Exception:
#     pass
#
# # Numba compatibility: ensure the package exposes `numba.core` as an attribute.
# # In some environments, submodules exist in sys.modules but are not attached to the package.
# try:
#     import sys as _sys
#     import numba as _numba
#     import numba.core as _numba_core  # noqa: F401
#     if not hasattr(_numba, "core"):
#         _numba.core = _sys.modules.get("numba.core", _numba_core)
# except Exception:
#     pass
#
#
# import scipy.io
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import confusion_matrix
#
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
#
# import matplotlib.pyplot as plt
#
# # Add local dtw modules path
# _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# _DTW_CODE_DIR = os.path.join(_THIS_DIR, "dtw_cnn_handoff", "code")
# import sys
# if _DTW_CODE_DIR not in sys.path:
#     sys.path.insert(0, _DTW_CODE_DIR)
#
# import run_dtw_multi_sweep as M
# import run_dtw_multi_quantile as MQ
# import run_dtw_clsmin_sweep as C
# import run_dtw_clsmin_dba as DBA
#
# # Some environments import numba but do not expose `numba.core` on package attrs.
# # This is required by numba internals in certain call paths.
# try:
#     import numba
#     if "numba.core" in sys.modules and not hasattr(numba, "core"):
#         numba.core = sys.modules["numba.core"]
# except Exception:
#     pass
#
#
# # -----------------------------
# # Utilities: split / metrics
# # -----------------------------
# def stratified_split(y, seed=42, train_ratio=0.70, val_ratio=0.15):
#     rng = np.random.default_rng(seed)
#     idx_tr, idx_va, idx_te = [], [], []
#     for cls in np.unique(y):
#         idx = np.where(y == cls)[0]
#         rng.shuffle(idx)
#         n = len(idx)
#         n_tr = int(round(train_ratio * n))
#         n_va = int(round(val_ratio * n))
#         idx_tr.extend(idx[:n_tr])
#         idx_va.extend(idx[n_tr:n_tr + n_va])
#         idx_te.extend(idx[n_tr + n_va:])
#     return np.array(idx_tr), np.array(idx_va), np.array(idx_te)
#
#
# def macro_f1(y_true, y_pred, n_classes=3):
#     y_true = np.asarray(y_true)
#     y_pred = np.asarray(y_pred)
#     f1s = []
#     for c in range(n_classes):
#         tp = np.sum((y_true == c) & (y_pred == c))
#         fp = np.sum((y_true != c) & (y_pred == c))
#         fn = np.sum((y_true == c) & (y_pred != c))
#         denom = 2 * tp + fp + fn
#         f1s.append(0.0 if denom == 0 else (2.0 * tp) / denom)
#     return float(np.mean(f1s))
#
#
# def split_counts(y, idx):
#     return np.bincount(y[idx], minlength=3)
#
#
# # -----------------------------
# # Data loading and preprocessing
# # -----------------------------
# def load_xyz_y_from_mat(mat_path: str):
#     """Load ProcessedData/targetLength from .mat into Python list and labels.
#
#     Returns
#     -------
#     xyz_list: list[np.ndarray], each (T_i, 3)
#     y: np.ndarray, shape (N,), int64
#     tlen: np.ndarray, shape (N,), int
#     """
#     mat = scipy.io.loadmat(mat_path)
#     PD = mat["ProcessedData"]
#     TL = mat["targetLength"]
#
#     xyz_list = []
#     y_list = []
#     tlen_list = []
#     for c in range(3):
#         cell = PD[0, c]  # shape (1, n_c)
#         tl = TL[0, c].reshape(-1)
#         n_c = cell.shape[1]
#         for i in range(n_c):
#             arr = np.array(cell[0, i], dtype=np.float32)
#             t = int(np.array(tl[i]).squeeze())
#             xyz_list.append(arr[:t, :])
#             y_list.append(c)
#             tlen_list.append(t)
#
#     y = np.array(y_list, dtype=np.int64)
#     tlen = np.array(tlen_list, dtype=int)
#     return xyz_list, y, tlen
#
#
# def resample_linear(x: np.ndarray, L: int) -> np.ndarray:
#     x = np.asarray(x, dtype=np.float32)
#     T, D = x.shape
#     if T == L:
#         return x
#     t0 = np.linspace(0.0, 1.0, T)
#     t1 = np.linspace(0.0, 1.0, L)
#     y = np.empty((L, D), np.float32)
#     for d in range(D):
#         y[:, d] = np.interp(t1, t0, x[:, d])
#     return y
#
#
# def make_4ch_seq(xyz: np.ndarray, L: int) -> np.ndarray:
#     xr = resample_linear(xyz, L)  # (L,3)
#     mag = np.sqrt(np.sum(xr ** 2, axis=1, keepdims=True))  # (L,1)
#     out = np.concatenate([xr, mag], axis=1)  # (L,4)
#     return out.T  # (4,L)
#
#
# def build_edge_feature_matrix(xyz_list, L_feat: int) -> np.ndarray:
#     """Edge feature: downsampled waveform (x,y,z,mag) concatenated (4*L_feat)."""
#     N = len(xyz_list)
#     X = np.zeros((N, 4 * L_feat), np.float32)
#     for i, xyz in enumerate(xyz_list):
#         xr = resample_linear(xyz, L_feat)  # (L_feat,3)
#         mag = np.sqrt(np.sum(xr ** 2, axis=1, keepdims=False))  # (L_feat,)
#         X[i] = np.concatenate([xr[:, 0], xr[:, 1], xr[:, 2], mag], axis=0)
#     return X
#
#
# def standardize_fit(X):
#     mu = X.mean(axis=0, keepdims=True).astype(np.float32)
#     sd = X.std(axis=0, keepdims=True).astype(np.float32)
#     sd[sd < 1e-6] = 1.0
#     return mu, sd
#
#
# def standardize_apply(X, mu, sd):
#     return (X - mu) / sd
#
#
# def compute_norm_stats_seq(X, idx_tr):
#     mu = X[idx_tr].mean(axis=(0, 2), keepdims=True).astype(np.float32)
#     sd = X[idx_tr].std(axis=(0, 2), keepdims=True).astype(np.float32)
#     sd[sd < 1e-6] = 1.0
#     return mu, sd
#
#
# def normalize_seq(X, mu, sd):
#     return (X - mu) / sd
#
#
# # -----------------------------
# # CNN model
# # -----------------------------
# class SeqDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.from_numpy(X.astype(np.float32))
#         self.y = torch.from_numpy(y.astype(np.int64))
#
#     def __len__(self):
#         return self.X.shape[0]
#
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
#
#
# class SimpleCNN(nn.Module):
#     """Lightweight 1D CNN (same for baseline / DTW variants)."""
#
#     def __init__(self, in_ch, n_classes=3):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(32, 64, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.Conv1d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool1d(1),
#             nn.Flatten(),
#             nn.Dropout(0.2),
#             nn.Linear(128, n_classes),
#         )
#
#     def forward(self, x):
#         return self.net(x)
#
#
# def predict_model(model, X, batch_size=256):
#     model.eval()
#     ds = SeqDataset(X, np.zeros((X.shape[0],), dtype=np.int64))
#     dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
#     preds = []
#     with torch.no_grad():
#         for xb, _ in dl:
#             logits = model(xb)
#             preds.append(torch.argmax(logits, dim=1).cpu().numpy())
#     return np.concatenate(preds)
#
#
# def train_cnn(X, y, idx_tr, idx_va, idx_te,
#               in_ch, seed=42, max_epochs=60, patience=10,
#               lr=1e-3, weight_decay=0.0):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     torch.set_num_threads(1)
#
#     model = SimpleCNN(in_ch)
#
#     # Loss: unweighted cross-entropy (keep consistent with baseline numbers)
#     crit = nn.CrossEntropyLoss()
#     optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
#
#     dl_tr = DataLoader(SeqDataset(X[idx_tr], y[idx_tr]), batch_size=64, shuffle=True)
#     dl_va = DataLoader(SeqDataset(X[idx_va], y[idx_va]), batch_size=256, shuffle=False)
#
#     best_f1 = -1.0
#     best_state = None
#     best_epoch = 0
#     bad = 0
#
#     for epoch in range(1, max_epochs + 1):
#         model.train()
#         for xb, yb in dl_tr:
#             optim.zero_grad()
#             loss = crit(model(xb), yb)
#             loss.backward()
#             optim.step()
#
#         # validation
#         model.eval()
#         pv, tv = [], []
#         with torch.no_grad():
#             for xb, yb in dl_va:
#                 pv.append(torch.argmax(model(xb), dim=1).cpu().numpy())
#                 tv.append(yb.cpu().numpy())
#         pv = np.concatenate(pv)
#         tv = np.concatenate(tv)
#         f1 = macro_f1(tv, pv, 3)
#
#         if f1 > best_f1 + 1e-6:
#             best_f1 = f1
#             best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#             best_epoch = epoch
#             bad = 0
#         else:
#             bad += 1
#             if bad >= patience:
#                 break
#
#     if best_state is not None:
#         model.load_state_dict(best_state)
#
#     # final eval
#     pred_va = predict_model(model, X[idx_va])
#     pred_te = predict_model(model, X[idx_te])
#
#     out = {
#         "best_epoch": int(best_epoch),
#         "val_acc": float(np.mean(pred_va == y[idx_va])),
#         "val_f1": macro_f1(y[idx_va], pred_va, 3),
#         "test_acc": float(np.mean(pred_te == y[idx_te])),
#         "test_f1": macro_f1(y[idx_te], pred_te, 3),
#         "y_pred_val": pred_va.astype(int).tolist(),
#         "y_pred_test": pred_te.astype(int).tolist(),
#     }
#     return model, out
#
#
# # -----------------------------
# # Plotting
# # -----------------------------
# # def ensure_plot_style():
# #     # Try a list of common CJK fonts (fallback to DejaVu)
# #     plt.rcParams["font.family"] = [
# #         "Noto Sans CJK JP",
# #         "SimHei",
# #         "Microsoft YaHei",
# #         "PingFang SC",
# #         "STSong",
# #         "DejaVu Sans",
# #     ]
# #     plt.rcParams["axes.unicode_minus"] = False
# #
# def ensure_plot_style():
#     import matplotlib.pyplot as plt
#     import logging
#
#     # Windows 常见中文字体优先；DejaVu 兜底
#     plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
#     plt.rcParams["axes.unicode_minus"] = False
#
#     # 可选：彻底禁止 font_manager 的 “findfont” 刷屏（不影响绘图）
#     logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
# def plot_confusion(y_true, y_pred, labels_cn, title, out_path_png, normalize=False):
#     cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels_cn))))
#     cm_disp = cm.astype(np.float32)
#     if normalize:
#         cm_disp = cm_disp / np.maximum(cm_disp.sum(axis=1, keepdims=True), 1.0)
#
#     fig, ax = plt.subplots(figsize=(5, 4))
#     im = ax.imshow(cm_disp, interpolation="nearest", cmap="viridis")
#     ax.set_title(title, fontsize=13)
#     ax.set_xticks(np.arange(len(labels_cn)))
#     ax.set_yticks(np.arange(len(labels_cn)))
#     ax.set_xticklabels(labels_cn)
#     ax.set_yticklabels(labels_cn)
#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("True")
#
#     fmt = ".2f" if normalize else "d"
#     thresh = cm_disp.max() / 2.0 if cm_disp.size > 0 else 0.0
#     for i in range(cm_disp.shape[0]):
#         for j in range(cm_disp.shape[1]):
#             val = cm_disp[i, j]
#             txt = format(val, fmt) if normalize else str(int(val))
#             ax.text(
#                 j, i, txt,
#                 ha="center", va="center",
#                 color="white" if val > thresh else "black",
#                 fontsize=11,
#             )
#
#     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     fig.tight_layout()
#     fig.savefig(out_path_png, dpi=200, bbox_inches="tight")
#     plt.close(fig)
#
#
# def zscore(a):
#     a = np.asarray(a, dtype=np.float32)
#     return (a - a.mean()) / (a.std() + 1e-6)
#
#
# def plot_motivation_short_long(mag_short, mag_long, fs, out_path_png):
#     t_short = np.arange(len(mag_short)) / fs
#     t_long = np.arange(len(mag_long)) / fs
#     fig, ax = plt.subplots(figsize=(6, 3.5))
#     ax.plot(t_short, mag_short, label=f"短事件（T={len(mag_short)}）")
#     ax.plot(t_long, mag_long, label=f"长事件（T={len(mag_long)}）")
#     ax.set_xlabel("时间 / s")
#     ax.set_ylabel("模值幅度")
#     ax.set_title("同一类别样本的时间伸缩现象（示例）")
#     ax.legend()
#     ax.grid(True, linestyle="--", alpha=0.4)
#     fig.tight_layout()
#     fig.savefig(out_path_png, dpi=200, bbox_inches="tight")
#     plt.close(fig)
#
#
# def plot_motivation_dtw_align_z(x_lin_4L, x_warp_L4, tpl_4L, out_path_png):
#     mag_lin = zscore(x_lin_4L[3])
#     mag_warp = zscore(x_warp_L4[:, 3])
#     mag_tpl = zscore(tpl_4L[3])
#     t = np.arange(len(mag_lin))
#     fig, ax = plt.subplots(figsize=(6, 3.5))
#     ax.plot(t, mag_lin, label="线性定长(原始输入)")
#     ax.plot(t, mag_warp, label="DTW 对齐后")
#     ax.plot(t, mag_tpl, label="参考模板", linestyle="--")
#     ax.set_xlabel("采样点")
#     ax.set_ylabel("模值（z-score）")
#     ax.set_title("线性定长 vs DTW 对齐（归一化示例）")
#     ax.legend()
#     ax.grid(True, linestyle="--", alpha=0.4)
#     fig.tight_layout()
#     fig.savefig(out_path_png, dpi=200, bbox_inches="tight")
#     plt.close(fig)
#
#
# def plot_k_ablation(df_k, out_path_png):
#     fig, ax = plt.subplots(figsize=(5.2, 3.6))
#     ax.plot(df_k["K"], df_k["test_f1"], marker="o", label="Test Macro-F1")
#     ax.plot(df_k["K"], df_k["val_f1"], marker="s", label="Val Macro-F1")
#     ax.set_xlabel("模板数 K（每类）")
#     ax.set_ylabel("Macro-F1")
#     ax.set_title("DTW 多模板数量消融")
#     ax.set_xticks(df_k["K"])
#     ax.set_ylim(0.0, 1.0)
#     ax.grid(True, linestyle="--", alpha=0.4)
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig(out_path_png, dpi=200, bbox_inches="tight")
#     plt.close(fig)
#
#
# def plot_replication(df_rep, out_path_png):
#     fig, ax = plt.subplots(figsize=(6.4, 3.6))
#     x = np.arange(len(df_rep))
#     ax.bar(x, df_rep["test_f1"])
#     ax.set_xticks(x)
#     ax.set_xticklabels(df_rep["method"], rotation=12, ha="right")
#     ax.set_ylabel("Test Macro-F1")
#     ax.set_title("Replication Control 对比（K=4）")
#     ax.set_ylim(0.0, 1.0)
#     ax.grid(True, axis="y", linestyle="--", alpha=0.4)
#     fig.tight_layout()
#     fig.savefig(out_path_png, dpi=200, bbox_inches="tight")
#     plt.close(fig)
#
#
# # -----------------------------
# # LaTeX row writer
# # -----------------------------
# def df_to_latex_rows(df, cols, float_fmt="{:.3f}"):
#     lines = []
#     for _, row in df.iterrows():
#         items = []
#         for c in cols:
#             v = row[c]
#             if isinstance(v, (float, np.floating)):
#                 items.append(float_fmt.format(float(v)))
#             else:
#                 items.append(str(v))
#         lines.append(" & ".join(items) + r" \\")
#     return "\n".join(lines)
#
#
# # -----------------------------
# # Main
# # -----------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     # parser.add_argument("--mat", required=True, help="Path to REALPACK .mat file")
#     parser.add_argument(
#         "--mat",
#         default=r"D:\xidian_Master\研究生论文\毕业论文\实验数据\第三章\数据\processedVehicleData_3class_REAL (2).mat",
#         help="Path to REALPACK .mat file"
#     )
#     parser.add_argument("--out_dir", default="ch3_thesis_out", help="Output root dir")
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--L", type=int, default=176, help="Fixed length for CNN")
#     parser.add_argument("--L_feat", type=int, default=128, help="Edge feature length")
#     parser.add_argument("--wR", type=float, default=0.15, help="DTW window ratio")
#     parser.add_argument("--step", type=float, default=0.05, help="DTW step penalty")
#     parser.add_argument("--dba_iter", type=int, default=5, help="DBA iterations for ClsMin-DBA")
#     parser.add_argument("--K_list", default="1,2,4", help="Comma-separated K candidates")
#     parser.add_argument("--max_epochs", type=int, default=30, help="max epochs for CNN training")
#     parser.add_argument("--patience", type=int, default=6, help="early-stopping patience")
#     parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate for CNN")
#     parser.add_argument("--weight_decay", type=float, default=0.0, help="Adam weight decay (L2) for CNN")
#
#     args = parser.parse_args()
#     train_kwargs = dict(
#         max_epochs=args.max_epochs,
#         patience=args.patience,
#         lr=args.lr,
#         weight_decay=args.weight_decay,
#     )
#
#     ensure_plot_style()
#
#     out_root = args.out_dir
#     fig_dir = os.path.join(out_root, "latex_out", "figures")
#     tab_dir = os.path.join(out_root, "latex_out", "tables")
#     res_dir = os.path.join(out_root, "results")
#     os.makedirs(fig_dir, exist_ok=True)
#     os.makedirs(tab_dir, exist_ok=True)
#     os.makedirs(res_dir, exist_ok=True)
#
#     # ---- load data
#     xyz_list, y, tlen = load_xyz_y_from_mat(args.mat)
#     idx_tr, idx_va, idx_te = stratified_split(y, seed=args.seed)
#     ytr, yva, yte = y[idx_tr], y[idx_va], y[idx_te]
#
#     # ---- persist split indices (hard constraint: all methods share exactly the same split)
#     split_npz = os.path.join(res_dir, f"split_indices_seed{args.seed}.npz")
#     np.savez(split_npz, idx_tr=idx_tr, idx_va=idx_va, idx_te=idx_te)
#     try:
#         from scipy.io import savemat
#         savemat(os.path.join(res_dir, f"split_indices_seed{args.seed}.mat"), {"idx_tr": idx_tr, "idx_va": idx_va, "idx_te": idx_te})
#     except Exception as _e:
#         print(f"[WARN] split .mat export skipped: {_e}")
#
#
#     labels_cn = ["小型车", "中型车", "大型车"]
#
#     # ---- dataset stats table
#     stats_rows = []
#     for c in range(3):
#         tl = tlen[y == c]
#         stats_rows.append({
#             "class": c,
#             "class_name": labels_cn[c],
#             "count": int(len(tl)),
#             "min": int(tl.min()),
#             "median": int(np.percentile(tl, 50)),
#             "max": int(tl.max()),
#         })
#     split_table = {
#         "train": split_counts(y, idx_tr).tolist(),
#         "val": split_counts(y, idx_va).tolist(),
#         "test": split_counts(y, idx_te).tolist(),
#     }
#
#     # ---- Edge baselines (Softmax/SVM/DecisionTree, <=K features)
#     print("[1/6] Edge baselines (Softmax/SVM/DT) with <=K features")
#     # Hand-crafted features follow the MATLAB v4 definition (48 candidates),
#     # then we select K<=7 via val Macro-F1 (epsilon-minK).
#     sys.path.insert(0, str(Path(__file__).resolve().parent / "ch3_addons"))
#     from run_ch3_edge_baselines_topk import build_feature_matrix as build_edge_X, run_edge_baselines
#     X_edge = build_edge_X(xyz_list)  # (N,48)
#     edge_res = run_edge_baselines(X_edge, y, idx_tr, idx_va, idx_te, k_min=5, k_max=7, seed=args.seed)
#     edge_metrics = edge_res.metrics
#     (Path(res_dir) / "edge_selected_features.json").write_text(json.dumps(edge_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
#     edge_soft_acc = edge_metrics["softmax"]["acc"]
#     edge_soft_f1  = edge_metrics["softmax"]["macro_f1"]
#     edge_svm_acc  = edge_metrics["svm"]["acc"]
#     edge_svm_f1   = edge_metrics["svm"]["macro_f1"]
#     edge_dt_acc   = edge_metrics["dt"]["acc"]
#     edge_dt_f1    = edge_metrics["dt"]["macro_f1"]
#     cm_soft = np.asarray(edge_metrics["softmax"]["cm"], dtype=int)
#     cm_svm  = np.asarray(edge_metrics["svm"]["cm"], dtype=int)
#     cm_dt   = np.asarray(edge_metrics["dt"]["cm"], dtype=int)
#     selected_feat_names = edge_metrics["selected_names"]
#     # Pack into the same dict format used later for CSV/LaTeX export
#     K_edge = int(edge_metrics["feature_k"])
#     edge_soft = {"method": f"端侧 Softmax（K={K_edge}）", "test_acc": edge_soft_acc, "test_f1": edge_soft_f1, "cm": cm_soft.tolist()}
#     edge_svm  = {"method": f"端侧 SVM（K={K_edge}）", "test_acc": edge_svm_acc,  "test_f1": edge_svm_f1,  "cm": cm_svm.tolist()}
#     edge_dt   = {"method": f"端侧 DT（K={K_edge}）", "test_acc": edge_dt_acc,   "test_f1": edge_dt_f1,   "cm": cm_dt.tolist()}
#     print("  Edge selected K={}, features={}".format(edge_metrics["feature_k"], selected_feat_names))
#
#     # ---- Baseline CNN (linear fixed length)
#     X_raw = M.build_baseline_data(xyz_list, args.L)
#     mu_b, sd_b = compute_norm_stats_seq(X_raw, idx_tr)
#     Xn = normalize_seq(X_raw, mu_b, sd_b)
#
#     t0 = time.time()
#     _, cnn_base = train_cnn(Xn, y, idx_tr, idx_va, idx_te, in_ch=4, seed=args.seed, **train_kwargs)
#     t_base = time.time() - t0
#
#     # Baseline confusion matrix
#     plot_confusion(yte, np.array(cnn_base["y_pred_test"]), labels_cn,
#                    "离线混淆矩阵（线性定长CNN）",
#                    os.path.join(fig_dir, "cm_cnn_baseline.png"), normalize=False)
#
#     # ---- DTW-ClsMin + CNN (single-template routing)
#     # NOTE: We keep BOTH variants for a clean ablation:
#     #   - ClsMin-Mean: mean template (ablation)
#     #   - ClsMin-DBA : DBA template (recommended in thesis; DTW barycenter)
#     # Convert wR to window length w (for demo plots)
#     w = int(round(args.wR * args.L))
#
#     # (A1) Mean-template ClsMin (ablation)
#     X_clsmin_mean = C.build_dtw_clsmin(X_raw, y, idx_tr, wR=args.wR, step=args.step)
#     mu_cm, sd_cm = compute_norm_stats_seq(X_clsmin_mean, idx_tr)
#     X_clsmin_mean_n = normalize_seq(X_clsmin_mean, mu_cm, sd_cm)
#     t0 = time.time()
#     _, cnn_clsmin_mean = train_cnn(X_clsmin_mean_n, y, idx_tr, idx_va, idx_te, in_ch=4, seed=args.seed, **train_kwargs)
#     t_clsmin_mean = time.time() - t0
#
#     # (O2) DBA-template ClsMin (main / key evidence)
#     X_clsmin_dba = DBA.build_dtw_clsmin_dba(X_raw, y, idx_tr, wR=args.wR, step=args.step, n_iter=args.dba_iter)
#     mu_cd, sd_cd = compute_norm_stats_seq(X_clsmin_dba, idx_tr)
#     X_clsmin_dba_n = normalize_seq(X_clsmin_dba, mu_cd, sd_cd)
#     t0 = time.time()
#     _, cnn_clsmin_dba = train_cnn(X_clsmin_dba_n, y, idx_tr, idx_va, idx_te, in_ch=4, seed=args.seed, **train_kwargs)
#     t_clsmin_dba = time.time() - t0
#
#     plot_confusion(yte, np.array(cnn_clsmin_dba["y_pred_test"]), labels_cn,
#                    "离线混淆矩阵（DTW-ClsMin-DBA + CNN）",
#                    os.path.join(fig_dir, "cm_dtw_clsmin_cnn.png"), normalize=False)
#
#     # ---- DTW MultiTemplate sweep over K candidates
#     K_list = [int(s) for s in args.K_list.split(",") if s.strip()]
#
#     multi_rows = []
#     best_multi = None
#
#     for K in K_list:
#         t_build0 = time.time()
#         X_multi, _templates = MQ.build_dtw_multi_quantile(
#             X_raw, y, idx_tr, tlen, templates_per_class=K, wR=args.wR, step=args.step
#         )
#         t_build = time.time() - t_build0
#
#         mu_m, sd_m = compute_norm_stats_seq(X_multi, idx_tr)
#         X_multi_n = normalize_seq(X_multi, mu_m, sd_m)
#
#         t_train0 = time.time()
#         _, cnn_multi = train_cnn(X_multi_n, y, idx_tr, idx_va, idx_te,
#                                  in_ch=X_multi_n.shape[1], seed=args.seed, **train_kwargs)
#         t_train = time.time() - t_train0
#
#         row = {
#             "K": int(K),
#             "in_ch": int(X_multi.shape[1]),
#             "build_s": float(t_build),
#             "train_s": float(t_train),
#             "val_acc": float(cnn_multi["val_acc"]),
#             "val_f1": float(cnn_multi["val_f1"]),
#             "test_acc": float(cnn_multi["test_acc"]),
#             "test_f1": float(cnn_multi["test_f1"]),
#         }
#         multi_rows.append(row)
#
#         if best_multi is None or row["val_f1"] > best_multi["row"]["val_f1"] + 1e-6:
#             best_multi = {
#                 "K": K,
#                 "X_multi": X_multi,
#                 "cnn": cnn_multi,
#                 "row": row,
#             }
#
#     # best MultiTemplate
#     K_star = int(best_multi["K"])
#     cnn_multi_star = best_multi["cnn"]
#     X_multi_star = best_multi["X_multi"]
#     plot_confusion(yte, np.array(cnn_multi_star["y_pred_test"]), labels_cn,
#                    f"离线混淆矩阵（DTW-MultiTemplate(K={K_star}) + CNN）",
#                    os.path.join(fig_dir, "cm_dtw_multi_cnn.png"), normalize=False)
#
#     # ---- Replication control (match channel count to best multi)
#     rep_factor = X_multi_star.shape[1] // X_raw.shape[1]
#     X_rep = np.tile(X_raw, (1, rep_factor, 1))
#     mu_r, sd_r = compute_norm_stats_seq(X_rep, idx_tr)
#     X_rep_n = normalize_seq(X_rep, mu_r, sd_r)
#
#     t0 = time.time()
#     _, cnn_rep = train_cnn(X_rep_n, y, idx_tr, idx_va, idx_te,
#                            in_ch=X_rep_n.shape[1], seed=args.seed, **train_kwargs)
#     t_rep = time.time() - t0
#
#     plot_confusion(yte, np.array(cnn_rep["y_pred_test"]), labels_cn,
#                    f"离线混淆矩阵（Replication Control, K={K_star})",
#                    os.path.join(fig_dir, "cm_replication_control.png"), normalize=False)
#
#     # ---- Motivation figures
#     fs = 50.0
#     # pick class 0 short/long
#     idx_c0 = np.where(y == 0)[0]
#     tl0 = tlen[idx_c0]
#     i_short = int(idx_c0[np.argmin(tl0)])
#     i_long = int(idx_c0[np.argmax(tl0)])
#
#     mag_short = np.sqrt(np.sum(xyz_list[i_short] ** 2, axis=1))
#     mag_long = np.sqrt(np.sum(xyz_list[i_long] ** 2, axis=1))
#     plot_motivation_short_long(
#         mag_short, mag_long, fs,
#         os.path.join(fig_dir, "fig_motivation_speed.png")
#     )
#
#     # DTW alignment demo: pick a test sample of class 1 if exists
#     idx_test_c1 = idx_te[y[idx_te] == 1]
#     if len(idx_test_c1) > 0:
#         i_ex = int(idx_test_c1[0])
#         cls = int(y[i_ex])
#         tpl = X_raw[idx_tr][y[idx_tr] == cls].mean(axis=0)  # (4,L)
#         x_lin = X_raw[i_ex]  # (4,L)
#         x_warp = C.dtw_warp_mv(x_lin.T, tpl.T, w, args.step)  # (L,4)
#         plot_motivation_dtw_align_z(
#             x_lin, x_warp, tpl,
#             os.path.join(fig_dir, "fig_motivation_dtw_align_z.png")
#         )
#
#     # ---- Build tables (csv + LaTeX rows)
#     import pandas as pd
#
#     df_stats = pd.DataFrame(stats_rows)
#     df_split = pd.DataFrame([
#         {"split": "train", "小型车": split_table["train"][0], "中型车": split_table["train"][1], "大型车": split_table["train"][2]},
#         {"split": "val",   "小型车": split_table["val"][0],   "中型车": split_table["val"][1],   "大型车": split_table["val"][2]},
#         {"split": "test",  "小型车": split_table["test"][0],  "中型车": split_table["test"][1],  "大型车": split_table["test"][2]},
#     ])
#
#     df_edge = pd.DataFrame([edge_soft, edge_svm, edge_dt])
#
#     # Main table (recommended for thesis): 4 rows (clean narrative)
#     df_overall = pd.DataFrame([
#         {"method": "端侧 Softmax（部署）", "test_acc": edge_soft["test_acc"], "test_f1": edge_soft["test_f1"]},
#         {"method": "端侧 SVM（对照）", "test_acc": edge_svm["test_acc"], "test_f1": edge_svm["test_f1"]},
#         {"method": "离线 CNN（线性定长）", "test_acc": cnn_base["test_acc"], "test_f1": cnn_base["test_f1"]},
#         {"method": f"DTW-MultiTemplate(K={K_star}) + CNN（本文方法）", "test_acc": cnn_multi_star["test_acc"], "test_f1": cnn_multi_star["test_f1"]},
#     ])
#
#     # ClsMin template ablation: Mean vs DBA (single-row evidence, not in main table)
#     df_clsmin_tpl = pd.DataFrame([
#         {"method": "DTW-ClsMin-Mean + CNN（消融）", "test_acc": cnn_clsmin_mean["test_acc"], "test_f1": cnn_clsmin_mean["test_f1"]},
#         {"method": f"DTW-ClsMin-DBA + CNN（消融，iter={args.dba_iter})", "test_acc": cnn_clsmin_dba["test_acc"], "test_f1": cnn_clsmin_dba["test_f1"]},
#     ])
#
#     df_k = pd.DataFrame(multi_rows).sort_values("K").reset_index(drop=True)
#
#     df_rep = pd.DataFrame([
#         {"method": f"Replication Control（通道复制，K={K_star}）", "test_acc": cnn_rep["test_acc"], "test_f1": cnn_rep["test_f1"]},
#         {"method": f"DTW-MultiTemplate（K={K_star}）", "test_acc": cnn_multi_star["test_acc"], "test_f1": cnn_multi_star["test_f1"]},
#     ])
#
#     # Save CSV
#     df_stats.to_csv(os.path.join(tab_dir, "ch3_dataset_stats.csv"), index=False)
#     df_split.to_csv(os.path.join(tab_dir, "ch3_dataset_split_counts.csv"), index=False)
#     df_edge.to_csv(os.path.join(tab_dir, "ch3_edge_baselines.csv"), index=False)
#     df_overall.to_csv(os.path.join(tab_dir, "ch3_overall.csv"), index=False)
#     df_k.to_csv(os.path.join(tab_dir, "ch3_ablation_K.csv"), index=False)
#     df_rep.to_csv(os.path.join(tab_dir, "ch3_ablation_replication.csv"), index=False)
#     df_clsmin_tpl.to_csv(os.path.join(tab_dir, "ch3_ablation_clsmin_template.csv"), index=False)
#
#     # Save LaTeX rows
#     with open(os.path.join(tab_dir, "ch3_dataset_stats_rows.tex"), "w", encoding="utf-8") as f:
#         f.write(df_to_latex_rows(df_stats, ["class_name", "count", "min", "median", "max"], float_fmt="{:.0f}"))
#     with open(os.path.join(tab_dir, "ch3_dataset_split_rows.tex"), "w", encoding="utf-8") as f:
#         f.write(df_to_latex_rows(df_split, ["split", "小型车", "中型车", "大型车"], float_fmt="{:.0f}"))
#     with open(os.path.join(tab_dir, "ch3_edge_rows.tex"), "w", encoding="utf-8") as f:
#         f.write(df_to_latex_rows(df_edge, ["method", "test_acc", "test_f1"]))
#     with open(os.path.join(tab_dir, "ch3_overall_rows.tex"), "w", encoding="utf-8") as f:
#         f.write(df_to_latex_rows(df_overall, ["method", "test_acc", "test_f1"]))
#     with open(os.path.join(tab_dir, "ch3_ablation_K_rows.tex"), "w", encoding="utf-8") as f:
#         f.write(df_to_latex_rows(df_k, ["K", "in_ch", "test_acc", "test_f1"]))
#     with open(os.path.join(tab_dir, "ch3_ablation_rep_rows.tex"), "w", encoding="utf-8") as f:
#         f.write(df_to_latex_rows(df_rep, ["method", "test_acc", "test_f1"]))
#
#     with open(os.path.join(tab_dir, "ch3_ablation_clsmin_template_rows.tex"), "w", encoding="utf-8") as f:
#         f.write(df_to_latex_rows(df_clsmin_tpl, ["method", "test_acc", "test_f1"]))
#
#     # Plots for ablations
#     plot_k_ablation(df_k, os.path.join(fig_dir, "ablation_K.png"))
#     plot_replication(df_rep, os.path.join(fig_dir, "ablation_replication.png"))
#
#     # ---- Save summary JSON
#     summary = {
#         "seed": int(args.seed),
#         "L": int(args.L),
#         "L_feat": int(args.L_feat),
#         "dtw": {"wR": float(args.wR), "step": float(args.step), "dba_iter": int(args.dba_iter)},
#         "split_counts": split_table,
#         "edge": {"softmax": edge_soft, "svm": edge_svm, "dt": edge_dt},
#         "offline": {
#             "cnn_baseline": {"metrics": cnn_base, "train_time_s": float(t_base)},
#             "dtw_clsmin_mean_cnn": {"metrics": cnn_clsmin_mean, "train_time_s": float(t_clsmin_mean)},
#             "dtw_clsmin_dba_cnn": {"metrics": cnn_clsmin_dba, "train_time_s": float(t_clsmin_dba)},
#             "dtw_multi_best": {"K": K_star, "metrics": cnn_multi_star},
#             "replication_control": {"K": K_star, "metrics": cnn_rep, "train_time_s": float(t_rep)},
#             "dtw_multi_sweep": multi_rows,
#         },
#     }
#     with open(os.path.join(res_dir, "summary.json"), "w", encoding="utf-8") as f:
#         json.dump(summary, f, ensure_ascii=False, indent=2)
#
#     print("Done. Outputs saved to:", os.path.abspath(out_root))
#
#
# if __name__ == "__main__":
#     main()




# -*- coding: utf-8 -*-
"""
CH3 Closed-loop pipeline (REALPACK)

This script is designed for thesis Chapter 3:
- Edge baselines: Softmax (LogReg), Linear SVM, Decision Tree (control)
- Offline baseline: 1D CNN on linear fixed-length resampling
- DTW enhancements:
    (1) DTW-ClsMin + CNN (single-template routing, constant 4 channels)
    (2) DTW-MultiTemplate(K) + CNN (final method; K selected by validation)
- Fairness control:
    Replication Control (channel replication to match MultiTemplate input channels)

Outputs:
- latex_out/figures/*.png : confusion matrices, ablation plots, motivation plots
- latex_out/tables/*.csv  : raw tables
- latex_out/tables/*_rows.tex : LaTeX row snippets for \input
- results/summary.json : machine-readable summary

Usage:
    python run_ch3_thesis_pipeline.py --mat PATH_TO_MAT --out_dir OUTDIR --seed 42

Notes:
- Uses the DTW implementations from dtw_cnn_handoff (numba accelerated).
- Patches numba/coverage compatibility for some environments.
"""
import os
import json
import time
import argparse
from pathlib import Path
import numpy as np
import warnings
# Silence sklearn deprecation spam in edge baselines (does not affect results)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*'multi_class' was deprecated.*",
)

# ---- numba/coverage compatibility patch (fixes "ModuleNotFoundError: coverage" and numba.core)
# dtw_cnn_handoff scripts import `coverage` only to access `coverage.types`. Some Windows venvs
# may not have `coverage` installed; we stub it to avoid crashing.
import sys as _sys
import types as _types

try:
    import coverage  # type: ignore
except Exception:  # pragma: no cover
    coverage = _types.ModuleType("coverage")  # type: ignore[assignment]
    coverage.types = _types.SimpleNamespace()  # type: ignore[attr-defined]
    _sys.modules["coverage"] = coverage  # type: ignore[assignment]

# Ensure expected typing aliases exist
try:
    if not hasattr(coverage, "types"):
        coverage.types = _types.SimpleNamespace()  # type: ignore[attr-defined]
    for _name in ["Tracer", "TTraceData", "TShouldTraceFn", "TFileDisposition", "TWarnFn", "TTraceFn"]:
        if not hasattr(coverage.types, _name):
            setattr(coverage.types, _name, object)
except Exception:
    pass

# Numba compatibility: some environments have numba but do not expose `numba.core`
try:
    import numba  # noqa
    if "numba.core" in _sys.modules and not hasattr(numba, "core"):
        numba.core = _sys.modules["numba.core"]  # type: ignore[attr-defined]
except Exception:
    pass

# Numba compatibility: ensure the package exposes `numba.core` as an attribute.
# In some environments, submodules exist in sys.modules but are not attached to the package.
try:
    import sys as _sys
    import numba as _numba
    import numba.core as _numba_core  # noqa: F401
    if not hasattr(_numba, "core"):
        _numba.core = _sys.modules.get("numba.core", _numba_core)
except Exception:
    pass


import scipy.io
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# Add local dtw modules path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DTW_CODE_DIR = os.path.join(_THIS_DIR, "dtw_cnn_handoff", "code")
import sys
if _DTW_CODE_DIR not in sys.path:
    sys.path.insert(0, _DTW_CODE_DIR)

import run_dtw_multi_sweep as M
import run_dtw_multi_quantile as MQ
import run_dtw_clsmin_sweep as C
import run_dtw_clsmin_dba as DBA

# Some environments import numba but do not expose `numba.core` on package attrs.
# This is required by numba internals in certain call paths.
try:
    import numba
    if "numba.core" in sys.modules and not hasattr(numba, "core"):
        numba.core = sys.modules["numba.core"]
except Exception:
    pass


# -----------------------------
# Utilities: split / metrics
# -----------------------------
def stratified_split(y, seed=42, train_ratio=0.70, val_ratio=0.15):
    rng = np.random.default_rng(seed)
    idx_tr, idx_va, idx_te = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(train_ratio * n))
        n_va = int(round(val_ratio * n))
        idx_tr.extend(idx[:n_tr])
        idx_va.extend(idx[n_tr:n_tr + n_va])
        idx_te.extend(idx[n_tr + n_va:])
    return np.array(idx_tr), np.array(idx_va), np.array(idx_te)


def macro_f1(y_true, y_pred, n_classes=3):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2.0 * tp) / denom)
    return float(np.mean(f1s))


def split_counts(y, idx):
    return np.bincount(y[idx], minlength=3)


# -----------------------------
# Data loading and preprocessing
# -----------------------------
def load_xyz_y_from_mat(mat_path: str):
    """Load ProcessedData/targetLength from .mat into Python list and labels.

    Returns
    -------
    xyz_list: list[np.ndarray], each (T_i, 3)
    y: np.ndarray, shape (N,), int64
    tlen: np.ndarray, shape (N,), int
    """
    mat = scipy.io.loadmat(mat_path)
    PD = mat["ProcessedData"]
    TL = mat["targetLength"]

    xyz_list = []
    y_list = []
    tlen_list = []
    for c in range(3):
        cell = PD[0, c]  # shape (1, n_c)
        tl = TL[0, c].reshape(-1)
        n_c = cell.shape[1]
        for i in range(n_c):
            arr = np.array(cell[0, i], dtype=np.float32)
            t = int(np.array(tl[i]).squeeze())
            xyz_list.append(arr[:t, :])
            y_list.append(c)
            tlen_list.append(t)

    y = np.array(y_list, dtype=np.int64)
    tlen = np.array(tlen_list, dtype=int)
    return xyz_list, y, tlen


def resample_linear(x: np.ndarray, L: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    T, D = x.shape
    if T == L:
        return x
    t0 = np.linspace(0.0, 1.0, T)
    t1 = np.linspace(0.0, 1.0, L)
    y = np.empty((L, D), np.float32)
    for d in range(D):
        y[:, d] = np.interp(t1, t0, x[:, d])
    return y


def make_4ch_seq(xyz: np.ndarray, L: int) -> np.ndarray:
    xr = resample_linear(xyz, L)  # (L,3)
    mag = np.sqrt(np.sum(xr ** 2, axis=1, keepdims=True))  # (L,1)
    out = np.concatenate([xr, mag], axis=1)  # (L,4)
    return out.T  # (4,L)


def build_edge_feature_matrix(xyz_list, L_feat: int) -> np.ndarray:
    """Edge feature: downsampled waveform (x,y,z,mag) concatenated (4*L_feat)."""
    N = len(xyz_list)
    X = np.zeros((N, 4 * L_feat), np.float32)
    for i, xyz in enumerate(xyz_list):
        xr = resample_linear(xyz, L_feat)  # (L_feat,3)
        mag = np.sqrt(np.sum(xr ** 2, axis=1, keepdims=False))  # (L_feat,)
        X[i] = np.concatenate([xr[:, 0], xr[:, 1], xr[:, 2], mag], axis=0)
    return X


def standardize_fit(X):
    mu = X.mean(axis=0, keepdims=True).astype(np.float32)
    sd = X.std(axis=0, keepdims=True).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def standardize_apply(X, mu, sd):
    return (X - mu) / sd


def compute_norm_stats_seq(X, idx_tr):
    mu = X[idx_tr].mean(axis=(0, 2), keepdims=True).astype(np.float32)
    sd = X[idx_tr].std(axis=(0, 2), keepdims=True).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def normalize_seq(X, mu, sd):
    return (X - mu) / sd


# -----------------------------
# CNN model
# -----------------------------
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleCNN(nn.Module):
    """Lightweight 1D CNN (same for baseline / DTW variants)."""

    def __init__(self, in_ch, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def predict_model(model, X, batch_size=256, device="cpu"):
    model.eval()
    ds = SeqDataset(X, np.zeros((X.shape[0],), dtype=np.int64))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device)
            logits = model(xb)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds)


def train_cnn(X, y, idx_tr, idx_va, idx_te,
              in_ch, seed=42, max_epochs=60, patience=10,
              lr=1e-3, weight_decay=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Training on device: {device}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)

    model = SimpleCNN(in_ch).to(device)

    # Loss: unweighted cross-entropy (keep consistent with baseline numbers)
    crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    dl_tr = DataLoader(SeqDataset(X[idx_tr], y[idx_tr]), batch_size=64, shuffle=True)
    dl_va = DataLoader(SeqDataset(X[idx_va], y[idx_va]), batch_size=256, shuffle=False)

    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    bad = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            optim.step()

        # validation
        model.eval()
        pv, tv = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                pv.append(torch.argmax(model(xb), dim=1).cpu().numpy())
                tv.append(yb.cpu().numpy())
        pv = np.concatenate(pv)
        tv = np.concatenate(tv)
        f1 = macro_f1(tv, pv, 3)

        if f1 > best_f1 + 1e-6:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final eval
    pred_va = predict_model(model, X[idx_va], device=device)
    pred_te = predict_model(model, X[idx_te], device=device)

    out = {
        "best_epoch": int(best_epoch),
        "val_acc": float(np.mean(pred_va == y[idx_va])),
        "val_f1": macro_f1(y[idx_va], pred_va, 3),
        "test_acc": float(np.mean(pred_te == y[idx_te])),
        "test_f1": macro_f1(y[idx_te], pred_te, 3),
        "y_pred_val": pred_va.astype(int).tolist(),
        "y_pred_test": pred_te.astype(int).tolist(),
    }
    return model, out


# -----------------------------
# Plotting
# -----------------------------
# def ensure_plot_style():
#     # Try a list of common CJK fonts (fallback to DejaVu)
#     plt.rcParams["font.family"] = [
#         "Noto Sans CJK JP",
#         "SimHei",
#         "Microsoft YaHei",
#         "PingFang SC",
#         "STSong",
#         "DejaVu Sans",
#     ]
#     plt.rcParams["axes.unicode_minus"] = False
#
def ensure_plot_style():
    import matplotlib.pyplot as plt
    import logging

    # Windows 常见中文字体优先；DejaVu 兜底
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 可选：彻底禁止 font_manager 的 “findfont” 刷屏（不影响绘图）
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
def _confusion_like_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "confusion_like",
        ["#f4d5ce", "#e6c7cf", "#8aa8cf", "#0072BD"],
        N=256,
    )


def _plot_confusion_block(ax, values, cmap, vmin, vmax, mode="count", square=True):
    arr = np.asarray(values, dtype=float)
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal" if square else "auto")
    ax.set_xlim(-0.5, arr.shape[1] - 0.5)
    ax.set_ylim(arr.shape[0] - 0.5, -0.5)
    ax.tick_params(length=0)

    for x in np.arange(-0.5, arr.shape[1], 1.0):
        ax.plot([x, x], [-0.5, arr.shape[0] - 0.5], color="black", linewidth=0.8)
    ax.plot([arr.shape[1] - 0.5, arr.shape[1] - 0.5], [-0.5, arr.shape[0] - 0.5], color="black", linewidth=0.8)
    for y in np.arange(-0.5, arr.shape[0], 1.0):
        ax.plot([-0.5, arr.shape[1] - 0.5], [y, y], color="black", linewidth=0.8)
    ax.plot([-0.5, arr.shape[1] - 0.5], [arr.shape[0] - 0.5, arr.shape[0] - 0.5], color="black", linewidth=0.8)

    thresh = vmin + 0.62 * (vmax - vmin) if vmax > vmin else vmin
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if mode == "percent":
                txt = f"{int(round(val))}%"
            elif mode == "float":
                txt = f"{val:.2f}"
            else:
                txt = str(int(round(val)))
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=15 if square else 12,
                color="white" if val >= thresh else "black",
            )


def plot_confusion(y_true, y_pred, labels_cn, title, out_path_png, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels_cn))))
    cm_disp = cm.astype(np.float32)
    if normalize:
        cm_disp = cm_disp / np.maximum(cm_disp.sum(axis=1, keepdims=True), 1.0)

    out_path_mat = out_path_png.replace(".png", "_data.mat")
    scipy.io.savemat(out_path_mat, {
        "cm": cm,
        "cm_normalized": cm_disp,
        "labels": np.array(labels_cn, dtype=object),
        "title": title,
    })

    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    tp = np.diag(cm)
    row_correct = np.where(row_sum > 0, np.round(100.0 * tp / row_sum), 0.0)
    col_correct = np.where(col_sum > 0, np.round(100.0 * tp / col_sum), 0.0)
    row_panel = np.column_stack([row_correct, 100.0 - row_correct])
    col_panel = np.vstack([col_correct, 100.0 - col_correct])

    cmap = _confusion_like_cmap()
    fig = plt.figure(figsize=(7.6, 6.2))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[3.0, 1.15],
        height_ratios=[3.0, 1.05],
        left=0.10,
        right=0.95,
        bottom=0.11,
        top=0.90,
        wspace=0.10,
        hspace=0.10,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_row = fig.add_subplot(gs[0, 1])
    ax_col = fig.add_subplot(gs[1, 0])

    if normalize:
        _plot_confusion_block(ax_main, cm_disp, cmap, 0.0, 1.0, mode="float", square=True)
    else:
        _plot_confusion_block(ax_main, cm, cmap, 0.0, float(cm.max()) if cm.size else 1.0, mode="count", square=True)
    _plot_confusion_block(ax_row, row_panel, cmap, 0.0, 100.0, mode="percent", square=False)
    _plot_confusion_block(ax_col, col_panel, cmap, 0.0, 100.0, mode="percent", square=False)

    ax_main.set_xticks([])
    ax_main.set_yticks(np.arange(len(labels_cn)))
    ax_main.set_yticklabels(labels_cn, fontsize=14)
    ax_main.set_ylabel("真实类", fontsize=15)

    ax_row.set_xticks([])
    ax_row.set_yticks([])

    ax_col.set_xticks(np.arange(len(labels_cn)))
    ax_col.set_xticklabels(labels_cn, fontsize=14)
    ax_col.set_yticks([])
    ax_col.set_xlabel("预测类", fontsize=15)

    fig.suptitle(title, fontsize=18, y=0.97)
    fig.savefig(out_path_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def zscore(a):
    a = np.asarray(a, dtype=np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)


def plot_motivation_short_long(mag_short, mag_long, fs, out_path_png):
    t_short = np.arange(len(mag_short)) / fs
    t_long = np.arange(len(mag_long)) / fs

    out_path_mat = out_path_png.replace(".png", "_data.mat")
    scipy.io.savemat(out_path_mat, {
        "t_short": t_short,
        "mag_short": mag_short,
        "t_long": t_long,
        "mag_long": mag_long,
        "fs": fs,
        "len_short": len(mag_short),
        "len_long": len(mag_long),
    })

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(t_short, mag_short, label=f"短事件（T={len(mag_short)}）")
    ax.plot(t_long, mag_long, label=f"长事件（T={len(mag_long)}）")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("模值幅度")
    ax.set_title("同一类别样本的时间伸缩现象（示例）")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_motivation_dtw_align_z(x_lin_4L, x_warp_L4, tpl_4L, out_path_png):
    mag_lin = zscore(x_lin_4L[3])
    mag_warp = zscore(x_warp_L4[:, 3])
    mag_tpl = zscore(tpl_4L[3])
    t = np.arange(len(mag_lin))

    out_path_mat = out_path_png.replace(".png", "_data.mat")
    scipy.io.savemat(out_path_mat, {
        "t": t,
        "mag_lin": mag_lin,
        "mag_warp": mag_warp,
        "mag_tpl": mag_tpl,
        "x_lin_4L": x_lin_4L,
        "x_warp_L4": x_warp_L4,
        "tpl_4L": tpl_4L,
    })

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(t, mag_lin, label="线性定长(原始输入)")
    ax.plot(t, mag_warp, label="DTW 对齐后")
    ax.plot(t, mag_tpl, label="参考模板", linestyle="--")
    ax.set_xlabel("采样点")
    ax.set_ylabel("模值（z-score）")
    ax.set_title("线性定长 vs DTW 对齐（归一化示例）")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_k_ablation(df_k, out_path_png):
    out_path_mat = out_path_png.replace(".png", "_data.mat")
    mat_payload = {
        "K": df_k["K"].values,
        "val_f1": df_k["val_f1"].values,
    }
    if "test_f1" in df_k.columns:
        mat_payload["test_f1"] = df_k["test_f1"].values
    scipy.io.savemat(out_path_mat, mat_payload)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(df_k["K"], df_k["val_f1"], marker="s", color="#1f77b4", linewidth=2.0, label="Val Macro-F1")
    ax.set_xlabel("模板数 K_MT（每类）")
    ax.set_ylabel("验证集 Macro-F1")
    ax.set_title("DTW 多模板数量消融")
    ax.set_xticks(df_k["K"])
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_edge_k_sweep(df_edge_k, best_k: int, out_path_png: str):
    out_path_mat = out_path_png.replace(".png", "_data.mat")
    scipy.io.savemat(out_path_mat, {
        "K": df_edge_k["K"].values,
        "val_f1": df_edge_k["val_f1"].values,
        "best_k": best_k,
    })

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(df_edge_k["K"], df_edge_k["val_f1"], marker="o")
    ax.axvline(best_k, linestyle="--")
    ax.set_xlabel("特征数 K")
    ax.set_ylabel("验证集 Macro-F1")
    ax.set_title("端侧特征维度选择（验证集）")
    ax.set_xticks(df_edge_k["K"])
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
def plot_replication(df_rep, out_path_png):
    out_path_mat = out_path_png.replace(".png", "_data.mat")
    scipy.io.savemat(out_path_mat, {
        "method": np.array(df_rep["method"].values, dtype=object),
        "test_f1": df_rep["test_f1"].values,
    })

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(df_rep))
    ax.bar(x, df_rep["test_f1"])
    ax.set_xticks(x)
    ax.set_xticklabels(df_rep["method"], rotation=12, ha="right")
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Replication Control 对比（K=4）")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# LaTeX row writer
# -----------------------------
def df_to_latex_rows(df, cols, float_fmt="{:.3f}"):
    lines = []
    for _, row in df.iterrows():
        items = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                items.append(float_fmt.format(float(v)))
            else:
                items.append(str(v))
        lines.append(" & ".join(items) + r" \\")
    return "\n".join(lines)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--mat", required=True, help="Path to REALPACK .mat file")
    parser.add_argument(
        "--mat",
        default=r"D:\xidian_Master\研究生论文\毕业论文\实验数据\第三章\数据\processedVehicleData_3class_REAL (2).mat",
        help="Path to REALPACK .mat file"
    )
    parser.add_argument("--out_dir", default="ch3_thesis_out", help="Output root dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--L", type=int, default=176, help="Fixed length for CNN")
    parser.add_argument("--L_feat", type=int, default=128, help="Edge feature length")
    parser.add_argument("--wR", type=float, default=0.15, help="DTW window ratio")
    parser.add_argument("--step", type=float, default=0.05, help="DTW step penalty")
    parser.add_argument("--dba_iter", type=int, default=5, help="DBA iterations for ClsMin-DBA")
    parser.add_argument("--K_list", default="1,2,3,4,5,6", help="Comma-separated K candidates")
    parser.add_argument("--max_epochs", type=int, default=30, help="max epochs for CNN training")
    parser.add_argument("--patience", type=int, default=6, help="early-stopping patience")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate for CNN")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Adam weight decay (L2) for CNN")
    parser.add_argument("--edge_k_min", type=int, default=1, help="Min K for edge feature sweep")
    parser.add_argument("--edge_k_max", type=int, default=20, help="Max K for edge feature sweep (<=0 means all)")
    parser.add_argument("--edge_only", action="store_true", help="Only run edge baselines and exit")

    args = parser.parse_args()
    train_kwargs = dict(
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    ensure_plot_style()

    out_root = args.out_dir
    fig_dir = os.path.join(out_root, "latex_out", "figures")
    tab_dir = os.path.join(out_root, "latex_out", "tables")
    res_dir = os.path.join(out_root, "results")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # ---- load data
    xyz_list, y, tlen = load_xyz_y_from_mat(args.mat)
    idx_tr, idx_va, idx_te = stratified_split(y, seed=args.seed)
    ytr, yva, yte = y[idx_tr], y[idx_va], y[idx_te]

    # ---- persist split indices (hard constraint: all methods share exactly the same split)
    split_npz = os.path.join(res_dir, f"split_indices_seed{args.seed}.npz")
    np.savez(split_npz, idx_tr=idx_tr, idx_va=idx_va, idx_te=idx_te)
    try:
        from scipy.io import savemat
        savemat(os.path.join(res_dir, f"split_indices_seed{args.seed}.mat"), {"idx_tr": idx_tr, "idx_va": idx_va, "idx_te": idx_te})
    except Exception as _e:
        print(f"[WARN] split .mat export skipped: {_e}")


    labels_cn = ["小型车", "中型车", "大型车"]

    # ---- dataset stats table
    stats_rows = []
    for c in range(3):
        tl = tlen[y == c]
        stats_rows.append({
            "class": c,
            "class_name": labels_cn[c],
            "count": int(len(tl)),
            "min": int(tl.min()),
            "median": int(np.percentile(tl, 50)),
            "max": int(tl.max()),
        })
    split_table = {
        "train": split_counts(y, idx_tr).tolist(),
        "val": split_counts(y, idx_va).tolist(),
        "test": split_counts(y, idx_te).tolist(),
    }

    # ---- Edge baselines (Softmax/SVM/DecisionTree, <=K features)
    print("[1/6] Edge baselines (Softmax/SVM/DT) with <=K features")
    # Hand-crafted features follow the MATLAB v4 definition (48 candidates),
    # then we select K<=7 via val Macro-F1 (epsilon-minK).
    sys.path.insert(0, str(Path(__file__).resolve().parent / "ch3_addons"))
    from run_ch3_edge_baselines_topk import build_feature_matrix as build_edge_X, run_edge_baselines
    X_edge = build_edge_X(xyz_list)  # (N,48)
    # edge_res = run_edge_baselines(X_edge, y, idx_tr, idx_va, idx_te, k_min=5, k_max=7, seed=args.seed)
    k_edge_min = max(1, int(args.edge_k_min))
    if int(args.edge_k_max) <= 0:
        k_edge_max = int(X_edge.shape[1])
    else:
        k_edge_max = min(int(args.edge_k_max), int(X_edge.shape[1]))

    edge_res = run_edge_baselines(
        X_edge, y, idx_tr, idx_va, idx_te,
        k_min=k_edge_min, k_max=k_edge_max,
        seed=args.seed
    )
    edge_metrics = edge_res.metrics
    (Path(res_dir) / "edge_selected_features.json").write_text(json.dumps(edge_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    edge_soft_acc = edge_metrics["softmax"]["acc"]
    edge_soft_f1  = edge_metrics["softmax"]["macro_f1"]
    edge_svm_acc  = edge_metrics["svm"]["acc"]
    edge_svm_f1   = edge_metrics["svm"]["macro_f1"]
    edge_dt_acc   = edge_metrics["dt"]["acc"]
    edge_dt_f1    = edge_metrics["dt"]["macro_f1"]
    cm_soft = np.asarray(edge_metrics["softmax"]["cm"], dtype=int)
    cm_svm  = np.asarray(edge_metrics["svm"]["cm"], dtype=int)
    cm_dt   = np.asarray(edge_metrics["dt"]["cm"], dtype=int)
    selected_feat_names = edge_metrics["selected_names"]
    # Pack into the same dict format used later for CSV/LaTeX export
    K_edge = int(edge_metrics["feature_k"])
    edge_soft = {"method": f"端侧 Softmax（K={K_edge}）", "test_acc": edge_soft_acc, "test_f1": edge_soft_f1, "cm": cm_soft.tolist()}
    edge_svm  = {"method": f"端侧 SVM（K={K_edge}）", "test_acc": edge_svm_acc,  "test_f1": edge_svm_f1,  "cm": cm_svm.tolist()}
    edge_dt   = {"method": f"端侧 DT（K={K_edge}）", "test_acc": edge_dt_acc,   "test_f1": edge_dt_f1,   "cm": cm_dt.tolist()}
    print("  Edge selected K={}, features={}".format(edge_metrics["feature_k"], selected_feat_names))

    # ---- Edge only mode: print results and exit
    if args.edge_only:
        df_edge_k = pd.DataFrame({
            "K": edge_metrics.get("k_grid", []),
            "val_f1": edge_metrics.get("val_macroF1_curve", []),
        })
        if len(df_edge_k) > 0:
            plot_edge_k_sweep(df_edge_k, K_edge, os.path.join(fig_dir, "ablation_edge_K.png"))
            print(f"Edge K ablation plot saved to: {os.path.join(fig_dir, 'ablation_edge_K.png')}")

        print("\n" + "="*60)
        print("Edge Baselines Results (edge_only mode)")
        print("="*60)
        print(f"Selected K: {K_edge}")
        print(f"Selected features: {selected_feat_names}")
        print("-"*60)
        print(f"Softmax: Test Acc={edge_soft_acc:.4f}, Test Macro-F1={edge_soft_f1:.4f}")
        print(f"SVM:     Test Acc={edge_svm_acc:.4f}, Test Macro-F1={edge_svm_f1:.4f}")
        print(f"DT:      Test Acc={edge_dt_acc:.4f}, Test Macro-F1={edge_dt_f1:.4f}")
        print("-"*60)
        if len(df_edge_k) > 0:
            print("K vs Val Macro-F1:")
            for _, row in df_edge_k.iterrows():
                print(f"  K={int(row['K']):2d}: Val F1={row['val_f1']:.4f}")
            print("-"*60)
        print("Confusion Matrices:")
        print(f"Softmax CM:\n{cm_soft}")
        print(f"SVM CM:\n{cm_svm}")
        print(f"DT CM:\n{cm_dt}")
        print("="*60)
        print(f"Results saved to: {res_dir}")
        return

    # ---- Baseline CNN (linear fixed length)
    print("[2/6] Baseline CNN (linear fixed length)")
    X_raw = M.build_baseline_data(xyz_list, args.L)
    mu_b, sd_b = compute_norm_stats_seq(X_raw, idx_tr)
    Xn = normalize_seq(X_raw, mu_b, sd_b)

    t0 = time.time()
    _, cnn_base = train_cnn(Xn, y, idx_tr, idx_va, idx_te, in_ch=4, seed=args.seed, **train_kwargs)
    t_base = time.time() - t0

    # Baseline confusion matrix
    plot_confusion(yte, np.array(cnn_base["y_pred_test"]), labels_cn,
                   "离线混淆矩阵（线性定长CNN）",
                   os.path.join(fig_dir, "cm_cnn_baseline.png"), normalize=False)

    # ---- DTW-ClsMin + CNN (single-template routing)
    print("[3/6] DTW-ClsMin + CNN (single-template routing)")
    # NOTE: We keep BOTH variants for a clean ablation:
    #   - ClsMin-Mean: mean template (ablation)
    #   - ClsMin-DBA : DBA template (recommended in thesis; DTW barycenter)
    # Convert wR to window length w (for demo plots)
    w = int(round(args.wR * args.L))

    # (A1) Mean-template ClsMin (ablation)
    X_clsmin_mean = C.build_dtw_clsmin(X_raw, y, idx_tr, wR=args.wR, step=args.step)
    mu_cm, sd_cm = compute_norm_stats_seq(X_clsmin_mean, idx_tr)
    X_clsmin_mean_n = normalize_seq(X_clsmin_mean, mu_cm, sd_cm)
    t0 = time.time()
    _, cnn_clsmin_mean = train_cnn(X_clsmin_mean_n, y, idx_tr, idx_va, idx_te, in_ch=4, seed=args.seed, **train_kwargs)
    t_clsmin_mean = time.time() - t0

    # (O2) DBA-template ClsMin (main / key evidence)
    X_clsmin_dba = DBA.build_dtw_clsmin_dba(X_raw, y, idx_tr, wR=args.wR, step=args.step, n_iter=args.dba_iter)
    mu_cd, sd_cd = compute_norm_stats_seq(X_clsmin_dba, idx_tr)
    X_clsmin_dba_n = normalize_seq(X_clsmin_dba, mu_cd, sd_cd)
    t0 = time.time()
    _, cnn_clsmin_dba = train_cnn(X_clsmin_dba_n, y, idx_tr, idx_va, idx_te, in_ch=4, seed=args.seed, **train_kwargs)
    t_clsmin_dba = time.time() - t0

    plot_confusion(yte, np.array(cnn_clsmin_dba["y_pred_test"]), labels_cn,
                   "离线混淆矩阵（DTW-ClsMin-DBA + CNN）",
                   os.path.join(fig_dir, "cm_dtw_clsmin_cnn.png"), normalize=False)

    # ---- DTW MultiTemplate sweep over K candidates
    print("[4/6] DTW MultiTemplate sweep over K candidates")
    K_list = [int(s) for s in args.K_list.split(",") if s.strip()]

    multi_rows = []
    best_multi = None

    for i_K, K in enumerate(K_list):
        print(f"  [4/6] Processing K={K} ({i_K+1}/{len(K_list)})")
        t_build0 = time.time()
        X_multi, _templates = MQ.build_dtw_multi_quantile(
            X_raw, y, idx_tr, tlen, templates_per_class=K, wR=args.wR, step=args.step
        )
        t_build = time.time() - t_build0

        mu_m, sd_m = compute_norm_stats_seq(X_multi, idx_tr)
        X_multi_n = normalize_seq(X_multi, mu_m, sd_m)

        t_train0 = time.time()
        _, cnn_multi = train_cnn(X_multi_n, y, idx_tr, idx_va, idx_te,
                                 in_ch=X_multi_n.shape[1], seed=args.seed, **train_kwargs)
        t_train = time.time() - t_train0

        row = {
            "K": int(K),
            "in_ch": int(X_multi.shape[1]),
            "build_s": float(t_build),
            "train_s": float(t_train),
            "val_acc": float(cnn_multi["val_acc"]),
            "val_f1": float(cnn_multi["val_f1"]),
            "test_acc": float(cnn_multi["test_acc"]),
            "test_f1": float(cnn_multi["test_f1"]),
        }
        multi_rows.append(row)

        if best_multi is None or row["val_f1"] > best_multi["row"]["val_f1"] + 1e-6:
            best_multi = {
                "K": K,
                "X_multi": X_multi,
                "cnn": cnn_multi,
                "row": row,
            }

    # best MultiTemplate
    K_star = int(best_multi["K"])
    cnn_multi_star = best_multi["cnn"]
    X_multi_star = best_multi["X_multi"]
    plot_confusion(yte, np.array(cnn_multi_star["y_pred_test"]), labels_cn,
                   f"离线混淆矩阵（DTW-MultiTemplate(K={K_star}) + CNN）",
                   os.path.join(fig_dir, "cm_dtw_multi_cnn.png"), normalize=False)

    # ---- Replication control (match channel count to best multi)
    print("[5/6] Replication control (match channel count to best multi)")
    rep_factor = X_multi_star.shape[1] // X_raw.shape[1]
    X_rep = np.tile(X_raw, (1, rep_factor, 1))
    mu_r, sd_r = compute_norm_stats_seq(X_rep, idx_tr)
    X_rep_n = normalize_seq(X_rep, mu_r, sd_r)

    t0 = time.time()
    _, cnn_rep = train_cnn(X_rep_n, y, idx_tr, idx_va, idx_te,
                           in_ch=X_rep_n.shape[1], seed=args.seed, **train_kwargs)
    t_rep = time.time() - t0

    plot_confusion(yte, np.array(cnn_rep["y_pred_test"]), labels_cn,
                   f"离线混淆矩阵（Replication Control, K={K_star})",
                   os.path.join(fig_dir, "cm_replication_control.png"), normalize=False)

    # ---- Motivation figures
    print("[6/6] Generating motivation figures and tables")
    fs = 50.0
    # pick class 0 short/long
    idx_c0 = np.where(y == 0)[0]
    tl0 = tlen[idx_c0]
    i_short = int(idx_c0[np.argmin(tl0)])
    i_long = int(idx_c0[np.argmax(tl0)])

    mag_short = np.sqrt(np.sum(xyz_list[i_short] ** 2, axis=1))
    mag_long = np.sqrt(np.sum(xyz_list[i_long] ** 2, axis=1))
    plot_motivation_short_long(
        mag_short, mag_long, fs,
        os.path.join(fig_dir, "fig_motivation_speed.png")
    )

    # DTW alignment demo: pick a test sample of class 1 if exists
    idx_test_c1 = idx_te[y[idx_te] == 1]
    if len(idx_test_c1) > 0:
        i_ex = int(idx_test_c1[0])
        cls = int(y[i_ex])
        tpl = X_raw[idx_tr][y[idx_tr] == cls].mean(axis=0)  # (4,L)
        x_lin = X_raw[i_ex]  # (4,L)
        x_warp = C.dtw_warp_mv(x_lin.T, tpl.T, w, args.step)  # (L,4)
        plot_motivation_dtw_align_z(
            x_lin, x_warp, tpl,
            os.path.join(fig_dir, "fig_motivation_dtw_align_z.png")
        )

    # ---- Build tables (csv + LaTeX rows)

    df_stats = pd.DataFrame(stats_rows)
    df_split = pd.DataFrame([
        {"split": "train", "小型车": split_table["train"][0], "中型车": split_table["train"][1], "大型车": split_table["train"][2]},
        {"split": "val",   "小型车": split_table["val"][0],   "中型车": split_table["val"][1],   "大型车": split_table["val"][2]},
        {"split": "test",  "小型车": split_table["test"][0],  "中型车": split_table["test"][1],  "大型车": split_table["test"][2]},
    ])
    df_edge_k = pd.DataFrame({
        "K": edge_metrics.get("k_grid", []),
        "val_f1": edge_metrics.get("val_macroF1_curve", []),
    })

    df_edge = pd.DataFrame([edge_soft, edge_svm, edge_dt])

    # Main table (recommended for thesis): 4 rows (clean narrative)
    df_overall = pd.DataFrame([
        {"method": "端侧 Softmax（部署）", "test_acc": edge_soft["test_acc"], "test_f1": edge_soft["test_f1"]},
        {"method": "端侧 SVM（对照）", "test_acc": edge_svm["test_acc"], "test_f1": edge_svm["test_f1"]},
        {"method": "离线 CNN（线性定长）", "test_acc": cnn_base["test_acc"], "test_f1": cnn_base["test_f1"]},
        {"method": f"DTW-MultiTemplate(K={K_star}) + CNN（本文方法）", "test_acc": cnn_multi_star["test_acc"], "test_f1": cnn_multi_star["test_f1"]},
    ])

    # ClsMin template ablation: Mean vs DBA (single-row evidence, not in main table)
    df_clsmin_tpl = pd.DataFrame([
        {"method": "DTW-ClsMin-Mean + CNN（消融）", "test_acc": cnn_clsmin_mean["test_acc"], "test_f1": cnn_clsmin_mean["test_f1"]},
        {"method": f"DTW-ClsMin-DBA + CNN（消融，iter={args.dba_iter})", "test_acc": cnn_clsmin_dba["test_acc"], "test_f1": cnn_clsmin_dba["test_f1"]},
    ])

    df_k = pd.DataFrame(multi_rows).sort_values("K").reset_index(drop=True)

    df_rep = pd.DataFrame([
        {"method": f"Replication Control（通道复制，K={K_star}）", "test_acc": cnn_rep["test_acc"], "test_f1": cnn_rep["test_f1"]},
        {"method": f"DTW-MultiTemplate（K={K_star}）", "test_acc": cnn_multi_star["test_acc"], "test_f1": cnn_multi_star["test_f1"]},
    ])

    # Save CSV
    df_stats.to_csv(os.path.join(tab_dir, "ch3_dataset_stats.csv"), index=False)
    df_split.to_csv(os.path.join(tab_dir, "ch3_dataset_split_counts.csv"), index=False)
    df_edge.to_csv(os.path.join(tab_dir, "ch3_edge_baselines.csv"), index=False)
    df_overall.to_csv(os.path.join(tab_dir, "ch3_overall.csv"), index=False)
    df_k.to_csv(os.path.join(tab_dir, "ch3_ablation_K.csv"), index=False)
    df_rep.to_csv(os.path.join(tab_dir, "ch3_ablation_replication.csv"), index=False)
    df_clsmin_tpl.to_csv(os.path.join(tab_dir, "ch3_ablation_clsmin_template.csv"), index=False)
    df_edge_k.to_csv(os.path.join(tab_dir, "ch3_edge_k_sweep.csv"), index=False)

    # Save LaTeX rows
    with open(os.path.join(tab_dir, "ch3_dataset_stats_rows.tex"), "w", encoding="utf-8") as f:
        f.write(df_to_latex_rows(df_stats, ["class_name", "count", "min", "median", "max"], float_fmt="{:.0f}"))
    with open(os.path.join(tab_dir, "ch3_dataset_split_rows.tex"), "w", encoding="utf-8") as f:
        f.write(df_to_latex_rows(df_split, ["split", "小型车", "中型车", "大型车"], float_fmt="{:.0f}"))
    with open(os.path.join(tab_dir, "ch3_edge_rows.tex"), "w", encoding="utf-8") as f:
        f.write(df_to_latex_rows(df_edge, ["method", "test_acc", "test_f1"]))
    with open(os.path.join(tab_dir, "ch3_overall_rows.tex"), "w", encoding="utf-8") as f:
        f.write(df_to_latex_rows(df_overall, ["method", "test_acc", "test_f1"]))
    with open(os.path.join(tab_dir, "ch3_ablation_K_rows.tex"), "w", encoding="utf-8") as f:
        f.write(df_to_latex_rows(df_k, ["K", "in_ch", "test_acc", "test_f1"]))
    with open(os.path.join(tab_dir, "ch3_ablation_rep_rows.tex"), "w", encoding="utf-8") as f:
        f.write(df_to_latex_rows(df_rep, ["method", "test_acc", "test_f1"]))
    with open(os.path.join(tab_dir, "ch3_edge_k_sweep_rows.tex"), "w", encoding="utf-8") as f:
        f.write(df_to_latex_rows(df_edge_k, ["K", "val_f1"]))

    with open(os.path.join(tab_dir, "ch3_ablation_clsmin_template_rows.tex"), "w", encoding="utf-8") as f:
        f.write(df_to_latex_rows(df_clsmin_tpl, ["method", "test_acc", "test_f1"]))

    # Plots for ablations
    plot_k_ablation(df_k, os.path.join(fig_dir, "ablation_K.png"))
    plot_replication(df_rep, os.path.join(fig_dir, "ablation_replication.png"))
    if len(df_edge_k) > 0:
        plot_edge_k_sweep(df_edge_k, best_k=K_edge, out_path_png=os.path.join(fig_dir, "ablation_edge_K.png"))

    # ---- Save summary JSON
    summary = {
        "seed": int(args.seed),
        "L": int(args.L),
        "L_feat": int(args.L_feat),
        "dtw": {"wR": float(args.wR), "step": float(args.step), "dba_iter": int(args.dba_iter)},
        "split_counts": split_table,
        "edge": {"softmax": edge_soft, "svm": edge_svm, "dt": edge_dt},
        "offline": {
            "cnn_baseline": {"metrics": cnn_base, "train_time_s": float(t_base)},
            "dtw_clsmin_mean_cnn": {"metrics": cnn_clsmin_mean, "train_time_s": float(t_clsmin_mean)},
            "dtw_clsmin_dba_cnn": {"metrics": cnn_clsmin_dba, "train_time_s": float(t_clsmin_dba)},
            "dtw_multi_best": {"K": K_star, "metrics": cnn_multi_star},
            "replication_control": {"K": K_star, "metrics": cnn_rep, "train_time_s": float(t_rep)},
            "dtw_multi_sweep": multi_rows,
        },
    }
    with open(os.path.join(res_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done. Outputs saved to:", os.path.abspath(out_root))


if __name__ == "__main__":
    main()
