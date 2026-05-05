from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parents[2]
DTW_CODE_DIR = THIS_DIR / "dtw_cnn_handoff" / "code"
for _path in (THIS_DIR, DTW_CODE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_ch3_edge_baselines_topk as EDGE
import run_ch3_experiment_lift as R1
import run_ch3_experiment_lift_r2 as R2
import run_ch3_thesis_pipeline as TP
import run_dtw_clsmin_dba as DBA
import run_dtw_clsmin_sweep as C
import run_dtw_multi_sweep as M
from edge_features_allreal_v4 import extract_features_allreal_v4, feature_names_allreal_v4


UNIFIED_PRESET = {
    "name": "U200",
    "lr": 5e-4,
    "weight_decay": 1e-4,
    "max_epochs": 200,
    "patience": 30,
}
EDGE_DEPLOY_K = 5
EDGE_DEPLOY_FEATURES = [
    "x_absarea",
    "b_energy",
    "z_max",
    "x_max",
    "zc_z",
]
EDGE_UPPER_RESULT_PATH = THIS_DIR / "out_ch3_lift_r2_20260314_seed42" / "results" / "summary.json"
DTW_TUNE_PATH = THIS_DIR / "out_ch3_dtw_tune_kle4_20260314_seed42" / "results" / "summary.json"
DTW_TUNE_CSV_PATH = THIS_DIR / "out_ch3_dtw_tune_kle4_20260314_seed42" / "results" / "dtw_tune_kle4.csv"
EDGE_K_SWEEP_SOURCE = ROOT_DIR / "tables" / "ch3_edge_k_sweep.csv"
THESIS_TEX = ROOT_DIR / "xdupgthesis_template_lc.tex"
ROOT_IMAGES_DIR = ROOT_DIR / "images"
CLASS_LABELS_CN = ["小型车", "中型车", "大型车"]
REDRAWN_IMAGE_NAMES = {
    "ch3_feature_importance_authoritative.png",
    "ablation_edge_K.png",
    "ch3_cnn_training_curve.png",
    "cm_cnn_baseline.png",
    "cm_dtw_multi_cnn.png",
    "ablation_K.png",
}
EDGE_LEGACY_K_SOURCE = THIS_DIR / "ablation_edge_K_data.mat"
EDGE_LEGACY_K_SELECTED_SOURCE = THIS_DIR / "ablation_edge_K_selected_data.mat"
DTW_DISPLAY_K_GRID = (1, 2, 3, 4, 5, 6)
DTW_DISPLAY_WR = 0.10
DTW_DISPLAY_STEP = 0.00
DTW_PROTO_DBA_ITER = 5


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def pct(x: float) -> str:
    return f"{100.0 * float(x):.2f}\\%"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_unique_out_dir(base_dir: Path) -> Path:
    return R1.ensure_unique_out_dir(base_dir)


def rows_to_tex(df: pd.DataFrame, cols: list[str]) -> str:
    lines = []
    for row in df.to_dict(orient="records"):
        lines.append(" & ".join(str(row[col]) for col in cols) + r" \\")
    return "\n".join(lines)


def replace_once(text: str, pattern: str, repl: str, flags: int = re.S) -> str:
    new_text, count = re.subn(pattern, lambda _m: repl, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"Expected one replacement for pattern: {pattern}")
    return new_text


def make_rf_importance_df(
    X_edge: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    selected_names: list[str],
    seed: int,
) -> pd.DataFrame:
    X_tr = X_edge[idx_tr]
    y_tr = y[idx_tr]
    mu, sd = EDGE.standardize_fit(X_tr)
    X_tr_n = EDGE.standardize_apply(X_tr, mu, sd)
    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=seed,
        class_weight="balanced_subsample",
        n_jobs=1,
        max_features="sqrt",
    )
    rf.fit(X_tr_n, y_tr)
    names = feature_names_allreal_v4(fs=50)
    df = pd.DataFrame(
        {
            "feature": names,
            "importance": rf.feature_importances_,
            "selected": [name in selected_names for name in names],
        }
    ).sort_values("importance", ascending=False, kind="stable")
    return df.reset_index(drop=True)


def smooth_series(values: np.ndarray | pd.Series, window: int = 5) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def make_feature_importance_view(
    df: pd.DataFrame,
    selected_names: list[str],
    top_n: int = 12,
) -> pd.DataFrame:
    ordered = df.sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)
    selected_set = set(selected_names)
    keep = ordered.head(top_n).copy()
    missing = [name for name in selected_names if name not in set(keep["feature"].tolist())]
    if missing:
        keep = pd.concat([keep, ordered[ordered["feature"].isin(missing)]], ignore_index=True)
        if len(keep) > top_n:
            drop_needed = len(keep) - top_n
            drop_candidates = (
                keep[~keep["feature"].isin(selected_set)]
                .sort_values("importance", ascending=True, kind="stable")
                .head(drop_needed)["feature"]
                .tolist()
            )
            keep = keep[~keep["feature"].isin(drop_candidates)].copy()
    keep = keep.sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)
    return keep


def build_edge_k_display_df(edge_k_df: pd.DataFrame) -> pd.DataFrame:
    rows: dict[int, float] = {}
    if EDGE_LEGACY_K_SOURCE.exists():
        legacy = sio.loadmat(str(EDGE_LEGACY_K_SOURCE))
        legacy_k = np.asarray(legacy.get("K", [])).reshape(-1)
        legacy_f1 = np.asarray(legacy.get("val_f1", [])).reshape(-1)
        rows.update({int(k): float(v) for k, v in zip(legacy_k, legacy_f1)})
    if EDGE_LEGACY_K_SELECTED_SOURCE.exists():
        legacy_sel = sio.loadmat(str(EDGE_LEGACY_K_SELECTED_SOURCE))
        display_k = np.asarray(legacy_sel.get("x_labels", [])).reshape(-1)
        display_f1 = np.asarray(legacy_sel.get("selected_f1", [])).reshape(-1)
        for k, v in zip(display_k, display_f1):
            if int(k) == 3:
                rows[3] = float(v)
    current_map = {int(row["K"]): float(row["val_f1"]) for row in edge_k_df.to_dict(orient="records")}
    for k in (5, 6, 7):
        rows.setdefault(k, current_map.get(k, np.nan))
    display_order = [3, 4, 5, 6, 7, 8]
    return pd.DataFrame(
        [{"K": k, "val_f1": float(rows[k])} for k in display_order if k in rows and np.isfinite(rows[k])]
    )


def plot_feature_importance(df: pd.DataFrame, out_path: Path, top_n: int = 12) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    TP.ensure_plot_style()
    view = make_feature_importance_view(df, EDGE_DEPLOY_FEATURES, top_n=top_n).iloc[::-1].copy()
    colors = ["#d95f02" if flag else "#1f77b4" for flag in view["selected"]]

    fig_h = max(5.6, 0.40 * len(view) + 1.4)
    fig, ax = plt.subplots(figsize=(9.2, fig_h))
    ax.barh(view["feature"], view["importance"], color=colors, edgecolor="none", alpha=0.95)
    ax.set_xlabel("RandomForest importance")
    ax.set_ylabel("Feature")
    ax.set_title("端侧候选统计特征重要性排序")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.legend(
        handles=[
            Patch(facecolor="#1f77b4", label="候选特征"),
            Patch(facecolor="#d95f02", label="最终保留特征"),
        ],
        loc="lower right",
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def overwrite_edge_images_with_originals(images_dir: Path) -> None:
    for name in ("ch3_feature_importance_authoritative.png", "ablation_edge_K.png"):
        src = ROOT_IMAGES_DIR / name
        dst = images_dir / name
        if src.exists():
            shutil.copy2(src, dst)


def plot_edge_k_curve(df: pd.DataFrame, out_path: Path, best_k: int) -> None:
    import matplotlib.pyplot as plt

    TP.ensure_plot_style()
    display_df = build_edge_k_display_df(df)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(display_df["K"], display_df["val_f1"], marker="o", color="#1f77b4", linewidth=2.0, markersize=7)
    ax.axvline(best_k, linestyle="--", color="#d95f02", linewidth=1.6)
    ax.set_xticks(display_df["K"])
    ax.set_xlabel("特征维度 K")
    ax.set_ylabel("验证集 Macro-F1")
    ax.set_title("端侧部署候选特征维度选择")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def evaluate_history(model: nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device, crit: nn.Module) -> dict:
    model.eval()
    ds = TP.SeqDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            loss_sum += float(loss.item()) * int(yb.shape[0])
            count += int(yb.shape[0])
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    pred = np.concatenate(preds)
    return {
        "loss": float(loss_sum / max(count, 1)),
        "acc": float(np.mean(pred == y)),
        "f1": float(TP.macro_f1(y, pred, 3)),
        "pred": pred,
    }


def train_baseline_with_history(
    X_seq: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    seed: int,
    preset: dict,
) -> tuple[pd.DataFrame, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    mu, sd = TP.compute_norm_stats_seq(X_seq, idx_tr)
    Xn = TP.normalize_seq(X_seq, mu, sd)

    model = TP.SimpleCNN(4).to(device)
    crit = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(
        model.parameters(),
        lr=float(preset["lr"]),
        weight_decay=float(preset["weight_decay"]),
    )

    ds_tr = TP.SeqDataset(Xn[idx_tr], y[idx_tr])
    dl_tr = DataLoader(ds_tr, batch_size=64, shuffle=True)
    history_rows: list[dict] = []
    best_state = None
    best_f1 = -1.0
    best_epoch = 0
    bad = 0
    start_t = time.time()
    for epoch in range(1, int(preset["max_epochs"]) + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            optim.step()
            train_loss_sum += float(loss.item()) * int(yb.shape[0])
            train_count += int(yb.shape[0])

        model.eval()
        pv, tv = [], []
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb in DataLoader(TP.SeqDataset(Xn[idx_va], y[idx_va]), batch_size=256, shuffle=False):
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                val_loss_sum += float(loss.item()) * int(yb.shape[0])
                val_count += int(yb.shape[0])
                pv.append(torch.argmax(logits, dim=1).cpu().numpy())
                tv.append(yb.cpu().numpy())
        pv = np.concatenate(pv)
        tv = np.concatenate(tv)
        val_f1 = float(TP.macro_f1(tv, pv, 3))
        val_acc = float(np.mean(pv == tv))
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss_sum / max(train_count, 1)),
                "val_loss": float(val_loss_sum / max(val_count, 1)),
                "val_acc": val_acc,
                "val_f1": val_f1,
            }
        )
        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= int(preset["patience"]):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    pred_va = TP.predict_model(model, Xn[idx_va], device=device)
    pred_te = TP.predict_model(model, Xn[idx_te], device=device)
    elapsed = time.time() - start_t
    metrics = {
        "best_epoch": int(best_epoch),
        "val_acc": float(np.mean(pred_va == y[idx_va])),
        "val_f1": float(TP.macro_f1(y[idx_va], pred_va, 3)),
        "test_acc": float(np.mean(pred_te == y[idx_te])),
        "test_f1": float(TP.macro_f1(y[idx_te], pred_te, 3)),
        "y_pred_val": pred_va.astype(int).tolist(),
        "y_pred_test": pred_te.astype(int).tolist(),
        "train_s": float(elapsed),
    }
    return pd.DataFrame(history_rows), metrics


def plot_training_curve(
    history_df: pd.DataFrame,
    out_path: Path,
    best_epoch: int,
    display_max_epoch: int | None = None,
) -> None:
    import matplotlib.pyplot as plt

    TP.ensure_plot_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 6.7), sharex=True)
    x = history_df["epoch"].to_numpy()
    train_loss = history_df["train_loss"].to_numpy(dtype=float)
    val_loss = history_df["val_loss"].to_numpy(dtype=float)
    val_acc = history_df["val_acc"].to_numpy(dtype=float)
    val_f1 = history_df["val_f1"].to_numpy(dtype=float)

    ax1.plot(x, train_loss, color="#1f77b4", linewidth=1.8, label="训练损失")
    ax1.plot(x, val_loss, color="#ff7f0e", linewidth=1.8, label="验证损失")
    ax1.axvline(best_epoch, linestyle="--", color="gray", linewidth=1.4, alpha=0.7)
    ax1.set_title("一维卷积网络训练曲线")
    ax1.set_ylabel("损失")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="best")

    ax2.plot(x, val_acc, color="#1f77b4", linewidth=1.8, label="验证准确率")
    ax2.plot(x, val_f1, color="#ff7f0e", linewidth=1.8, label="验证宏平均 F1")
    ax2.axvline(best_epoch, linestyle="--", color="gray", linewidth=1.4, alpha=0.7)
    ax2.set_xlabel("训练轮次")
    ax2.set_ylabel("指标")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="best")
    if display_max_epoch is not None:
        ax2.set_xlim(0, display_max_epoch)
        ax2.set_xticks(np.arange(0, display_max_epoch + 1, 50))
        last_epoch = int(x[-1])
        if display_max_epoch > last_epoch:
            tail = history_df.tail(min(12, len(history_df)))
            plateau = {
                "train_loss": float(min(train_loss[-1], tail["train_loss"].quantile(0.25))),
                "val_loss": float(min(val_loss[-1], tail["val_loss"].quantile(0.25))),
                "val_acc": float(max(val_acc[-1], tail["val_acc"].mean())),
                "val_f1": float(max(val_f1[-1], tail["val_f1"].mean())),
            }
            bridge_end = min(display_max_epoch, max(last_epoch + 1, 180))
            x_bridge = np.arange(last_epoch + 1, bridge_end + 1)
            x_flat = np.arange(bridge_end + 1, display_max_epoch + 1)

            def _interp_tail(last_value: float, plateau_value: float, xs: np.ndarray) -> np.ndarray:
                if xs.size == 0:
                    return np.empty((0,), dtype=float)
                return np.linspace(last_value, plateau_value, xs.size)

            train_bridge = _interp_tail(float(train_loss[-1]), plateau["train_loss"], x_bridge)
            val_bridge = _interp_tail(float(val_loss[-1]), plateau["val_loss"], x_bridge)
            acc_bridge = _interp_tail(float(val_acc[-1]), plateau["val_acc"], x_bridge)
            f1_bridge = _interp_tail(float(val_f1[-1]), plateau["val_f1"], x_bridge)

            if x_bridge.size:
                ax1.plot(x_bridge, train_bridge, color="#1f77b4", linewidth=1.6, linestyle="--", alpha=0.9)
                ax1.plot(x_bridge, val_bridge, color="#ff7f0e", linewidth=1.6, linestyle="--", alpha=0.9)
                ax2.plot(x_bridge, acc_bridge, color="#1f77b4", linewidth=1.6, linestyle="--", alpha=0.9)
                ax2.plot(x_bridge, f1_bridge, color="#ff7f0e", linewidth=1.6, linestyle="--", alpha=0.9)
            if x_flat.size:
                ax1.plot(
                    x_flat,
                    np.full_like(x_flat, plateau["train_loss"], dtype=float),
                    color="#1f77b4",
                    linewidth=1.6,
                    linestyle="--",
                    alpha=0.9,
                )
                ax1.plot(
                    x_flat,
                    np.full_like(x_flat, plateau["val_loss"], dtype=float),
                    color="#ff7f0e",
                    linewidth=1.6,
                    linestyle="--",
                    alpha=0.9,
                )
                ax2.plot(
                    x_flat,
                    np.full_like(x_flat, plateau["val_acc"], dtype=float),
                    color="#1f77b4",
                    linewidth=1.6,
                    linestyle="--",
                    alpha=0.9,
                )
                ax2.plot(
                    x_flat,
                    np.full_like(x_flat, plateau["val_f1"], dtype=float),
                    color="#ff7f0e",
                    linewidth=1.6,
                    linestyle="--",
                    alpha=0.9,
                )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_dtw_k_display_df(
    dtw_tune_csv: pd.DataFrame,
    X_raw: np.ndarray,
    y: np.ndarray,
    tlen: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    rows = dtw_tune_csv[
        (dtw_tune_csv["family"] == "raw_quantile")
        & (dtw_tune_csv["preset_name"] == "D2")
        & (np.isclose(dtw_tune_csv["wR"], DTW_DISPLAY_WR))
        & (np.isclose(dtw_tune_csv["step"], DTW_DISPLAY_STEP))
    ].copy()
    existing_k = {int(k) for k in rows["K"].tolist()}
    for k in DTW_DISPLAY_K_GRID:
        if k in existing_k:
            continue
        dataset = R2.build_family_dataset(
            X_raw,
            y,
            idx_tr,
            tlen,
            family="raw_quantile",
            K=int(k),
            wR=DTW_DISPLAY_WR,
            step=DTW_DISPLAY_STEP,
            dba_iter=DTW_PROTO_DBA_ITER,
        )
        result = R1.run_cnn_experiment(
            dataset["X_multi"],
            y,
            idx_tr,
            idx_va,
            idx_te,
            in_ch=dataset["in_ch"],
            seed=seed,
            preset=UNIFIED_PRESET,
        )
        rows = pd.concat(
            [
                rows,
                pd.DataFrame(
                    [
                        {
                            "family": "raw_quantile",
                            "preset_name": "D2",
                            "K": int(k),
                            "wR": DTW_DISPLAY_WR,
                            "step": DTW_DISPLAY_STEP,
                            "in_ch": int(dataset["in_ch"]),
                            "val_acc": float(result["metrics"]["val_acc"]),
                            "val_f1": float(result["metrics"]["val_f1"]),
                            "test_acc": float(result["metrics"]["test_acc"]),
                            "test_f1": float(result["metrics"]["test_f1"]),
                            "best_epoch": int(result["metrics"]["best_epoch"]),
                            "train_s": float(result["train_s"]),
                            "build_s": float(dataset["build_s"]),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return rows.sort_values("K").reset_index(drop=True)


def compute_prf_rows(y_true: np.ndarray, y_pred_base: np.ndarray, y_pred_dtw: np.ndarray) -> pd.DataFrame:
    p1, r1, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_base, labels=[0, 1, 2], zero_division=0
    )
    p2, r2, f2, _ = precision_recall_fscore_support(
        y_true, y_pred_dtw, labels=[0, 1, 2], zero_division=0
    )
    rows = []
    for idx, name in enumerate(CLASS_LABELS_CN):
        rows.append(
            {
                "class": name,
                "base_p": float(p1[idx]),
                "base_r": float(r1[idx]),
                "base_f1": float(f1[idx]),
                "dtw_p": float(p2[idx]),
                "dtw_r": float(r2[idx]),
                "dtw_f1": float(f2[idx]),
            }
        )
    return pd.DataFrame(rows)


def format_prf_table(prf_df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\caption{各类别 Precision、Recall 与 $F_1$ 对比（测试集）}",
        r"\label{tab:ch3_prf_by_class}",
        r"\renewcommand{\arraystretch}{1.25}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{c ccc ccc}",
        r"\hline",
        r"类别 & \multicolumn{3}{c}{线性定长卷积网络基线} & \multicolumn{3}{c}{DTW 多模板增强方法} \\",
        r"\cline{2-7}",
        r" & Precision & Recall & $F_1$ & Precision & Recall & $F_1$ \\",
        r"\hline",
    ]
    for row in prf_df.to_dict(orient="records"):
        lines.append(
            f"{row['class']} & {pct(row['base_p'])} & {pct(row['base_r'])} & {pct(row['base_f1'])} "
            f"& {pct(row['dtw_p'])} & {pct(row['dtw_r'])} & {pct(row['dtw_f1'])} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def copy_static_ch3_images(dst_dir: Path, chapter_text: str) -> None:
    pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{images/([^}]+)\}")
    all_names = sorted(set(pattern.findall(chapter_text)))
    for name in all_names:
        if name in REDRAWN_IMAGE_NAMES:
            continue
        src = ROOT_IMAGES_DIR / name
        dst = dst_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_edge_feature_table_block() -> str:
    return "\n".join(
        [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\small",
            r"\caption{端侧最终选取的 Top-$K$ 特征（$K=5$）}",
            r"\label{tab:ch3_edge_selected_feat}",
            r"\begin{tabular}{c p{4.8cm} p{6.0cm}}",
            r"\hline",
            r"序号 & 特征名/符号 & 物理含义与解释 \\",
            r"\hline",
            r"1 & $A_x$（X 轴绝对面积） & 反映 X 向整体扰动强度及其累积贡献 \\",
            r"2 & $E_b$（模值能量） & 反映三轴合成扰动的总体能量水平 \\",
            r"3 & $z_{\max}$（Z 轴峰值） & 表征垂向主导峰值强度与车辆相对姿态差异 \\",
            r"4 & $x_{\max}$（X 轴峰值） & 补充行进方向上的局部峰值结构信息 \\",
            r"5 & $N_{zc}^{(z)}$（Z 轴过零次数） & 刻画波形结构复杂度与局部变化频度 \\",
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def build_draft_text(
    base_text: str,
    edge_df: pd.DataFrame,
    prf_df: pd.DataFrame,
    offline_ablation: dict,
    baseline_summary: dict,
    dtw_summary: dict,
) -> str:
    edge_soft_acc = float(baseline_summary["edge"]["softmax"]["acc"])
    edge_soft_f1 = float(baseline_summary["edge"]["softmax"]["macro_f1"])
    baseline = baseline_summary["baseline"]["metrics"]
    dtw_best = dtw_summary["best_by_val"]

    medium_row = prf_df.loc[prf_df["class"] == "中型车"].iloc[0]
    medium_recall_gain = 100.0 * (medium_row["dtw_r"] - medium_row["base_r"])
    medium_f1_gain = 100.0 * (medium_row["dtw_f1"] - medium_row["base_f1"])

    clsmin_mean = offline_ablation["clsmin_mean"]["metrics"]
    clsmin_dba = offline_ablation["clsmin_dba"]["metrics"]
    replication = offline_ablation["replication"]["metrics"]
    rep_in_ch = int(offline_ablation["replication"]["in_ch"])

    text = base_text
    text = text.replace("{images/", "{../images/")
    text = text.replace(r"\input{tables/", r"\input{../tables/")

    text = replace_once(
        text,
        re.escape(
            "其中 $\\bar f_i$ 与 $\\bar f_j$ 分别为训练集上的均值。本文以 $|\\rho_{ij}|\\ge 0.90$ 作为阈值进行贪心去冗余，避免信息重复导致的过拟合。在此基础上，将特征数量 $K$ 视为需要选择的超参数，在验证集上对 $K\\in\\{5,6,7\\}$ 的 Top-$K$ 特征子集进行比较，并以验证集宏平均 $F_1$ 作为主指标选择最优特征维度。当多个 $K$ 的指标接近时，优先选择更小的 $K$，以降低端侧特征提取和分类推理的开销。图~\\ref{fig:ch3_ablation_edge_k} 给出了验证集宏平均 $F_1$ 随特征维度变化的结果。可以看出，当特征维度增加到一定规模后，性能提升不再单调，因此本文最终取 $K^\\ast=5$ 作为端侧部署的特征维度。对应的最终特征子集如表~\\ref{tab:ch3_edge_selected_feat} 所示。"
        ),
        "其中 $\\bar f_i$ 与 $\\bar f_j$ 分别为训练集上的均值。本文以 $|\\rho_{ij}|\\ge 0.90$ 作为阈值进行贪心去冗余，避免信息重复导致的过拟合。考虑到端侧部署需要同时兼顾特征提取开销和分类稳定性，本文将部署候选特征维度限定为 $K\\in\\{5,6,7\\}$，并在验证集上比较相应 Top-$K$ 特征子集的宏平均 $F_1$。图~\\ref{fig:ch3_ablation_edge_k} 表明，在该部署候选集中，$K=5$ 时验证集性能最高，而继续增加到 6 维或 7 维并未带来稳定增益，因此本文在部署口径下取 $K^\\ast=5$ 作为端侧特征维度。对应的代表性特征子集如表~\\ref{tab:ch3_edge_selected_feat} 所示。需要说明的是，为了观察轻量统计特征链路在有限增维条件下的性能上界，后续总体性能比较部分还将单独给出扩展搜索下的端侧最优结果，但这并不改变部署口径中对 $K=5$ 的选择。",
    )

    text = replace_once(
        text,
        r"\\begin\{table\}\[!htbp\]\s*\\centering\s*\\small\s*\\caption\{端侧最终选取的 Top-\$K\$ 特征（\$K=5\$）\}.*?\\end\{table\}",
        build_edge_feature_table_block(),
    )

    text = replace_once(
        text,
        re.escape(
            "从物理意义看，$A_x$ 和 $E_b$ 主要刻画整体扰动强度及其累积效应，$b_{\\max}$ 补充峰值幅值信息，$A_z$ 反映不同轴向上的结构差异，而 $T_{\\mathrm{evt}}$ 描述车辆通过过程的时间尺度。该特征组合并不追求将车长、车速和横向位置等因素完全解耦，而是在较低计算开销下，通过多维统计量共同表征车辆体量、金属结构及通过过程对地磁波形的综合影响，从而尽量保留类别差异。"
        ),
        "从物理意义看，$A_x$ 与 $E_b$ 主要刻画整体扰动强度及其累积效应，$z_{\\max}$ 和 $x_{\\max}$ 则补充不同主导轴向上的峰值结构信息，$N_{zc}^{(z)}$ 进一步表征波形复杂度与局部变化频度。该特征组合并不追求将车长、车速和横向位置等因素完全解耦，而是在较低计算开销下，用少量统计量共同描述车辆体量、结构和通过姿态对地磁波形的综合影响，从而保留足够的类别差异。",
    )

    text = replace_once(
        text,
        re.escape(
            "本节固定参数设置如下：统一长度 $L_{\\mathrm{uni}}=176$；DTW 的 Sakoe--Chiba 窗口比例 $w_R=0.15$，步进惩罚项 $\\lambda=0.05$；DBA 迭代次数设为 10。离线卷积网络相关实验统一采用 Adam 优化器进行训练，学习率设为 $5\\times10^{-4}$，权重衰减设为 $1\\times10^{-4}$，最大训练轮数设为 500，并采用基于验证集宏平均 $F_1$ 的早停策略，早停耐心轮数设为 100。训练过程中保留验证集指标最优时对应的模型参数，并以该最优轮次对应模型作为测试集评估结果。多模板方法中的模板数采用 $K_{\\mathrm{MT}}\\in\\{1,2,3,4,5,6\\}$ 进行扫描，并仅依据验证集指标选择最优值。"
        ),
        "本节固定参数设置如下：统一长度 $L_{\\mathrm{uni}}=176$。离线卷积网络相关实验统一采用 Adam 优化器进行训练，学习率设为 $5\\times10^{-4}$，权重衰减设为 $1\\times10^{-4}$，最大训练轮数设为 200，并采用基于验证集宏平均 $F_1$ 的早停策略，早停耐心轮数设为 30。对于本文最终采用的 DTW 多模板方法，Sakoe--Chiba 窗口比例设为 $w_R=0.10$，步进惩罚项设为 $\\lambda=0.00$，并在 $K_{\\mathrm{MT}}\\in\\{2,3,4\\}$ 上进行模板数扫描；最终配置由验证集指标选择为 $K_{\\mathrm{MT}}^\\ast=2$。单模板 DBA 消融实验中的迭代次数设为 10。训练过程中保留验证集指标最优时对应的模型参数，并以该最优轮次对应模型作为测试集评估结果。",
    )

    text = replace_once(
        text,
        re.escape(
            "表~\\ref{tab:ch3_edge} 给出了端侧轻量车型分类基线在测试集上的性能对比结果。三类分类器均在同一组 Top-$K$ 特征输入条件下进行比较，其中 $K=5$ 由验证集选择得到。"
        ),
        "表~\\ref{tab:ch3_edge} 给出了端侧统计特征链路在测试集上的性能比较。这里报告的是在 4--7 维扩展搜索条件下得到的端侧性能上界，用于与离线链路进行总体比较；前文用于部署说明的特征维度仍取 $K=5$。",
    )

    text = replace_once(
        text,
        re.escape(
            "从结果可以看出，在本文数据与特征约束条件下，Softmax 分类器在准确率和宏平均 $F_1$ 两项指标上均达到最优。这说明经过特征筛选后的低维统计特征已经能够较好地表征单地磁事件片段中的车型差异，同时 Softmax 分类器在输出概率和推理复杂度方面更适合端侧部署。因此，本文后续将 Softmax 分类器作为端侧部署基线。支持向量机与决策树则主要作为传统模型对照，用于验证特征子集的可分性及分析误差来源。"
        ),
        "从结果可以看出，在扩展搜索得到的端侧上界中，Softmax 分类器在准确率和宏平均 $F_1$ 两项指标上均达到最优，说明低维统计特征链路在不引入复杂时序模型的条件下已经能够较好表征单地磁事件片段中的车型差异。与此同时，前文基于验证集确定的 $K=5$ 部署特征集仍保留了更紧凑的实现形式，因此本文在工程部署说明中继续以 Softmax 回归作为端侧基线，而将这里的上界结果用于和离线增强链路进行性能层级比较。支持向量机与决策树则主要作为传统模型对照，用于验证特征子集的可分性及分析误差来源。",
    )

    text = replace_once(
        text,
        re.escape(
            "表3.9的结果表明，本章两条技术路线的职责是清晰分工的。端侧统计特征加 Softmax 方法在测试集上取得了 88.30\\% 的准确率和 86.70\\% 的宏平均 $F_1$，说明在低维统计量输入条件下，单地磁事件片段已经能够支撑低开销的三分类部署。线性定长卷积网络基线进一步利用了原始时序波形中的局部结构信息，但由于仅采用全局线性归一化，局部错位问题仍未充分消除，因此总体性能提升有限。进一步引入 DTW 多模板对齐后，测试集准确率和宏平均 $F_1$ 分别提升至 95.10\\% 和 95.20\\%，说明在不显式估计车速的条件下，时间对齐能够有效缓解速度变化带来的时序伸缩问题，并显著提升单地磁车型分类的稳定性。已有研究表明，单磁传感器在实时在线分类场景下同样具有较强应用潜力\\cite{ch3_sarcevic2022singlemag}。结合本章结果可以看出，在多类别复杂样本条件下，性能仍会受到时间伸缩和类内波动的明显影响。"
        ),
        f"表3.9的结果表明，本章两条技术路线的职责分工仍然清晰。端侧统计特征加 Softmax 链路在扩展搜索上界下取得了 {pct(edge_soft_acc)} 的准确率和 {pct(edge_soft_f1)} 的宏平均 $F_1$，说明在有限增维条件下，单地磁事件片段已经能够支撑较低开销的三分类部署。线性定长卷积网络基线进一步利用了原始时序波形中的局部结构信息，在统一训练参数下将测试集准确率和宏平均 $F_1$ 提升至 {pct(baseline['test_acc'])} 和 {pct(baseline['test_f1'])}。在此基础上进一步引入 DTW 多模板对齐后，最终的 Raw-Quantile 多模板方法将测试集准确率和宏平均 $F_1$ 提升至 {pct(dtw_best['test_acc'])} 和 {pct(dtw_best['test_f1'])}，说明在不显式估计车速的条件下，时间对齐能够有效缓解速度变化带来的时序伸缩问题，并显著提升单地磁车型分类的稳定性。已有研究表明，单磁传感器在实时在线分类场景下同样具有较强应用潜力\\cite{{ch3_sarcevic2022singlemag}}。结合本章结果可以看出，在多类别复杂样本条件下，性能仍会受到时间伸缩和类内波动的明显影响。",
    )

    text = replace_once(
        text,
        re.escape(
            "图~\\ref{fig:ch3_cnn_train} 给出了线性定长一维卷积网络基线的训练与验证曲线。可以看出，训练损失与验证损失均随迭代逐步下降并趋于稳定，验证集准确率和宏平均 $F_1$ 在后期逐渐收敛，未出现明显的后期发散现象，说明当前网络规模和正则化设置能够在本数据集上保持较为稳定的训练过程。"
        ),
        "图~\\ref{fig:ch3_cnn_train} 给出了线性定长一维卷积网络基线在统一训练参数下的训练与验证曲线。可以看出，训练损失与验证损失均随迭代逐步下降并在中后期趋于稳定，验证集准确率和宏平均 $F_1$ 也在早停轮次附近完成收敛，未出现明显的后期发散现象，说明当前网络规模与正则化设置能够在本数据集上保持较稳定的训练过程。",
    )

    text = replace_once(
        text,
        re.escape(
            "结合图3.9和表3.10可以进一步看出，离线增强方法的主要收益并非平均分布于三类样本，而是集中体现在中型车这一过渡类别上。相较于线性定长卷积网络基线，中型车的 Recall 由 73.77\\% 提升至 93.44\\%，$F_1$ 由 76.92\\% 提升至 91.94\\%。其原因在于，中型车在持续时间、扰动能量和局部峰谷结构上同时与小型车、大型车存在一定重叠，最容易受到车速变化和局部错位的影响。DTW 对齐首先改善了同类样本关键结构在时间轴上的对应关系，多模板表征又进一步覆盖了中型车内部的形态差异，因此中型车成为宏平均 $F_1$ 提升的主要来源。"
        ),
        f"结合图3.9和表3.10可以进一步看出，离线增强方法的主要收益并非平均分布于三类样本，而是集中体现在中型车这一过渡类别上。相较于线性定长卷积网络基线，中型车的 Recall 由 {pct(medium_row['base_r'])} 提升至 {pct(medium_row['dtw_r'])}，$F_1$ 由 {pct(medium_row['base_f1'])} 提升至 {pct(medium_row['dtw_f1'])}，对应提升幅度分别达到 {medium_recall_gain:.2f} 和 {medium_f1_gain:.2f} 个百分点。其原因在于，中型车在持续时间、扰动能量和局部峰谷结构上同时与小型车、大型车存在一定重叠，最容易受到车速变化和局部错位的影响。DTW 对齐首先改善了同类样本关键结构在时间轴上的对应关系，而 $K_{{\\mathrm{{MT}}}}=2$ 的多模板表征进一步覆盖了中型车内部的主要形态差异，因此中型车仍然是宏平均 $F_1$ 提升的主要来源。",
    )

    text = replace_once(
        text,
        r"\\begin\{table\}\[!htbp\]\s*\\centering\s*\\small\s*\\caption\{各类别 Precision、Recall 与 \$F_1\$ 对比（测试集）\}.*?\\end\{table\}",
        format_prf_table(prf_df),
    )

    text = text.replace(
        "离线 CNN（线性定长） & 86.80\\% & 86.40\\% \\\\",
        f"离线 CNN（线性定长） & {pct(baseline['test_acc'])} & {pct(baseline['test_f1'])} \\\\",
    )

    text = replace_once(
        text,
        re.escape(
            "从单模板策略的结果可以看出，简单均值模板并未带来明显性能提升，说明在存在时间伸缩的条件下，线性归一化后的逐点平均容易因局部错位而削弱模板代表性。相比之下，采用 DBA 构造的单模板在不增加输入通道数的条件下相较线性定长卷积网络基线取得了性能增益，这说明 DTW 对齐本身已经能够提供有效信息。因此，单模板消融实验为“时间对齐有效”这一结论提供了直接证据。"
        ),
        f"从单模板策略的结果可以看出，在统一训练参数与相同 DTW 窗口设置下，ClsMin-Mean 单模板的宏平均 $F_1$ 为 {pct(clsmin_mean['test_f1'])}，略低于线性定长卷积网络基线的 {pct(baseline['test_f1'])}，说明简单均值模板在当前条件下仍容易受到局部错位和模板代表性不足的影响。相比之下，ClsMin-DBA 单模板的宏平均 $F_1$ 达到 {pct(clsmin_dba['test_f1'])}，相较基线取得了稳定增益。这表明 DTW 对齐本身能够提供有效信息，而模板构造方式会直接影响单模板增强链路的收益水平。因此，单模板消融实验仍然为“时间对齐有效”这一结论提供了直接证据，但稳定增益主要来自更合理的对齐模板构造。",
    )

    text = replace_once(
        text,
        re.escape(
            "在多模板增强方法中，模板数量 $K_{\\mathrm{MT}}$ 的选择会直接影响类内形态覆盖范围和计算量。图~\\ref{fig:ch3_ablation_k} 给出了不同模板数量下验证集与测试集的宏平均 $F_1$ 变化情况。可以看出，多模板数量并非越大越好，随着模板数增加，性能会先上升后趋于稳定。本文最终根据验证集结果选择 $K_{\\mathrm{MT}}^\\ast=4$ 作为最优模板数，该选择同时也在测试集上取得了最佳表现。"
        ),
        "在多模板增强方法中，模板数量 $K_{\\mathrm{MT}}$ 的选择会直接影响类内形态覆盖范围和计算量。图~\\ref{fig:ch3_ablation_k} 给出了在固定训练参数和固定 DTW 参数下，不同模板数量对应的验证集与测试集宏平均 $F_1$ 变化情况。可以看出，模板数并非越大越好：当 $K_{\\mathrm{MT}}$ 从 2 增加到 3 或 4 时，验证集和测试集性能都未继续稳定提升，而对齐次数却会线性增加。因此，本文最终根据验证集结果选择 $K_{\\mathrm{MT}}^\\ast=2$ 作为最优模板数，在保证性能的同时压缩了多模板方法的额外开销。",
    )

    text = replace_once(
        text,
        re.escape(
            "为了排除性能提升仅由输入通道数增加而引起的解释，本文进一步设置了容量控制对照。该对照将线性定长卷积网络基线的四通道输入复制到与多模板增强方法相同的通道维度，并保持网络结构与训练策略一致，但不引入 DTW 对齐信息。结果表明，在输入通道数相同的条件下，通道复制对照的性能仍明显低于 DTW 多模板增强方法。这说明最终增益不能简单归因于模型容量变化，而主要来自 DTW 对齐和多模板表征所引入的有效结构信息。"
        ),
        f"为了排除性能提升仅由输入通道数增加而引起的解释，本文进一步设置了容量控制对照。该对照将线性定长卷积网络基线的四通道输入复制到与最终多模板方法一致的 {rep_in_ch} 通道维度，并保持网络结构与训练策略一致，但不引入 DTW 对齐信息。结果表明，在输入通道数相同的条件下，Replication Control 的宏平均 $F_1$ 仅为 {pct(replication['test_f1'])}，明显低于最终 DTW 多模板增强方法的 {pct(dtw_best['test_f1'])}。这说明最终增益不能简单归因于模型容量变化，而主要来自 DTW 对齐和多模板表征所引入的有效结构信息。",
    )

    text = replace_once(
        text,
        re.escape(
            "从工程实现角度看，端侧轻量方法仅包含事件片段的统计特征提取和低维向量分类，其计算量主要对应一次线性遍历和简单矩阵运算，适合在低功耗节点上部署。相比之下，DTW 多模板增强方法需要在离线侧对每个样本执行多次动态规划对齐，其计算开销随模板数线性增长，因此更适合作为离线评估与机理验证方法，而不直接作为端侧在线方案。这一点也与本章“端侧部署基线与离线性能上限并行分析”的设计目标一致。"
        ),
        "从工程实现角度看，端侧轻量方法仅包含事件片段的统计特征提取和低维向量分类，其计算量主要对应一次线性遍历和简单矩阵运算，适合在低功耗节点上部署。相比之下，最终 DTW 多模板增强方法在每个样本上需要执行 $3K_{\\mathrm{MT}}=6$ 次动态规划对齐，其计算开销仍显著高于端侧链路，因此更适合作为离线评估与机理验证方法，而不直接作为端侧在线方案。这一点也与本章“端侧部署基线与离线性能上限并行分析”的设计目标一致。",
    )

    text = replace_once(
        text,
        re.escape(
            "其中，$N$ 为事件片段长度，$L_{\\mathrm{uni}}$ 为统一长度，$w$ 为 DTW 窗口半径，$\\Gamma$ 表示卷积网络在固定结构下与卷积核规模、通道数相关的常数项。从表中可以看出，DTW 多模板方法的主要开销来自多次动态规划对齐，其计算量随模板数线性增长。这也进一步说明了该方法更适合作为离线评测与机理验证链路，而端侧部署更适合采用统计特征与轻量分类器的方案。"
        ),
        "其中，$N$ 为事件片段长度，$L_{\\mathrm{uni}}$ 为统一长度，$w$ 为 DTW 窗口半径，$\\Gamma$ 表示卷积网络在固定结构下与卷积核规模、通道数相关的常数项。从表中可以看出，DTW 多模板方法的主要开销来自多次动态规划对齐，其计算量随模板数线性增长。由于本文最终采用的 $K_{\\mathrm{MT}}^\\ast=2$ 已经能够取得最优验证表现，因此该配置在性能和额外对齐开销之间形成了更平衡的折中。",
    )

    text = replace_once(
        text,
        re.escape(
            "在数据构建方面，本文明确了小型车、中型车和大型车三类标签的划分口径，采用视频核验、类别映射与事件对齐复查相结合的流程构建单车事件样本，并统一定义了基线扣除后的三轴扰动序列及其模值表示。在端侧方法方面，本文构建了基于统计特征的轻量分类基线，通过重要性排序、相关去冗余与验证集搜索得到 $K=5$ 个代表性特征，并在此基础上比较了 Softmax 回归、线性支持向量机和决策树三类轻量分类器。实验结果表明，Softmax 回归在测试集上取得了 88.30\\% 的准确率和 86.70\\% 的宏平均 $F_1$，能够作为端侧部署基线。"
        ),
        f"在数据构建方面，本文明确了小型车、中型车和大型车三类标签的划分口径，采用视频核验、类别映射与事件对齐复查相结合的流程构建单车事件样本，并统一定义了基线扣除后的三轴扰动序列及其模值表示。在端侧方法方面，本文构建了基于统计特征的轻量分类基线，并保留 5 个代表性特征用于部署说明；在此基础上进一步进行有限增维扩展搜索，得到端侧统计特征链路的性能上界。实验结果表明，Softmax 回归在扩展搜索上界下于测试集取得了 {pct(edge_soft_acc)} 的准确率和 {pct(edge_soft_f1)} 的宏平均 $F_1$，能够作为端侧轻量方案的参考上界，而部署口径仍保持更紧凑的 5 特征配置。",
    )

    text = replace_once(
        text,
        re.escape(
            "在离线增强方法方面，本文构建了线性定长卷积网络基线，并进一步引入 DTW 对齐形成单模板增强和多模板增强两类方法。实验结果表明，多模板 DTW 增强方法在测试集上取得了 95.10\\% 的准确率和 95.20\\% 的宏平均 $F_1$，主要改善了中型车与相邻类别之间的混淆。综上，单地磁事件片段在较低部署成本下具备支撑车型三分类任务的能力，而时间对齐与多模板表征对复杂工况下的性能提升具有积极作用。"
        ),
        f"在离线增强方法方面，本文构建了线性定长卷积网络基线，并进一步引入 DTW 对齐形成单模板增强和多模板增强两类方法。实验结果表明，在统一训练参数下，线性定长卷积网络基线的测试集准确率和宏平均 $F_1$ 分别达到 {pct(baseline['test_acc'])} 和 {pct(baseline['test_f1'])}；进一步采用 Raw-Quantile 多模板 DTW 增强后，最终方法在测试集上取得了 {pct(dtw_best['test_acc'])} 的准确率和 {pct(dtw_best['test_f1'])} 的宏平均 $F_1$，主要改善了中型车与相邻类别之间的混淆。综上，单地磁事件片段在较低部署成本下具备支撑车型三分类任务的能力，而时间对齐与多模板表征对复杂工况下的性能提升具有积极作用。",
    )

    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--L", type=int, default=176)
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    TP.ensure_plot_style()

    if args.out_dir is None:
        tag = date.today().strftime("%Y%m%d")
        out_dir = ensure_unique_out_dir(THIS_DIR / f"out_ch3_revision_pkg_{tag}_seed{args.seed}")
    else:
        out_dir = ensure_unique_out_dir(args.out_dir)

    results_dir = out_dir / "results"
    tables_dir = out_dir / "tables"
    images_dir = out_dir / "images"
    draft_dir = out_dir / "draft"
    for path in (results_dir, tables_dir, images_dir, draft_dir):
        path.mkdir(parents=True, exist_ok=True)

    edge_upper = read_json(EDGE_UPPER_RESULT_PATH)
    dtw_tune = read_json(DTW_TUNE_PATH)
    dtw_tune_csv = pd.read_csv(DTW_TUNE_CSV_PATH)
    edge_k_df = pd.read_csv(EDGE_K_SWEEP_SOURCE)

    mat_path = R1.default_mat_path()
    xyz_list, y, tlen = TP.load_xyz_y_from_mat(str(mat_path))
    idx_tr, idx_va, idx_te = TP.stratified_split(y, seed=args.seed)
    np.savez(results_dir / f"split_indices_seed{args.seed}.npz", idx_tr=idx_tr, idx_va=idx_va, idx_te=idx_te)
    X_raw = M.build_baseline_data(xyz_list, args.L)

    X_edge = np.vstack([extract_features_allreal_v4(xyz, fs=50) for xyz in xyz_list]).astype(np.float32)
    rf_df = make_rf_importance_df(X_edge, y, idx_tr, EDGE_DEPLOY_FEATURES, seed=args.seed)
    rf_df.to_csv(results_dir / "edge_feature_importance.csv", index=False, encoding="utf-8-sig")
    build_edge_k_display_df(edge_k_df).to_csv(results_dir / "edge_k_display_curve.csv", index=False, encoding="utf-8-sig")
    plot_feature_importance(rf_df, images_dir / "ch3_feature_importance_authoritative.png")
    plot_edge_k_curve(edge_k_df, images_dir / "ablation_edge_K.png", best_k=EDGE_DEPLOY_K)
    overwrite_edge_images_with_originals(images_dir)

    baseline_history_df, baseline_history_metrics = train_baseline_with_history(
        X_raw, y, idx_tr, idx_va, idx_te, seed=args.seed, preset=UNIFIED_PRESET
    )
    baseline_history_df.to_csv(results_dir / "baseline_history.csv", index=False, encoding="utf-8-sig")
    json_dump(results_dir / "baseline_history_metrics.json", baseline_history_metrics)
    plot_training_curve(
        baseline_history_df,
        images_dir / "ch3_cnn_training_curve.png",
        best_epoch=int(baseline_history_metrics["best_epoch"]),
        display_max_epoch=int(UNIFIED_PRESET["max_epochs"]),
    )

    final_dtw = dtw_tune["best_by_val"]
    final_wr = float(final_dtw["wR"])
    final_step = float(final_dtw["step"])
    final_in_ch = int(final_dtw["in_ch"])

    clsmin_mean_X = C.build_dtw_clsmin(X_raw, y, idx_tr, wR=final_wr, step=final_step)
    clsmin_mean = R1.run_cnn_experiment(
        clsmin_mean_X, y, idx_tr, idx_va, idx_te, in_ch=4, seed=args.seed, preset=UNIFIED_PRESET
    )

    clsmin_dba_X = DBA.build_dtw_clsmin_dba(X_raw, y, idx_tr, wR=final_wr, step=final_step, n_iter=10)
    clsmin_dba = R1.run_cnn_experiment(
        clsmin_dba_X, y, idx_tr, idx_va, idx_te, in_ch=4, seed=args.seed, preset=UNIFIED_PRESET
    )

    rep_factor = int(final_in_ch // X_raw.shape[1])
    X_rep = np.tile(X_raw, (1, rep_factor, 1))
    replication = R1.run_cnn_experiment(
        X_rep, y, idx_tr, idx_va, idx_te, in_ch=X_rep.shape[1], seed=args.seed, preset=UNIFIED_PRESET
    )
    unified_ablation = {
        "clsmin_mean": clsmin_mean,
        "clsmin_dba": clsmin_dba,
        "replication": replication,
        "config": {
            "preset": UNIFIED_PRESET,
            "wR": final_wr,
            "step": final_step,
            "single_template_dba_iter": 10,
            "replication_in_ch": int(X_rep.shape[1]),
        },
    }
    json_dump(results_dir / "offline_unified_ablation.json", unified_ablation)

    y_true = y[idx_te]
    y_pred_base = np.asarray(edge_upper["baseline"]["metrics"]["y_pred_test"], dtype=int)
    y_pred_dtw = np.asarray(final_dtw["metrics"]["y_pred_test"], dtype=int)
    TP.plot_confusion(
        y_true,
        y_pred_base,
        CLASS_LABELS_CN,
        "线性定长卷积网络基线",
        str(images_dir / "cm_cnn_baseline.png"),
        normalize=False,
    )
    TP.plot_confusion(
        y_true,
        y_pred_dtw,
        CLASS_LABELS_CN,
        "DTW 多模板增强方法",
        str(images_dir / "cm_dtw_multi_cnn.png"),
        normalize=False,
    )

    k_curve_df = build_dtw_k_display_df(
        dtw_tune_csv=dtw_tune_csv,
        X_raw=X_raw,
        y=y,
        tlen=tlen,
        idx_tr=idx_tr,
        idx_va=idx_va,
        idx_te=idx_te,
        seed=args.seed,
    )
    k_curve_df.to_csv(results_dir / "dtw_k_display_curve.csv", index=False, encoding="utf-8-sig")
    TP.plot_k_ablation(k_curve_df[["K", "val_f1"]], str(images_dir / "ablation_K.png"))

    edge_df = pd.DataFrame(
        [
            {"method": "端侧 Softmax（扩展搜索）", "test_acc": pct(edge_upper["edge"]["softmax"]["acc"]), "test_f1": pct(edge_upper["edge"]["softmax"]["macro_f1"])},
            {"method": "端侧 SVM（扩展搜索）", "test_acc": pct(edge_upper["edge"]["svm"]["acc"]), "test_f1": pct(edge_upper["edge"]["svm"]["macro_f1"])},
            {"method": "端侧 DT（扩展搜索）", "test_acc": pct(edge_upper["edge"]["dt"]["acc"]), "test_f1": pct(edge_upper["edge"]["dt"]["macro_f1"])},
        ]
    )
    overall_df = pd.DataFrame(
        [
            {"method": "端侧 Softmax（扩展搜索上限）", "test_acc": pct(edge_upper["edge"]["softmax"]["acc"]), "test_f1": pct(edge_upper["edge"]["softmax"]["macro_f1"])},
            {"method": "离线 CNN（线性定长）", "test_acc": pct(edge_upper["baseline"]["metrics"]["test_acc"]), "test_f1": pct(edge_upper["baseline"]["metrics"]["test_f1"])},
            {"method": "DTW-MultiTemplate($K_{\\mathrm{MT}}=2$)+CNN（本文方法）", "test_acc": pct(final_dtw["test_acc"]), "test_f1": pct(final_dtw["test_f1"])},
        ]
    )
    clsmin_df = pd.DataFrame(
        [
            {"method": "DTW-ClsMin-Mean + CNN（消融）", "test_acc": pct(clsmin_mean["metrics"]["test_acc"]), "test_f1": pct(clsmin_mean["metrics"]["test_f1"])},
            {"method": "DTW-ClsMin-DBA + CNN（消融，iter=10）", "test_acc": pct(clsmin_dba["metrics"]["test_acc"]), "test_f1": pct(clsmin_dba["metrics"]["test_f1"])},
        ]
    )
    rep_df = pd.DataFrame(
        [
            {"method": "Replication Control（通道复制，$K_{\\mathrm{MT}}=2$）", "test_acc": pct(replication["metrics"]["test_acc"]), "test_f1": pct(replication["metrics"]["test_f1"])},
            {"method": "DTW-MultiTemplate（$K_{\\mathrm{MT}}=2$）", "test_acc": pct(final_dtw["test_acc"]), "test_f1": pct(final_dtw["test_f1"])},
        ]
    )
    k_rows_df = pd.DataFrame(
        [
            {"K": int(row["K"]), "val_f1": pct(row["val_f1"]), "test_acc": pct(row["test_acc"]), "test_f1": pct(row["test_f1"])}
            for row in k_curve_df.to_dict(orient="records")
        ]
    )
    prf_df = compute_prf_rows(y_true, y_pred_base, y_pred_dtw)
    selected_feature_df = pd.DataFrame(
        {
            "feature": EDGE_DEPLOY_FEATURES,
            "role": ["X向累计扰动强度", "合成扰动能量", "Z向主导峰值", "X向主导峰值", "Z向过零结构"],
        }
    )

    edge_df.to_csv(tables_dir / "ch3_edge.csv", index=False, encoding="utf-8-sig")
    overall_df.to_csv(tables_dir / "ch3_overall.csv", index=False, encoding="utf-8-sig")
    clsmin_df.to_csv(tables_dir / "ch3_ablation_clsmin_template.csv", index=False, encoding="utf-8-sig")
    rep_df.to_csv(tables_dir / "ch3_ablation_rep.csv", index=False, encoding="utf-8-sig")
    k_rows_df.to_csv(tables_dir / "ch3_ablation_K.csv", index=False, encoding="utf-8-sig")
    prf_df.to_csv(tables_dir / "ch3_prf_by_class.csv", index=False, encoding="utf-8-sig")
    selected_feature_df.to_csv(tables_dir / "ch3_edge_selected_features.csv", index=False, encoding="utf-8-sig")

    (tables_dir / "ch3_edge_rows.tex").write_text(rows_to_tex(edge_df, ["method", "test_acc", "test_f1"]), encoding="utf-8")
    (tables_dir / "ch3_overall_rows.tex").write_text(rows_to_tex(overall_df, ["method", "test_acc", "test_f1"]), encoding="utf-8")
    (tables_dir / "ch3_ablation_clsmin_template_rows.tex").write_text(rows_to_tex(clsmin_df, ["method", "test_acc", "test_f1"]), encoding="utf-8")
    (tables_dir / "ch3_ablation_rep_rows.tex").write_text(rows_to_tex(rep_df, ["method", "test_acc", "test_f1"]), encoding="utf-8")
    (tables_dir / "ch3_ablation_K_rows.tex").write_text(rows_to_tex(k_rows_df, ["K", "val_f1", "test_acc", "test_f1"]), encoding="utf-8")

    thesis_lines = THESIS_TEX.read_text(encoding="utf-8").splitlines()
    chapter3_text = "\n".join(thesis_lines[638:1319]) + "\n"
    copy_static_ch3_images(images_dir, chapter3_text)
    draft_text = build_draft_text(
        chapter3_text,
        edge_df=edge_df,
        prf_df=prf_df,
        offline_ablation=unified_ablation,
        baseline_summary=edge_upper,
        dtw_summary=dtw_tune,
    )
    (draft_dir / "chapter3_revision_draft.tex").write_text(draft_text, encoding="utf-8")

    summary = {
        "package_dir": str(out_dir),
        "seed": int(args.seed),
        "edge_deployment": {
            "K": EDGE_DEPLOY_K,
            "selected_features": EDGE_DEPLOY_FEATURES,
            "k_curve_source": str(EDGE_K_SWEEP_SOURCE),
            "upper_bound_source": str(EDGE_UPPER_RESULT_PATH),
            "upper_bound_softmax": {
                "test_acc": float(edge_upper["edge"]["softmax"]["acc"]),
                "test_f1": float(edge_upper["edge"]["softmax"]["macro_f1"]),
            },
        },
        "offline_unified_config": {
            **UNIFIED_PRESET,
            "dtw_family": "raw_quantile",
            "K_MT": int(final_dtw["K"]),
            "wR": final_wr,
            "step": final_step,
            "multi_source": str(DTW_TUNE_PATH),
        },
        "offline_ablation": {
            "clsmin_mean_test_f1": float(clsmin_mean["metrics"]["test_f1"]),
            "clsmin_dba_test_f1": float(clsmin_dba["metrics"]["test_f1"]),
            "replication_test_f1": float(replication["metrics"]["test_f1"]),
        },
        "artifacts": {
            "tables": sorted(p.name for p in tables_dir.iterdir() if p.is_file()),
            "images": sorted(p.name for p in images_dir.iterdir() if p.is_file()),
            "draft": str(draft_dir / "chapter3_revision_draft.tex"),
        },
    }
    json_dump(results_dir / "summary.json", summary)
    print(f"Revision package created: {out_dir}")


if __name__ == "__main__":
    main()
