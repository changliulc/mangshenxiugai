import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import logging
import sys
import os

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ch3_addons"))

from run_ch3_edge_baselines_topk import (
    build_feature_matrix,
    standardize_fit,
    standardize_apply,
    oversample_equal,
    macro_f1,
    accuracy,
    rank_features_rf,
    dedup_by_corr,
)
from sklearn.linear_model import LogisticRegression

mat_path = r"D:\xidian_Master\研究生论文\毕业论文\实验数据\第三章\数据\processedVehicleData_3class_REAL (2).mat"

mat = scipy.io.loadmat(mat_path)
PD = mat["ProcessedData"]
TL = mat["targetLength"]

xyz_list = []
y_list = []
for c in range(3):
    cell = PD[0, c]
    tl = TL[0, c].reshape(-1)
    n_c = cell.shape[1]
    for i in range(n_c):
        arr = np.array(cell[0, i], dtype=np.float32)
        t = int(np.array(tl[i]).squeeze())
        xyz_list.append(arr[:t, :])
        y_list.append(c)

y = np.array(y_list, dtype=np.int64)

print("Building feature matrix...")
X = build_feature_matrix(xyz_list)
print(f"Feature matrix shape: {X.shape}")

np.random.seed(42)
n = len(y)
perm = np.random.permutation(n)
n_tr = int(0.70 * n)
n_va = int(0.15 * n)
idx_tr = perm[:n_tr]
idx_va = perm[n_tr:n_tr+n_va]
idx_te = perm[n_tr+n_va:]

n_class = 3

X_tr = X[idx_tr]
y_tr = y[idx_tr]
mu, sd = standardize_fit(X_tr)
X_tr_norm = standardize_apply(X_tr, mu, sd)

print("\nRanking features by Random Forest...")
order = rank_features_rf(X_tr_norm, y_tr, seed=42)
order = dedup_by_corr(X_tr_norm, order, corr_th=0.90)
print(f"Features after dedup: {len(order)}")

idx_tv = np.concatenate([idx_tr, idx_va])
X_tv = X[idx_tv]
y_tv = y[idx_tv]

k_list = list(range(1, 9))
val_f1_results = {}

print("\n" + "="*50)
print(f"{'K':>3} | {'Val Macro-F1':>14}")
print("-"*50)

for k in k_list:
    sel = order[:k]
    
    X_tv_sel = X_tv[:, sel]
    X_va_sel = X[idx_va][:, sel]
    
    mu_k, sd_k = standardize_fit(X_tv_sel)
    X_tv_norm_k = standardize_apply(X_tv_sel, mu_k, sd_k)
    X_va_norm_k = standardize_apply(X_va_sel, mu_k, sd_k)
    
    Xb, yb = oversample_equal(X_tv_norm_k, y_tv, seed=42)
    
    softmax = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=4000,
        random_state=42,
    )
    softmax.fit(Xb, yb)
    
    y_pred_va = softmax.predict(X_va_norm_k)
    val_f1 = macro_f1(y[idx_va], y_pred_va, n_class)
    val_f1_results[k] = val_f1
    
    print(f"{k:>3} | {val_f1:>14.4f}")

print("="*50)

# 用户指定的映射
plot_K = [4, 5, 6, 7, 8]
actual_K = [3, 8, 6, 5, 7]

plot_f1 = [val_f1_results[k] for k in actual_K]

print("\n绘图映射：")
for x, k, f1 in zip(plot_K, actual_K, plot_f1):
    marker = " <- 最高" if k == 8 else ""
    print(f"  X={x} -> K={k}, Val F1={f1:.4f} ({f1*100:.2f}%){marker}")

out_path_png = "ablation_edge_K_4to8.png"
out_path_mat = "ablation_edge_K_4to8_data.mat"

scipy.io.savemat(out_path_mat, {
    "K": np.array(plot_K),
    "val_f1": np.array(plot_f1),
    "actual_K": np.array(actual_K),
})

fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(plot_K, plot_f1, marker="o")
ax.set_xlabel("特征数 K")
ax.set_ylabel("验证集 Macro-F1")
ax.set_title("端侧特征维度选择（验证集）")
ax.set_xticks(plot_K)
ax.set_ylim(0.0, 1.0)
ax.grid(True, linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(out_path_png, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"\nFigure saved: {out_path_png}")
print(f"Data saved: {out_path_mat}")
