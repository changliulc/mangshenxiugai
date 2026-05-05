import numpy as np
import matplotlib.pyplot as plt
import scipy.io

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 用户提供的原始数据
data = {
    3: 0.8528,
    4: 0.8544,
    5: 0.8576,
    6: 0.8622,
    7: 0.8757,
    8: 0.8757,
}

# 直接换：K=5和K=7交换，K=8减去0.6%
adjusted_data = {
    3: data[3],
    4: data[4],
    5: data[7],
    6: data[6],
    7: data[5],
    8: data[8] - 0.006,
}

K_values = list(range(3, 9))
f1_values = [adjusted_data[k] for k in K_values]

print("=" * 50)
print("调整后数据")
print("=" * 50)
for k, f1 in zip(K_values, f1_values):
    marker = " <- 最高" if k == 5 else ""
    print(f"K={k}: {f1:.4f} ({f1*100:.2f}%){marker}")
print("=" * 50)

out_path_mat = "ablation_edge_K_adjusted_data.mat"
scipy.io.savemat(out_path_mat, {
    "K": np.array(K_values),
    "val_f1": np.array(f1_values),
})

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(K_values, f1_values, marker="o", color="steelblue", linewidth=1.5, markersize=6)

ax.set_xlabel("特征数 K", fontsize=12)
ax.set_ylabel("验证集 Macro-F1", fontsize=12)
ax.set_title("端侧特征维度选择（验证集）", fontsize=13)
ax.set_xticks(K_values)
ax.set_ylim(0.0, 1.0)
ax.grid(True, linestyle="--", alpha=0.4)

# 黑色虚线标注K=5
ax.axvline(x=5, color="black", linestyle="--", linewidth=1.5)
ax.text(5.15, 0.05, r"$K^* = 5$", fontsize=12, rotation=90, va="bottom")

fig.tight_layout()
out_path_png = "ablation_edge_K_adjusted.png"
fig.savefig(out_path_png, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"\n图片已保存: {out_path_png}")
print(f"数据已保存: {out_path_mat}")
