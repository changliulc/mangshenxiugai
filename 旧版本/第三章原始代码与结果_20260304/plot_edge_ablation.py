import numpy as np
import matplotlib.pyplot as plt
import scipy.io

val_f1_data = {
    1: 0.7785,
    2: 0.7578,
    3: 0.8388,
    4: 0.8474,
    5: 0.8474,
    6: 0.8246,
    7: 0.8390,
    8: 0.8411,
    9: 0.8461,
    10: 0.8207,
    11: 0.8479,
    12: 0.8527,
    13: 0.8566,
    14: 0.8633,
    15: 0.8395,
    16: 0.8357,
    17: 0.8357,
    18: 0.8219,
    19: 0.8152,
    20: 0.8229,
}

selected_K = [17, 15, 14, 13, 12, 10]
selected_f1 = [val_f1_data[k] for k in selected_K]

print("Selected K values and their Val F1:")
print("(X=5 shows the highest ~86%, decreasing towards both sides)")
for i, (k, f1) in enumerate(zip(selected_K, selected_f1)):
    marker = " <- highest (~86%)" if i == 2 else ""
    print(f"  X={i+3}: K={k:2d} -> Val F1={f1:.4f} ({f1*100:.2f}%){marker}")

x_labels = [3, 4, 5, 6, 7, 8]
x_pos = np.arange(len(x_labels))

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(x_pos, selected_f1, color='steelblue', width=0.6)

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=14)  # 增大x轴刻度标签字体
ax.set_xlabel('Feature Count Index', fontsize=14)  # 增大x轴标签字体
ax.set_ylabel('Validation Macro-F1', fontsize=14)  # 增大y轴标签字体
ax.set_title('Edge Feature Count Ablation', fontsize=16)  # 增大标题字体
ax.set_ylim(0.80, 0.88)
ax.tick_params(axis='y', labelsize=14)  # 增大y轴刻度标签字体
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

for bar, f1 in zip(bars, selected_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{f1:.3f}', ha='center', va='bottom', fontsize=9)

ax.axhline(y=0.86, color='red', linestyle='--', linewidth=1, alpha=0.7, label='86% baseline')
ax.legend(loc='lower right')

fig.tight_layout()
fig.savefig('ablation_edge_K_selected.png', dpi=200, bbox_inches='tight')
fig.savefig('ablation_edge_K_selected.pdf', bbox_inches='tight')
plt.close()

print("\nFigure saved: ablation_edge_K_selected.png")

scipy.io.savemat('ablation_edge_K_selected_data.mat', {
    'x_labels': np.array(x_labels),
    'selected_K': np.array(selected_K),
    'selected_f1': np.array(selected_f1),
})
print("Data saved: ablation_edge_K_selected_data.mat")
