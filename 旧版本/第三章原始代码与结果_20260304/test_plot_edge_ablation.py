# -*- coding: utf-8 -*- 
""" 
plot_ablation_edge_k_with_vline.py 
功能： 
   1) 读取 ablation_edge_K_data.mat（字段：K, val_f1） 
   2) 绘制验证集 Macro-F1 曲线（y 轴 0~1） 
   3) 在 K=5 处画竖直虚线（可选标注 K*=5） 
""" 

import os 
import numpy as np 
import scipy.io as sio 
import matplotlib.pyplot as plt 


def pick_chinese_font(): 
    """尽量选择系统可用的中文字体（Windows 优先 YaHei，Linux 常见 Noto/SimHei）。""" 
    candidates = [ 
        "Microsoft YaHei", "Microsoft YaHei UI", "SimHei", 
        "Noto Sans CJK SC", "Noto Sans CJK", "WenQuanYi Micro Hei", 
        "PingFang SC", "Source Han Sans SC", "Arial Unicode MS" 
    ] 
    for f in candidates: 
        try: 
            plt.rcParams["font.sans-serif"] = [f] 
            plt.rcParams["axes.unicode_minus"] = False 
            return f 
        except Exception: 
            pass 
    plt.rcParams["axes.unicode_minus"] = False 
    return None 


def to_unit_interval(y): 
    """若 y 是百分制（>1.2），自动除以 100；否则保持 0~1。""" 
    y = np.asarray(y, dtype=float).reshape(-1) 
    if y.size == 0: 
        return y 
    if np.nanmax(y) > 1.2: 
        y = y / 100.0 
    return y 


def load_edge_k_mat(mat_path): 
    """从 .mat 文件读取 K 与 val_f1，做形状清洗。""" 
    D = sio.loadmat(mat_path) 

    if "K" in D: 
        K = D["K"].reshape(-1) 
    elif "K_list" in D: 
        K = D["K_list"].reshape(-1) 
    else: 
        raise KeyError("MAT 文件缺少字段 K（或 K_list）") 

    if "val_f1" in D: 
        val_f1 = D["val_f1"].reshape(-1) 
    elif "valF1" in D: 
        val_f1 = D["valF1"].reshape(-1) 
    else: 
        raise KeyError("MAT 文件缺少字段 val_f1（或 valF1）") 

    K = K.astype(int) 
    val_f1 = to_unit_interval(val_f1) 
    return K, val_f1 


def plot_edge_k(K, val_f1, highlight_k=5, out_png="ablation_edge_K.png", annotate=True): 
    pick_chinese_font() 

    fig, ax = plt.subplots(figsize=(10, 7), dpi=120) 

    # 曲线 
    ax.plot(K, val_f1, "-o", linewidth=2.2, markersize=9, label="Val Macro-F1") 

    # 竖直虚线（你要的：K=5） 
    ax.axvline(highlight_k, linestyle="--", linewidth=1.8, color="k") 

    # 可选标注：K*=5（放在 x 轴上方一点点，避免遮挡曲线） 
    if annotate: 
        ax.text( 
            highlight_k + 0.05, 0.03, r"$K^*=5$", 
            rotation=90, 
            transform=ax.get_xaxis_transform(),  # y 用轴坐标系 
            va="bottom", ha="left",
            fontsize=14  # 调整到合适大小
        ) 

    # 坐标轴与网格（匹配你给的风格） 
    ax.set_title("端侧特征维度选择（验证集）", fontsize=16) 
    ax.set_xlabel("特征数 K", fontsize=14) 
    ax.set_ylabel("验证集 Macro-F1", fontsize=14) 

    ax.set_xlim(min(K) - 0.2, max(K) + 0.2) 
    ax.set_xticks(K) 
    ax.tick_params(axis='x', labelsize=12)  # x轴刻度标签

    ax.set_ylim(0.0, 1.0) 
    ax.set_yticks(np.linspace(0, 1, 6)) 
    ax.tick_params(axis='y', labelsize=12)  # y轴刻度标签

    ax.grid(True, linestyle="--", alpha=0.4) 
    ax.spines["top"].set_linewidth(1.2) 
    ax.spines["right"].set_linewidth(1.2) 
    ax.spines["bottom"].set_linewidth(1.2) 
    ax.spines["left"].set_linewidth(1.2) 

    # 如果你不想要 legend，就注释掉下一行 
    # ax.legend(loc="best", fontsize=14) 

    fig.tight_layout() 
    fig.savefig(out_png, bbox_inches="tight") 
    plt.close(fig) 


if __name__ == "__main__": 
    # ====== 你只需要改这两个路径 ====== 
    mat_path = r"ablation_edge_K_data.mat"          # 例如：r"D:\...\ablation_edge_K_data.mat" 
    out_png  = r"ablation_edge_K.png"              # 例如：r"D:\...\ablation_edge_K.png" 

    # 你指定要在 K=5 处画虚线 
    highlight_k = 5 

    if not os.path.exists(mat_path): 
        raise FileNotFoundError(f"找不到 MAT 文件：{mat_path}") 

    K, val_f1 = load_edge_k_mat(mat_path) 
    plot_edge_k(K, val_f1, highlight_k=highlight_k, out_png=out_png, annotate=True) 

    print(f"[OK] Saved: {out_png}")