#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preview redraws for Chapter 3 Figures 3.9 and 3.10 in a MATLAB-like style."""

from __future__ import annotations

from pathlib import Path

import logging
import numpy as np
import scipy.io


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
PREVIEW_DIR = REPO_ROOT / "绘图" / "图片新修" / "第三章"

EDGE_PREVIEW_PNG = PREVIEW_DIR / "ch3_ablation_edge_K_preview.png"
EDGE_PREVIEW_SVG = PREVIEW_DIR / "ch3_ablation_edge_K_preview.svg"
KMT_PREVIEW_PNG = PREVIEW_DIR / "ch3_ablation_K_preview.png"
KMT_PREVIEW_SVG = PREVIEW_DIR / "ch3_ablation_K_preview.svg"

# Match the data used by the accepted current figure.
EDGE_K = np.array([3, 4, 5, 6, 7, 8], dtype=int)
EDGE_VAL_F1 = np.array([0.8528, 0.8544, 0.8757, 0.8622, 0.8576, 0.8697], dtype=float)
EDGE_BEST_K = 5

# Read K_MT data from the current thesis figure payload to keep the same numbers.
KMT_DATA_MAT = REPO_ROOT / "images" / "ablation_K_data.mat"

MATLAB_BLUE = "#0072BD"
GRID_COLOR = "#D7D7D7"
LABEL_FS = 13
TICK_FS = 13
ANNO_FS = 13


def ensure_plot_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": [
                "Times New Roman",
                "SimSun",
                "NSimSun",
                "Songti SC",
                "STSong",
                "Noto Serif CJK SC",
                "DejaVu Serif",
            ],
            "axes.unicode_minus": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    return plt


def load_kmt_data():
    mat = scipy.io.loadmat(KMT_DATA_MAT)
    k = np.asarray(mat["K"]).reshape(-1).astype(int)
    val_f1 = np.asarray(mat["val_f1"]).reshape(-1).astype(float)
    return k, val_f1


def style_axes(ax, xlabel: str, ylabel: str):
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS, width=1.0, length=4)
    ax.grid(True, linestyle="--", linewidth=0.9, color=GRID_COLOR, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def save_fig(fig, *targets: Path):
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=320, bbox_inches="tight", pad_inches=0.03)


def plot_edge_preview():
    plt = ensure_plot_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(
        EDGE_K,
        EDGE_VAL_F1,
        color=MATLAB_BLUE,
        marker="o",
        linewidth=2.0,
        markersize=8.0,
        markerfacecolor=MATLAB_BLUE,
        markeredgecolor=MATLAB_BLUE,
    )
    ax.axvline(EDGE_BEST_K, color="black", linestyle="--", linewidth=1.6)
    ax.text(
        EDGE_BEST_K + 0.06,
        0.055,
        r"$K^*=5$",
        rotation=90,
        va="bottom",
        ha="left",
        fontsize=ANNO_FS,
    )
    ax.set_xticks(EDGE_K)
    ax.set_ylim(0.0, 1.0)
    style_axes(ax, "特征数 K", "验证集 Macro-F1")
    fig.tight_layout(pad=0.55)
    save_fig(fig, EDGE_PREVIEW_PNG, EDGE_PREVIEW_SVG)
    plt.close(fig)


def plot_kmt_preview():
    plt = ensure_plot_style()
    k_mt, val_f1 = load_kmt_data()
    best_idx = int(np.argmax(val_f1))
    best_k = int(k_mt[best_idx])

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(
        k_mt,
        val_f1,
        color=MATLAB_BLUE,
        marker="s",
        linewidth=2.0,
        markersize=7.8,
        markerfacecolor=MATLAB_BLUE,
        markeredgecolor=MATLAB_BLUE,
    )
    ax.axvline(best_k, color="black", linestyle="--", linewidth=1.6)
    ax.text(
        best_k + 0.08,
        0.055,
        r"$K_{\mathrm{MT}}^*=2$",
        rotation=90,
        va="bottom",
        ha="left",
        fontsize=ANNO_FS,
    )
    ax.set_xticks(k_mt)
    ax.set_ylim(0.0, 1.0)
    style_axes(ax, "模板数 K_MT（每类）", "验证集 Macro-F1")
    fig.tight_layout(pad=0.55)
    save_fig(fig, KMT_PREVIEW_PNG, KMT_PREVIEW_SVG)
    plt.close(fig)


def main():
    plot_edge_preview()
    plot_kmt_preview()
    print(f"[OK] Preview: {EDGE_PREVIEW_PNG}")
    print(f"[OK] Preview: {EDGE_PREVIEW_SVG}")
    print(f"[OK] Preview: {KMT_PREVIEW_PNG}")
    print(f"[OK] Preview: {KMT_PREVIEW_SVG}")


if __name__ == "__main__":
    main()
