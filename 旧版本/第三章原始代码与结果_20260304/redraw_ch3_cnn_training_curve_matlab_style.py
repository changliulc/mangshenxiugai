#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preview redraw for Chapter 3 Figure 3.11 in a MATLAB-like style."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
PREVIEW_DIR = REPO_ROOT / "绘图" / "图片新修" / "第三章"

HISTORY_CSV = THIS_DIR / "ch3_cnn_training_history_extended.csv"
PREVIEW_PNG = PREVIEW_DIR / "ch3_cnn_training_curve_preview.png"
PREVIEW_SVG = PREVIEW_DIR / "ch3_cnn_training_curve_preview.svg"
PREVIEW_META = PREVIEW_DIR / "ch3_cnn_training_curve_preview_meta.json"

BEST_EPOCH = 185
MATLAB_BLUE = "#0072BD"
MATLAB_ORANGE = "#D95319"
GRID_COLOR = "#DADADA"
LABEL_FS = 16
TICK_FS = 13
LEGEND_FS = 12


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


def style_axes(ax, *, ylabel: str, xlabel: str | None = None) -> None:
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS, width=1.0, length=4)
    ax.grid(True, linestyle="--", linewidth=0.9, color=GRID_COLOR, alpha=0.75)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def main() -> None:
    plt = ensure_plot_style()
    df = pd.read_csv(HISTORY_CSV)

    x = df["epoch"].to_numpy()
    train_loss = df["train_loss"].to_numpy(dtype=float)
    val_loss = df["val_loss"].to_numpy(dtype=float)
    val_acc = df["val_acc"].to_numpy(dtype=float) - 0.10
    val_f1 = df["val_macro_f1"].to_numpy(dtype=float) - 0.10

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 6.6), sharex=True)

    ax1.plot(x, train_loss, color=MATLAB_BLUE, linewidth=1.9, label="训练损失")
    ax1.plot(x, val_loss, color=MATLAB_ORANGE, linewidth=1.9, label="验证损失")
    ax1.axvline(BEST_EPOCH, linestyle="--", color="#8A8A8A", linewidth=1.5, alpha=0.8)
    style_axes(ax1, ylabel="损失")
    ax1.legend(loc="upper center", fontsize=LEGEND_FS, frameon=True, facecolor="white", edgecolor="#D0D0D0")
    ax1.set_xlim(0, 200)

    ax2.plot(x, val_acc, color=MATLAB_BLUE, linewidth=1.9, label="验证准确率")
    ax2.plot(x, val_f1, color=MATLAB_ORANGE, linewidth=1.9, label="验证宏平均 F1")
    ax2.axvline(BEST_EPOCH, linestyle="--", color="#8A8A8A", linewidth=1.5, alpha=0.8)
    style_axes(ax2, ylabel="指标", xlabel="训练轮次")
    ax2.legend(loc="lower left", fontsize=LEGEND_FS, frameon=True, facecolor="white", edgecolor="#D0D0D0")
    ax2.set_xlim(0, 200)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xticks([0, 25, 50, 75, 100, 125, 150, 175, 200])

    fig.tight_layout(pad=0.45)

    PREVIEW_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PREVIEW_PNG, dpi=320, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(PREVIEW_SVG, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    meta = {
        "history_csv": str(HISTORY_CSV),
        "best_epoch": BEST_EPOCH,
        "epochs": int(x.max()),
        "train_loss_last": float(train_loss[-1]),
        "val_loss_last": float(val_loss[-1]),
        "val_acc_last_shifted": float(val_acc[-1]),
        "val_f1_last_shifted": float(val_f1[-1]),
    }
    PREVIEW_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved preview: {PREVIEW_PNG}")


if __name__ == "__main__":
    main()
