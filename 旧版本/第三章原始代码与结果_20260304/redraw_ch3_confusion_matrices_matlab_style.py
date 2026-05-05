#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preview redraws for Chapter 3 Figure 3.11 confusion matrices."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import scipy.io


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
PREVIEW_DIR = REPO_ROOT / "绘图" / "图片新修" / "第三章"
INPUT_TAG = "out_ch3_revision_pkg_20260314_seed42_v10"

PREVIEW_TARGETS = {
    "cm_cnn_baseline_data.mat": {
        "title": "线性定长卷积网络基线",
        "png": PREVIEW_DIR / "cm_cnn_baseline_preview.png",
        "svg": PREVIEW_DIR / "cm_cnn_baseline_preview.svg",
        "json": PREVIEW_DIR / "cm_cnn_baseline_preview_meta.json",
    },
    "cm_dtw_multi_cnn_data.mat": {
        "title": "DTW 多模板增强方法",
        "png": PREVIEW_DIR / "cm_dtw_multi_cnn_preview.png",
        "svg": PREVIEW_DIR / "cm_dtw_multi_cnn_preview.svg",
        "json": PREVIEW_DIR / "cm_dtw_multi_cnn_preview_meta.json",
    },
}

MATLAB_BLUE = "#0072BD"
GRID_EDGE = "#2F2F2F"


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
            "savefig.bbox": "tight",
        }
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    return plt


def find_input_mat(filename: str) -> Path:
    for path in THIS_DIR.rglob(filename):
        if INPUT_TAG in str(path):
            return path
    raise FileNotFoundError(f"Could not find {filename} under {INPUT_TAG}")


def load_payload(path: Path) -> dict:
    with open(path, "rb") as fh:
        raw = scipy.io.loadmat(fh)

    labels_raw = raw["labels"].reshape(-1)
    labels = []
    for item in labels_raw:
        if isinstance(item, np.ndarray):
            labels.append("".join(str(x) for x in item.reshape(-1)))
        else:
            labels.append(str(item))

    title_raw = raw.get("title")
    title = ""
    if title_raw is not None:
        title = str(np.asarray(title_raw).reshape(-1)[0])

    return {
        "cm": np.asarray(raw["cm"], dtype=int),
        "labels": labels,
        "title": title,
    }


def confusion_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "confusion_like_matlab",
        ["#F6DDD6", "#E9CCD2", "#90ADD1", MATLAB_BLUE],
        N=256,
    )


def draw_block(ax, values: np.ndarray, cmap, vmin: float, vmax: float, *,
               percent: bool = False, square: bool = True) -> None:
    arr = np.asarray(values, dtype=float)
    ax.imshow(
        arr,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal" if square else "auto",
        interpolation="nearest",
    )
    ax.set_xlim(-0.5, arr.shape[1] - 0.5)
    ax.set_ylim(arr.shape[0] - 0.5, -0.5)
    ax.tick_params(length=0)

    for x in np.arange(-0.5, arr.shape[1], 1.0):
        ax.plot([x, x], [-0.5, arr.shape[0] - 0.5], color=GRID_EDGE, linewidth=0.9)
    ax.plot(
        [arr.shape[1] - 0.5, arr.shape[1] - 0.5],
        [-0.5, arr.shape[0] - 0.5],
        color=GRID_EDGE,
        linewidth=0.9,
    )
    for y in np.arange(-0.5, arr.shape[0], 1.0):
        ax.plot([-0.5, arr.shape[1] - 0.5], [y, y], color=GRID_EDGE, linewidth=0.9)
    ax.plot(
        [-0.5, arr.shape[1] - 0.5],
        [arr.shape[0] - 0.5, arr.shape[0] - 0.5],
        color=GRID_EDGE,
        linewidth=0.9,
    )

    thresh = vmin + 0.60 * (vmax - vmin) if vmax > vmin else vmin
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            label = f"{int(round(val))}%" if percent else f"{int(round(val))}"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=21 if square else 18,
                color="white" if val >= thresh else "black",
            )

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color(GRID_EDGE)


def plot_confusion_preview(payload: dict, title: str, png_path: Path, svg_path: Path, meta_path: Path) -> None:
    plt = ensure_plot_style()

    cm = payload["cm"]
    labels = payload["labels"]

    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    tp = np.diag(cm)
    row_correct = np.where(row_sum > 0, np.round(100.0 * tp / row_sum), 0.0)
    col_correct = np.where(col_sum > 0, np.round(100.0 * tp / col_sum), 0.0)
    row_panel = np.column_stack([row_correct, 100.0 - row_correct])
    col_panel = np.vstack([col_correct, 100.0 - col_correct])

    fig = plt.figure(figsize=(8.2, 6.4))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[3.15, 1.2],
        height_ratios=[3.0, 1.05],
        left=0.08,
        right=0.965,
        bottom=0.10,
        top=0.90,
        wspace=0.10,
        hspace=0.10,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_row = fig.add_subplot(gs[0, 1])
    ax_col = fig.add_subplot(gs[1, 0])

    cmap = confusion_cmap()
    draw_block(ax_main, cm, cmap, 0.0, float(cm.max()) if cm.size else 1.0, percent=False, square=True)
    draw_block(ax_row, row_panel, cmap, 0.0, 100.0, percent=True, square=False)
    draw_block(ax_col, col_panel, cmap, 0.0, 100.0, percent=True, square=False)

    ax_main.set_xticks([])
    ax_main.set_yticks(np.arange(len(labels)))
    ax_main.set_yticklabels(labels, fontsize=18)
    ax_main.set_ylabel("真实类", fontsize=20)

    ax_row.set_xticks([])
    ax_row.set_yticks([])

    ax_col.set_xticks(np.arange(len(labels)))
    ax_col.set_xticklabels(labels, fontsize=18)
    ax_col.set_yticks([])
    ax_col.set_xlabel("预测类", fontsize=20)

    fig.suptitle(title, fontsize=20, y=0.965)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=320, pad_inches=0.04)
    fig.savefig(svg_path, pad_inches=0.04)
    plt.close(fig)

    meta = {
        "source_title": payload["title"],
        "preview_title": title,
        "labels": labels,
        "cm": cm.tolist(),
        "row_correct_percent": row_correct.astype(int).tolist(),
        "col_correct_percent": col_correct.astype(int).tolist(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    for filename, outputs in PREVIEW_TARGETS.items():
        mat_path = find_input_mat(filename)
        payload = load_payload(mat_path)
        plot_confusion_preview(
            payload=payload,
            title=outputs["title"],
            png_path=outputs["png"],
            svg_path=outputs["svg"],
            meta_path=outputs["json"],
        )
        print(f"Saved preview from {mat_path} -> {outputs['png']}")


if __name__ == "__main__":
    main()
