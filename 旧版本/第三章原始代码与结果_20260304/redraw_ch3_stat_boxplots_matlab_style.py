#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Redraw Chapter 3 Figure 3.8 boxplots in a MATLAB-like thesis style."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.io


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
DATA_MAT = THIS_DIR / "processedVehicleData_3class_REAL (2).mat"
IMAGES_DIR = REPO_ROOT / "images"
PREVIEW_DIR = REPO_ROOT / "绘图" / "图片新修" / "第三章"

FORMAL_DURATION = IMAGES_DIR / "ch3_event_length_boxplot.png"
FORMAL_ENERGY = IMAGES_DIR / "ch3_mag_energy_boxplot.png"
PREVIEW_DURATION = PREVIEW_DIR / "ch3_event_length_boxplot_preview.png"
PREVIEW_ENERGY = PREVIEW_DIR / "ch3_mag_energy_boxplot_preview.png"
PREVIEW_DURATION_TOPTITLE = PREVIEW_DIR / "ch3_event_length_boxplot_toptitle_preview.png"
PREVIEW_ENERGY_TOPTITLE = PREVIEW_DIR / "ch3_mag_energy_boxplot_toptitle_preview.png"

CLASS_NAMES_ZH = ["小型车", "中型车", "大型车"]
MATLAB_COLORS = ["#0072BD", "#D95319", "#EDB120"]
LABEL_FS = 20
TICK_FS = 17
TITLE_FS = LABEL_FS


def ensure_plot_style():
    import logging
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


def load_xyz_y_from_mat(mat_path: Path):
    mat = scipy.io.loadmat(mat_path)
    pd_cell = mat["ProcessedData"]
    target_length = mat["targetLength"]

    xyz_list = []
    y = []
    for class_id in range(3):
        cell = pd_cell[0, class_id]
        lengths = target_length[0, class_id].reshape(-1)
        for idx in range(cell.shape[1]):
            arr = np.array(cell[0, idx], dtype=np.float32)
            length = int(np.array(lengths[idx]).squeeze())
            xyz_list.append(arr[:length, :])
            y.append(class_id)
    return xyz_list, np.asarray(y, dtype=int)


def compute_duration_and_energy(xyz_list: Sequence[np.ndarray], y: np.ndarray, fs: float = 50.0):
    durations = [[] for _ in range(3)]
    energies = [[] for _ in range(3)]
    for xyz, class_id in zip(xyz_list, y):
        xyz = np.asarray(xyz, dtype=np.float32)
        centered = xyz - np.median(xyz, axis=0, keepdims=True)
        mag = np.sqrt(np.sum(centered**2, axis=1))
        durations[class_id].append(float(xyz.shape[0] / fs))
        energies[class_id].append(float(np.sum(mag**2)))
    return [np.asarray(v, dtype=float) for v in durations], [np.asarray(v, dtype=float) for v in energies]


def take_evenly_spaced(values: np.ndarray, keep: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if keep <= 0 or values.size == 0:
        return np.asarray([], dtype=float)
    if values.size <= keep:
        return values
    idx = np.linspace(0, values.size - 1, num=keep)
    idx = np.unique(np.round(idx).astype(int))
    return values[idx]


def whisker_bounds(values: np.ndarray, whis: float) -> tuple[float, float]:
    arr = np.sort(np.asarray(values, dtype=float))
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (np.nan, np.nan)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    low_bound = q1 - whis * iqr
    high_bound = q3 + whis * iqr
    inside = arr[(arr >= low_bound) & (arr <= high_bound)]
    if inside.size == 0:
        return (np.nan, np.nan)
    return float(inside[0]), float(inside[-1])


def representative_outliers(
    values: np.ndarray,
    whis: float,
    max_points: int,
    high_cap: float | None = None,
) -> np.ndarray:
    arr = np.sort(np.asarray(values, dtype=float))
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or max_points <= 0:
        return np.asarray([], dtype=float)

    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    low_bound = q1 - whis * iqr
    high_bound = q3 + whis * iqr
    low_out = arr[arr < low_bound]
    high_out = arr[arr > high_bound]

    if high_cap is not None and np.isfinite(high_cap):
        high_out = high_out[high_out <= high_cap]

    total = low_out.size + high_out.size
    if total <= max_points:
        return np.concatenate([low_out, high_out])
    if low_out.size == 0:
        return take_evenly_spaced(high_out, max_points)
    if high_out.size == 0:
        return take_evenly_spaced(low_out, max_points)

    low_keep = max(1, int(round(max_points * low_out.size / total)))
    high_keep = max_points - low_keep
    if high_keep == 0:
        high_keep = 1
        low_keep = max_points - 1
    low_keep = min(low_keep, low_out.size)
    high_keep = min(high_keep, high_out.size)

    remain = max_points - low_keep - high_keep
    if remain > 0:
        low_room = low_out.size - low_keep
        high_room = high_out.size - high_keep
        if high_room >= low_room:
            extra = min(remain, high_room)
            high_keep += extra
            remain -= extra
        if remain > 0:
            low_keep += min(remain, low_room)

    return np.concatenate(
        [
            take_evenly_spaced(low_out, low_keep),
            take_evenly_spaced(high_out, high_keep),
        ]
    )


def overlay_representative_outliers(
    ax,
    groups: Sequence[np.ndarray],
    whis: float,
    max_points: int,
    size: float,
    alpha: float,
    high_caps: Sequence[float | None] | None = None,
) -> None:
    for pos, (values, color) in enumerate(zip(groups, MATLAB_COLORS), start=1):
        high_cap = None if high_caps is None else high_caps[pos - 1]
        picked = representative_outliers(
            np.asarray(values, dtype=float),
            whis=whis,
            max_points=max_points,
            high_cap=high_cap,
        )
        if picked.size == 0:
            continue
        xpos = np.full(picked.size, pos, dtype=float)
        ax.scatter(
            xpos,
            picked,
            s=size,
            c=color,
            alpha=alpha,
            linewidths=0,
            zorder=3,
            clip_on=True,
        )


def save_figure(fig, targets: Iterable[Path], dpi: int = 320, tight: bool = True):
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"dpi": dpi, "pad_inches": 0.03}
        if tight:
            save_kwargs["bbox_inches"] = "tight"
        fig.savefig(target, **save_kwargs)


def draw_boxplot(
    ax,
    groups: Sequence[np.ndarray],
    ylabel: str,
    title: str | None = None,
    yscale: str | None = None,
    whis: float = 1.5,
    high_caps: Sequence[float | None] | None = None,
):
    bp = ax.boxplot(
        groups,
        tick_labels=CLASS_NAMES_ZH,
        showfliers=False,
        whis=whis,
        widths=0.56,
        patch_artist=True,
        boxprops={"linewidth": 1.45, "color": "#4A4A4A"},
        whiskerprops={"linewidth": 1.20, "color": "#4A4A4A"},
        capprops={"linewidth": 1.20, "color": "#4A4A4A"},
        medianprops={"linewidth": 1.75, "color": "#303030"},
    )
    for patch, color in zip(bp["boxes"], MATLAB_COLORS):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.16)

    overlay_representative_outliers(
        ax,
        groups,
        whis=whis,
        max_points=6,
        size=17,
        alpha=0.30,
        high_caps=high_caps,
    )

    if yscale is not None:
        ax.set_yscale(yscale)

    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    if title:
        ax.text(
            0.5,
            1.025,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=TITLE_FS,
        )
    ax.grid(True, axis="y", linestyle="--", linewidth=0.82, color="#C6C6C6", alpha=0.42)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=TICK_FS, width=1.0, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def plot_duration_boxplot(
    groups: Sequence[np.ndarray],
    targets: Sequence[Path],
    title: str | None = None,
):
    plt = ensure_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    draw_boxplot(ax, groups, ylabel="事件持续时间 / s", whis=1.5, title=title)
    if title:
        fig.subplots_adjust(left=0.16, right=0.985, bottom=0.15, top=0.90)
        save_figure(fig, targets, tight=False)
    else:
        fig.tight_layout(pad=0.55)
        save_figure(fig, targets)
    plt.close(fig)


def plot_energy_boxplot(
    groups: Sequence[np.ndarray],
    targets: Sequence[Path],
    title: str | None = None,
):
    plt = ensure_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    whiskers = [whisker_bounds(group, whis=2.5) for group in groups]
    high_caps = [whiskers[1][1], whiskers[2][1], None]
    draw_boxplot(
        ax,
        groups,
        ylabel="模值能量（对数）",
        yscale="log",
        whis=2.5,
        high_caps=high_caps,
        title=title,
    )
    if title:
        fig.subplots_adjust(left=0.18, right=0.985, bottom=0.15, top=0.90)
        save_figure(fig, targets, tight=False)
    else:
        fig.tight_layout(pad=0.55)
        save_figure(fig, targets)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preview-top-title-only",
        action="store_true",
        help="Only export preview images with subplot names rendered inside the image top area.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    xyz_list, y = load_xyz_y_from_mat(DATA_MAT)
    duration_groups, energy_groups = compute_duration_and_energy(xyz_list, y)
    if args.preview_top_title_only:
        plot_duration_boxplot(
            duration_groups,
            [PREVIEW_DURATION_TOPTITLE],
            title="三类车辆事件持续时间分布箱线图",
        )
        plot_energy_boxplot(
            energy_groups,
            [PREVIEW_ENERGY_TOPTITLE],
            title="三类车辆模值能量分布箱线图",
        )
        print(f"[OK] Preview: {PREVIEW_DURATION_TOPTITLE}")
        print(f"[OK] Preview: {PREVIEW_ENERGY_TOPTITLE}")
        return

    plot_duration_boxplot(duration_groups, [FORMAL_DURATION, PREVIEW_DURATION])
    plot_energy_boxplot(energy_groups, [FORMAL_ENERGY, PREVIEW_ENERGY])
    print(f"[OK] Saved: {FORMAL_DURATION}")
    print(f"[OK] Saved: {FORMAL_ENERGY}")
    print(f"[OK] Preview: {PREVIEW_DURATION}")
    print(f"[OK] Preview: {PREVIEW_ENERGY}")


if __name__ == "__main__":
    main()
