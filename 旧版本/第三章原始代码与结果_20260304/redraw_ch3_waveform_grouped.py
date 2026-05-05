#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新生成第三章图 3.6：
把原先“每类 1 个代表样本的 1x3 子图”改成“每类 3 个样本的分组对比图”。

设计目标：
1. 让小型车 / 中型车 / 大型车的差异更直观。
2. 仍然只基于现有 .mat 数据文件，不改论文中的图片路径。
3. 输出到：
   - figures/ch3_waveform_by_class.png
   - images/ch3_waveform_by_class.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
import scipy.io as sio


FS = 50
N0 = 10
L_REF = 176
CLASS_NAMES = ["小型车", "中型车", "大型车"]
AXIS_LABELS = ["X轴", "Y轴", "Z轴"]
AXIS_COLORS = ["#0072BD", "#D95319", "#EDB120"]


def pick_font() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def extract_event(bpad: np.ndarray, n: int, fs: int = FS, n0: int = N0):
    b = np.asarray(bpad[: int(n), :], dtype=float)
    base_len = min(int(n0), len(b))
    b0 = b[:base_len].mean(axis=0)
    db = b - b0
    mag = np.sqrt(np.sum(db**2, axis=1))
    t = np.arange(len(b), dtype=float) / fs
    return b, db, mag, t


def crop_by_mag(mag: np.ndarray, frac: float = 0.10, margin: int = 5) -> np.ndarray:
    mag = np.asarray(mag, dtype=float).reshape(-1)
    if mag.size == 0:
        return np.array([], dtype=int)
    thr = frac * np.max(mag)
    pos = np.flatnonzero(mag >= thr)
    if pos.size == 0:
        return np.arange(mag.size, dtype=int)
    start = max(0, int(pos[0]) - margin)
    end = min(mag.size, int(pos[-1]) + margin + 1)
    return np.arange(start, end, dtype=int)


def resample_linear(x: np.ndarray, out_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == out_len:
        return x.copy()
    xp = np.linspace(0.0, 1.0, x.size)
    xq = np.linspace(0.0, 1.0, out_len)
    return np.interp(xq, xp, x)


def pct(x: np.ndarray, p: float) -> float:
    return float(np.percentile(np.asarray(x, dtype=float), p))


def scale01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = np.min(x)
    hi = np.max(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.full_like(x, 0.5, dtype=float)
    return (x - lo) / (hi - lo)


def load_processed_data(mat_path: Path):
    mat = sio.loadmat(mat_path)
    return mat["ProcessedData"], mat["targetLength"]


def select_grouped_samples(processed_data, target_length, fs: int = FS, n0: int = N0, l_ref: int = L_REF):
    n_class = processed_data.shape[1]
    feat_cell: List[np.ndarray] = []
    energy_cell: List[np.ndarray] = []
    amp_cell: List[np.ndarray] = []
    len_cell: List[np.ndarray] = []
    tmpl = np.zeros((n_class, 3 * l_ref), dtype=float)

    for c in range(n_class):
        len_vec = np.asarray(target_length[0, c]).ravel().astype(float)
        class_cell = processed_data[0, c]
        n_samp = len_vec.size

        feat = np.zeros((n_samp, 3 * l_ref), dtype=float)
        energy = np.zeros(n_samp, dtype=float)
        amp = np.zeros(n_samp, dtype=float)

        for i in range(n_samp):
            _, db, mag, _ = extract_event(class_cell[0, i], len_vec[i], fs, n0)
            energy[i] = np.sum(mag**2)
            amp[i] = np.max(np.abs(db))

            x_ref = np.zeros((l_ref, 3), dtype=float)
            for k in range(3):
                xk = resample_linear(db[:, k], l_ref)
                xk = (xk - xk.mean()) / (xk.std() + np.finfo(float).eps)
                x_ref[:, k] = xk
            feat[i, :] = x_ref.reshape(-1)

        feat_cell.append(feat)
        energy_cell.append(energy)
        amp_cell.append(amp)
        len_cell.append(len_vec)

        tmp = feat.mean(axis=0)
        tmpl[c, :] = (tmp - tmp.mean()) / (tmp.std() + np.finfo(float).eps)

    q_list = [30, 50, 70]
    rep_idx = np.zeros((n_class, 3), dtype=int)
    rep_len = np.zeros((n_class, 3), dtype=int)
    rep_score = np.zeros((n_class, 3), dtype=float)

    for c in range(n_class):
        feat = feat_cell[c]
        energy = energy_cell[c]
        amp = amp_cell[c]
        len_vec = len_cell[c]

        own = feat @ tmpl[c, :].T / (3 * l_ref)
        other = np.full(feat.shape[0], -np.inf)
        for oc in range(n_class):
            if oc == c:
                continue
            oc_score = feat @ tmpl[oc, :].T / (3 * l_ref)
            other = np.maximum(other, oc_score)

        margin = own - other
        base = own + 0.80 * margin + 0.10 * scale01(np.log(energy + 1)) + 0.10 * scale01(amp)

        mask = (energy >= pct(energy, 40)) & (amp >= pct(amp, 40))
        if mask.sum() < 9:
            mask = energy >= pct(energy, 25)
        if mask.sum() < 6:
            mask = np.ones_like(mask, dtype=bool)

        used = np.zeros(feat.shape[0], dtype=bool)
        len_std = np.std(len_vec) + np.finfo(float).eps

        for j, q in enumerate(q_list):
            len_target = pct(len_vec, q)
            len_score = -np.abs(len_vec - len_target) / len_std
            total = base + 0.20 * len_score
            total[~mask] = -np.inf
            total[used] = -np.inf

            best_idx = int(np.argmax(total))
            best_score = total[best_idx]
            if not np.isfinite(best_score):
                total = base + 0.10 * len_score
                total[used] = -np.inf
                best_idx = int(np.argmax(total))
                best_score = total[best_idx]

            used[best_idx] = True
            rep_idx[c, j] = best_idx
            rep_len[c, j] = int(len_vec[best_idx])
            rep_score[c, j] = float(best_score)

    return rep_idx, rep_len, rep_score


def draw_grouped_waveform(processed_data, rep_idx, rep_len, out_png: Path, export_dir: Path) -> List[Dict[str, float]]:
    fig, ax = plt.subplots(figsize=(11.8, 4.6), dpi=300)
    ax.grid(True, alpha=0.35)
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("磁场扰动 / nT")

    waveform_examples: List[Dict[str, float]] = []
    t_cursor = 0.0
    gap_event = 0.18
    gap_group = 0.70
    evt_no = 0
    y_min_all, y_max_all = np.inf, -np.inf
    group_info = []

    for c in range(3):
        group_start = t_cursor
        group_y_min, group_y_max = np.inf, -np.inf
        class_cell = processed_data[0, c]

        for j in range(3):
            evt_no += 1
            idx = int(rep_idx[c, j])
            n = int(rep_len[c, j])

            _, db, mag, t = extract_event(class_cell[0, idx], n)
            use = crop_by_mag(mag, 0.10, 5)
            t_use = t[use] - t[use[0]]
            db_use = db[use, :]
            mag_use = mag[use]
            t_plot = t_cursor + t_use

            for k in range(3):
                ax.plot(
                    t_plot,
                    db_use[:, k],
                    color=AXIS_COLORS[k],
                    linewidth=1.4,
                    label=AXIS_LABELS[k] if evt_no == 1 else None,
                )

            export_data = np.column_stack([t_use, t_plot, db_use, mag_use])
            header = "t_local_s,t_plot_s,dBx_nT,dBy_nT,dBz_nT,b_nT"
            np.savetxt(
                export_dir / f"waveform_group_class{c+1}_evt{evt_no}_idx{idx+1}.csv",
                export_data,
                delimiter=",",
                header=header,
                comments="",
            )

            local_top = float(np.max(db_use))
            local_bot = float(np.min(db_use))
            local_x = float((t_plot[0] + t_plot[-1]) / 2)
            y_min_all = min(y_min_all, local_bot)
            y_max_all = max(y_max_all, local_top)
            group_y_min = min(group_y_min, local_bot)
            group_y_max = max(group_y_max, local_top)

            waveform_examples.append(
                {
                    "event_id": evt_no,
                    "class_id": c + 1,
                    "class_name": CLASS_NAMES[c],
                    "sample_index_1based": idx + 1,
                    "length": n,
                    "label_x": local_x,
                    "label_y": local_top,
                }
            )
            t_cursor = float(t_plot[-1]) + gap_event

        group_end = t_cursor - gap_event
        group_info.append((group_start, group_end, group_y_min, group_y_max, CLASS_NAMES[c]))
        t_cursor = group_end + gap_group

    y_range = y_max_all - y_min_all
    if y_range < 1e-9:
        y_range = max(1.0, abs(y_max_all))
    y_pad = 0.18 * y_range
    y_box_pad = 0.08 * y_range
    y_low = y_min_all - y_pad
    y_high = y_max_all + y_pad
    ax.set_xlim(0, t_cursor - gap_group + 0.25)
    ax.set_ylim(y_low, y_high)

    for x0, x1, y0, y1, name in group_info:
        rect = Rectangle(
            (x0 - 0.08, y0 - y_box_pad),
            (x1 - x0) + 0.16,
            (y1 - y0) + 2 * y_box_pad,
            fill=False,
            linestyle=":",
            linewidth=1.2,
            edgecolor=(0.35, 0.35, 0.35),
        )
        ax.add_patch(rect)
        ax.text((x0 + x1) / 2, y_low + 0.10 * y_range, name, ha="center", va="center", fontsize=12)

    for item in waveform_examples:
        label_y = min(item["label_y"] + 0.07 * y_range, y_high - 0.05 * y_range)
        ax.text(
            item["label_x"],
            label_y,
            str(item["event_id"]),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return waveform_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="重绘第三章三类车辆分组对比波形图")
    parser.add_argument("--mat", type=Path, default=None, help="processedVehicleData_3class_REAL (2).mat 路径")
    parser.add_argument("--out-dir", type=Path, default=None, help="输出目录，默认脚本同级 figures/")
    parser.add_argument("--copy-to-images", action="store_true", help="同时复制到论文 images/ 目录")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    mat_path = args.mat or (script_dir / "processedVehicleData_3class_REAL (2).mat")
    out_dir = args.out_dir or (script_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = repo_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if not mat_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {mat_path}")

    pick_font()
    processed_data, target_length = load_processed_data(mat_path)
    rep_idx, rep_len, rep_score = select_grouped_samples(processed_data, target_length)
    out_png = out_dir / "ch3_waveform_by_class.png"
    waveform_examples = draw_grouped_waveform(processed_data, rep_idx, rep_len, out_png, out_dir)

    np.savez(
        out_dir / "ch3_waveform_examples_grouped.npz",
        rep_idx=rep_idx,
        rep_len=rep_len,
        rep_score=rep_score,
    )

    if args.copy_to_images:
        (images_dir / "ch3_waveform_by_class.png").write_bytes(out_png.read_bytes())

    print("selected sample indices (1-based):")
    print(rep_idx + 1)
    print("selected lengths:")
    print(rep_len)
    print(f"saved figure: {out_png}")
    if args.copy_to_images:
        print(f"copied figure: {images_dir / 'ch3_waveform_by_class.png'}")
    print(f"saved grouped metadata: {out_dir / 'ch3_waveform_examples_grouped.npz'}")
    print(f"saved {len(waveform_examples)} grouped waveform csv files to: {out_dir}")


if __name__ == "__main__":
    main()
