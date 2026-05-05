#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Draw the Chapter 3 edge-side lightweight classification baseline flowchart.

The figure is designed for thesis use:
1. No in-figure title, since LaTeX caption already provides it.
2. Monochrome styling for print-friendliness.
3. Clear separation between offline training and online inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def ensure_plot_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK JP",
        "PingFang SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    return plt


def add_round_box(ax, x, y, w, h, title, body_lines, *, fc="#ffffff", ec="#202020", lw=1.35):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h * 0.73,
        title,
        ha="center",
        va="center",
        fontsize=15.8,
        fontweight="semibold",
        color="#111111",
    )
    ax.text(
        x + w / 2,
        y + h * 0.44,
        "\n".join(body_lines),
        ha="center",
        va="center",
        fontsize=12.9,
        linespacing=1.35,
        color="#2a2a2a",
    )


def add_arrow(ax, xy1, xy2, *, lw=1.4, ls="-", color="#222222", ms=22):
    arrow = FancyArrowPatch(
        xy1,
        xy2,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        linestyle=ls,
        color=color,
        shrinkA=0,
        shrinkB=0,
        joinstyle="miter",
        capstyle="butt",
    )
    ax.add_patch(arrow)


def draw_flowchart(out_no_ext: Path) -> None:
    plt = ensure_plot_style()
    fig, ax = plt.subplots(figsize=(14.2, 5.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    lane_fc = "#f7f7f7"
    lane_ec = "#bdbdbd"
    text_ec = "#222222"

    # Two broad lanes
    upper_lane = FancyBboxPatch(
        (0.022, 0.56),
        0.95,
        0.28,
        boxstyle="round,pad=0.013,rounding_size=0.012",
        linewidth=1.1,
        edgecolor=lane_ec,
        facecolor=lane_fc,
    )
    lower_lane = FancyBboxPatch(
        (0.022, 0.13),
        0.95,
        0.28,
        boxstyle="round,pad=0.013,rounding_size=0.012",
        linewidth=1.1,
        edgecolor=lane_ec,
        facecolor=lane_fc,
    )
    ax.add_patch(upper_lane)
    ax.add_patch(lower_lane)

    ax.text(0.035, 0.81, "离线训练与模型固化", ha="left", va="center", fontsize=16.4, color=text_ec)
    ax.text(0.035, 0.33, "端侧在线推理", ha="left", va="center", fontsize=16.4, color=text_ec)

    # Shared geometry
    bw, bh = 0.24, 0.175
    top_y = 0.62
    bot_y = 0.18
    x1, x2, x3 = 0.10, 0.375, 0.65

    add_round_box(
        ax,
        x1,
        top_y,
        bw,
        bh,
        "训练/验证数据",
        ["事件片段与标签"],
    )
    add_round_box(
        ax,
        x2,
        top_y,
        bw,
        bh,
        "候选统计特征构建与筛选",
        ["重要性排序、相关去冗余", "验证集选 $K^{\\ast}=5$"],
    )
    add_round_box(
        ax,
        x3,
        top_y,
        bw,
        bh,
        "轻量分类器训练与固化",
        ["部署主方案：Softmax 回归", "导出模型、标准化参数与特征子集"],
    )

    add_round_box(
        ax,
        x1,
        bot_y,
        bw,
        bh,
        "输入",
        ["单车事件片段", "由事件检测模块输出"],
    )
    add_round_box(
        ax,
        x2,
        bot_y,
        bw,
        bh,
        "统一预处理与特征计算",
        ["按训练口径标准化", "仅计算 $K^{\\ast}=5$ 特征"],
    )
    add_round_box(
        ax,
        x3,
        bot_y,
        bw,
        bh,
        "轻量分类器推理与输出",
        ["Softmax 回归", "输出类别标签与置信度"],
    )

    # Horizontal arrows
    y_top_mid = top_y + bh / 2
    y_bot_mid = bot_y + bh / 2
    add_arrow(ax, (x1 + bw, y_top_mid), (x2, y_top_mid))
    add_arrow(ax, (x2 + bw, y_top_mid), (x3, y_top_mid))
    add_arrow(ax, (x1 + bw, y_bot_mid), (x2, y_bot_mid))
    add_arrow(ax, (x2 + bw, y_bot_mid), (x3, y_bot_mid))

    # Deployment arrow
    x_dep = x3 + bw / 2
    add_arrow(ax, (x_dep, top_y), (x_dep, bot_y + bh), lw=1.4, ls=(0, (4, 4)), color="#444444", ms=20)
    ax.text(
        x_dep + 0.010,
        0.49,
        "部署：模型参数、标准化参数\n与特征子集",
        ha="left",
        va="center",
        fontsize=12.6,
        linespacing=1.25,
        color="#333333",
    )

    fig.tight_layout(pad=0.12)
    out_no_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_no_ext.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(out_no_ext.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_no_ext.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    base = Path(__file__).resolve().parents[3]
    out_no_ext = base / "images" / "ch3_baseline_flow"
    draw_flowchart(out_no_ext)
    print(f"[OK] Wrote {out_no_ext.with_suffix('.png')}")
    print(f"[OK] Wrote {out_no_ext.with_suffix('.pdf')}")
    print(f"[OK] Wrote {out_no_ext.with_suffix('.svg')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
