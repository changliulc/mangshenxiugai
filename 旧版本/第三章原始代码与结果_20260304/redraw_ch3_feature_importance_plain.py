#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager as fm

import run_ch3_thesis_pipeline as TP


FONT_FAMILIES = ["Times New Roman", "SimSun", "STSong", "DejaVu Serif"]
XLABEL_FS = 15.5
YTICK_FS = 14.2
XTICK_FS = 13.2

LABEL_MAP = {
    "x_absarea": "\u0058 \u8f74\u7edd\u5bf9\u9762\u79ef",
    "y_absarea": "\u0059 \u8f74\u7edd\u5bf9\u9762\u79ef",
    "z_absarea": "\u005a \u8f74\u7edd\u5bf9\u9762\u79ef",
    "b_absarea": "\u6a21\u503c\u7edd\u5bf9\u9762\u79ef",
    "x_energy": "\u0058 \u8f74\u80fd\u91cf",
    "y_energy": "\u0059 \u8f74\u80fd\u91cf",
    "z_energy": "\u005a \u8f74\u80fd\u91cf",
    "b_energy": "\u6a21\u503c\u80fd\u91cf",
    "len_T_sec_fs50": "\u4e8b\u4ef6\u6301\u7eed\u65f6\u95f4",
    "event_duration": "\u4e8b\u4ef6\u6301\u7eed\u65f6\u95f4",
    "len_N": "\u4e8b\u4ef6\u6301\u7eed\u70b9\u6570",
    "corr_xy": "\u0058\u0059 \u76f8\u5173\u7cfb\u6570",
    "corr_xz": "\u0058\u005a \u76f8\u5173\u7cfb\u6570",
    "corr_yz": "\u0059\u005a \u76f8\u5173\u7cfb\u6570",
    "x_range": "\u0058 \u8f74\u6781\u5dee",
    "y_range": "\u0059 \u8f74\u6781\u5dee",
    "z_range": "\u005a \u8f74\u6781\u5dee",
    "x_std": "\u0058 \u8f74\u6807\u51c6\u5dee",
    "y_std": "\u0059 \u8f74\u6807\u51c6\u5dee",
    "z_std": "\u005a \u8f74\u6807\u51c6\u5dee",
    "b_std_over_mean": "\u6a21\u503c\u53d8\u5f02\u7cfb\u6570",
    "b_mean": "\u6a21\u503c\u5747\u503c",
    "b_max": "\u6a21\u503c\u5cf0\u503c",
    "b_rms": "\u6a21\u503c\u5747\u65b9\u6839",
    "x_rms": "\u0058 \u8f74\u5747\u65b9\u6839",
    "y_rms": "\u0059 \u8f74\u5747\u65b9\u6839",
    "z_rms": "\u005a \u8f74\u5747\u65b9\u6839",
    "b_range": "\u6a21\u503c\u6781\u5dee",
    "x_max": "\u0058 \u8f74\u5cf0\u503c",
    "y_max": "\u0059 \u8f74\u5cf0\u503c",
    "z_max": "\u005a \u8f74\u5cf0\u503c",
    "zc_z": "\u005a \u8f74\u8fc7\u96f6\u6b21\u6570",
}


def configure_fonts() -> None:
    plt.rcParams["font.family"] = FONT_FAMILIES
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["mathtext.fontset"] = "stix"


def mixed_font(size: float | None = None) -> fm.FontProperties:
    props = fm.FontProperties(family=FONT_FAMILIES)
    if size is not None:
        props.set_size(size)
    return props


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--topn", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    df = pd.read_csv(csv_path)
    df = df.sort_values("importance", ascending=False, kind="stable").head(args.topn).copy()
    df["label_zh"] = df["feature"].map(LABEL_MAP).fillna(df["feature"])
    df = df.iloc[::-1].copy()

    TP.ensure_plot_style()
    configure_fonts()
    fig_h = max(6.2, 0.40 * len(df) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))

    ax.barh(
        df["label_zh"],
        df["importance"],
        color="#1f77b4",
        edgecolor="none",
        alpha=0.95,
    )
    ax.set_xlabel(
        "\u91cd\u8981\u6027\u503c\uff08\u968f\u673a\u68ee\u6797\uff09",
        fontsize=XLABEL_FS,
        fontproperties=mixed_font(XLABEL_FS),
    )
    ax.tick_params(axis="y", labelsize=YTICK_FS, length=5.0, width=1.0)
    ax.tick_params(axis="x", labelsize=XTICK_FS, length=5.0, width=1.0)
    ax.grid(axis="x", linestyle="--", alpha=0.28)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
