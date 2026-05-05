#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据 ch3_feature_selection_consistent_out/ch3_feature_importance_consistent.csv
重绘“端侧候选统计特征重要性排序”图。

用途：
1. 解决中文字体在部分环境中显示为方框的问题；
2. 保证图中的中文标签、最终保留特征高亮、坐标轴标题与论文正文一致；
3. 不需要重新跑完整特征筛选流程，只要有 CSV 即可重画图。

推荐用法（Windows PowerShell）：
python .\plot_ch3_feature_importance_authoritative_local.py `
  --csv ".\out_ch3_feature_selection_consistent\ch3_feature_importance_consistent.csv" `
  --out ".\out_ch3_feature_selection_consistent\ch3_feature_importance_authoritative.png"
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch


FEATURE_LABEL_ZH = {
    # 推荐版最终正文口径
    'x_absarea': 'X 轴绝对面积',
    'y_absarea': 'Y 轴绝对面积',
    'z_absarea': 'Z 轴绝对面积',
    'x_energy': 'X 轴能量',
    'y_energy': 'Y 轴能量',
    'z_energy': 'Z 轴能量',
    'b_energy': '模值能量',
    'b_max': '模值峰值',
    'b_mean': '模值均值',
    'b_rms': '模值均方根',
    'b_range': '模值极差',
    'x_std': 'X 轴标准差',
    'y_std': 'Y 轴标准差',
    'z_std': 'Z 轴标准差',
    'x_range': 'X 轴极差',
    'y_range': 'Y 轴极差',
    'z_range': 'Z 轴极差',
    'x_absmean': 'X 轴绝对均值',
    'y_absmean': 'Y 轴绝对均值',
    'z_absmean': 'Z 轴绝对均值',
    'corr_xy': 'XY 相关系数',
    'corr_xz': 'XZ 相关系数',
    'corr_yz': 'YZ 相关系数',
    'len_T_sec_fs50': '事件持续时间',
    'event_duration': '事件持续时间',
}


def configure_chinese_font(preferred: str | None = None) -> str | None:
    """配置中文字体，优先使用用户本地常见字体。"""
    candidates = []
    if preferred:
        candidates.append(preferred)
    candidates.extend([
        'Microsoft YaHei',
        'SimHei',
        'Noto Sans CJK SC',
        'Source Han Sans SC',
        'PingFang SC',
        'WenQuanYi Zen Hei',
        'Arial Unicode MS',
    ])

    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in available:
            chosen = name
            break

    if chosen is not None:
        plt.rcParams['font.sans-serif'] = [chosen] + plt.rcParams.get('font.sans-serif', [])
    plt.rcParams['axes.unicode_minus'] = False
    return chosen


def infer_label_zh(feature: str) -> str:
    return FEATURE_LABEL_ZH.get(feature, feature)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True, help='ch3_feature_importance_consistent.csv 路径')
    ap.add_argument('--out', required=True, help='输出 PNG 路径')
    ap.add_argument('--font', default='', help='可选：指定中文字体名称，例如 Microsoft YaHei')
    ap.add_argument('--topk', type=int, default=0, help='可选：只显示前 topk 个特征；0 表示全部显示')
    ap.add_argument('--dpi', type=int, default=220)
    args = ap.parse_args()

    chosen_font = configure_chinese_font(args.font if args.font else None)

    df = pd.read_csv(args.csv)
    if 'label_zh' not in df.columns:
        df['label_zh'] = df['feature'].map(infer_label_zh)
    else:
        # 仍然用统一口径兜底一遍，避免旧 CSV 里的 label_zh 与正文不一致
        df['label_zh'] = df['feature'].map(infer_label_zh).fillna(df['label_zh'])

    if 'selected_final' not in df.columns:
        df['selected_final'] = False

    df = df.sort_values('importance_mean', ascending=False).reset_index(drop=True)
    if args.topk and args.topk > 0:
        df = df.iloc[:args.topk].copy()

    # 为了让最重要的特征显示在最上方，反转顺序
    df_plot = df.iloc[::-1].copy()

    colors = ['#d95f02' if bool(v) else '#1f77b4' for v in df_plot['selected_final']]

    # 画布高度随特征数量自适应
    n = len(df_plot)
    fig_h = max(5.8, 0.38 * n + 1.6)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))

    ax.barh(
        df_plot['label_zh'],
        df_plot['importance_mean'],
        color=colors,
        edgecolor='none',
        alpha=0.95,
    )

    ax.set_title('端侧候选统计特征重要性排序', fontsize=17)
    ax.set_xlabel('重要性值（随机森林）', fontsize=14)
    ax.tick_params(axis='y', labelsize=13)
    ax.tick_params(axis='x', labelsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.28)

    legend_handles = [
        Patch(facecolor='#1f77b4', label='候选特征'),
        Patch(facecolor='#d95f02', label='最终保留特征'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', frameon=True, fontsize=12)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)

    print(f'[OK] saved: {out}')
    if chosen_font:
        print(f'[INFO] using font: {chosen_font}')
    else:
        print('[WARN] no preferred Chinese font found; current environment may still show tofu boxes.')


if __name__ == '__main__':
    main()
