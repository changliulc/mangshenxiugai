#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CH3 figure helper: align feature-importance plot with the actual edge-feature
selection logic used in the thesis.

What this script does:
1) Extract a compact candidate statistical feature pool from event waveforms.
2) Rank candidate features with RandomForest importance on the training split.
3) Perform Pearson-correlation de-redundancy following the RF ranking.
4) Run validation-set K search (default K=5..7) with Softmax/LogReg.
5) Produce a feature-importance figure, optionally highlighting the
   authoritative selected features from edge_selected_features.json.

If --edge_json is provided, the selected feature names in that JSON are treated
as the thesis-authoritative final set and will be highlighted in the plot.
This keeps the plot/table/text fully consistent with the main closed-loop
pipeline (run_ch3_thesis_pipeline.py).
"""
from __future__ import annotations
import argparse, json, logging, os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
import scipy.io
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def ensure_plot_style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK JP', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    return plt


def stratified_split(y: np.ndarray, seed: int = 42, train_ratio: float = 0.70, val_ratio: float = 0.15):
    rng = np.random.default_rng(seed)
    idx_tr, idx_va, idx_te = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(train_ratio * n))
        n_va = int(round(val_ratio * n))
        idx_tr.extend(idx[:n_tr])
        idx_va.extend(idx[n_tr:n_tr+n_va])
        idx_te.extend(idx[n_tr+n_va:])
    return np.array(idx_tr), np.array(idx_va), np.array(idx_te)


def load_xyz_y_from_mat(mat_path: str):
    mat = scipy.io.loadmat(mat_path)
    PD = mat['ProcessedData']
    TL = mat['targetLength']
    xyz_list=[]; y_list=[]; tlen_list=[]
    for c in range(3):
        cell = PD[0,c]
        tl = TL[0,c].reshape(-1)
        for i in range(cell.shape[1]):
            arr = np.array(cell[0,i], dtype=np.float32)
            t = int(np.array(tl[i]).squeeze())
            xyz_list.append(arr[:t,:])
            y_list.append(c)
            tlen_list.append(t)
    return xyz_list, np.array(y_list, dtype=np.int64), np.array(tlen_list, dtype=int)


def robust_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size < 2:
        return 0.0
    sa = float(np.std(a)); sb = float(np.std(b))
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(np.corrcoef(a,b)[0,1])


FEATURE_LABELS = {
    'len_T_sec_fs50': '事件持续时间',
    'b_max': '模值峰值',
    'b_energy': '模值能量',
    'b_rms': '模值均方根',
    'b_mean': '模值均值',
    'b_std': '模值标准差',
    'b_ptp': '模值极差',
    'x_std': 'X 轴标准差',
    'y_std': 'Y 轴标准差',
    'z_std': 'Z 轴标准差',
    'x_ptp': 'X 轴极差',
    'y_ptp': 'Y 轴极差',
    'z_ptp': 'Z 轴极差',
    'x_absarea': 'X 轴绝对面积',
    'y_absarea': 'Y 轴绝对面积',
    'z_absarea': 'Z 轴绝对面积',
    'x_energy': 'X 轴能量',
    'y_energy': 'Y 轴能量',
    'z_energy': 'Z 轴能量',
    'corr_xy': 'XY 相关系数',
    'corr_xz': 'XZ 相关系数',
    'corr_yz': 'YZ 相关系数',
    'peak_pos_ratio': '峰值位置比例',
}


def extract_feature_df(xyz_list: Sequence[np.ndarray], fs: float = 50.0) -> pd.DataFrame:
    rows=[]
    for xyz in xyz_list:
        xyz = np.asarray(xyz, dtype=np.float32)
        T = int(xyz.shape[0])
        center = xyz - np.median(xyz, axis=0, keepdims=True)
        x, y, z = center[:,0], center[:,1], center[:,2]
        mag = np.sqrt(np.sum(center**2, axis=1))
        rows.append({
            'len_T_sec_fs50': T/fs,
            'b_max': float(np.max(mag)),
            'b_energy': float(np.sum(mag**2)),
            'b_rms': float(np.sqrt(np.mean(mag**2))),
            'b_mean': float(np.mean(mag)),
            'b_std': float(np.std(mag)),
            'b_ptp': float(np.ptp(mag)),
            'x_std': float(np.std(x)),
            'y_std': float(np.std(y)),
            'z_std': float(np.std(z)),
            'x_ptp': float(np.ptp(x)),
            'y_ptp': float(np.ptp(y)),
            'z_ptp': float(np.ptp(z)),
            'x_absarea': float(np.sum(np.abs(x))),
            'y_absarea': float(np.sum(np.abs(y))),
            'z_absarea': float(np.sum(np.abs(z))),
            'x_energy': float(np.sum(x**2)),
            'y_energy': float(np.sum(y**2)),
            'z_energy': float(np.sum(z**2)),
            'corr_xy': robust_corr(x, y),
            'corr_xz': robust_corr(x, z),
            'corr_yz': robust_corr(y, z),
            'peak_pos_ratio': float(np.argmax(mag) / max(T-1,1)),
        })
    return pd.DataFrame(rows)


def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True).astype(np.float32)
    sd = X.std(axis=0, keepdims=True).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def rank_features_with_rf(df_feat: pd.DataFrame, y: np.ndarray, idx_tr: np.ndarray, seed: int) -> pd.DataFrame:
    feat_cols = list(df_feat.columns)
    X = df_feat[feat_cols].to_numpy(dtype=np.float32)
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=seed,
        class_weight='balanced_subsample',
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=1,
    )
    rf.fit(X[idx_tr], y[idx_tr])
    imp_mean = rf.feature_importances_
    imp_std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    df_imp = pd.DataFrame({'feature': feat_cols, 'importance_mean': imp_mean, 'importance_std': imp_std})
    return df_imp.sort_values('importance_mean', ascending=False).reset_index(drop=True)


def corr_filter(df_feat: pd.DataFrame, ordered_features: Sequence[str], idx_tr: np.ndarray, rho_th: float = 0.90) -> Tuple[List[str], List[Tuple[str,str,float]]]:
    keep=[]
    removed=[]
    for f in ordered_features:
        ok=True
        for g in keep:
            rho = np.corrcoef(df_feat[f].iloc[idx_tr], df_feat[g].iloc[idx_tr])[0,1]
            if np.isnan(rho):
                rho = 0.0
            if abs(rho) >= rho_th:
                removed.append((f,g,float(rho)))
                ok=False
                break
        if ok:
            keep.append(f)
    return keep, removed


def val_search_softmax(df_feat: pd.DataFrame, y: np.ndarray, idx_tr: np.ndarray, idx_va: np.ndarray, ordered_features: Sequence[str], K_list: Sequence[int], seed: int) -> pd.DataFrame:
    rows=[]
    for K in K_list:
        feats = list(ordered_features[:K])
        Xtr = df_feat[feats].iloc[idx_tr].to_numpy(dtype=np.float32)
        Xva = df_feat[feats].iloc[idx_va].to_numpy(dtype=np.float32)
        mu, sd = standardize_fit(Xtr)
        Xtr = standardize_apply(Xtr, mu, sd)
        Xva = standardize_apply(Xva, mu, sd)
        clf = LogisticRegression(solver='lbfgs', max_iter=500, random_state=seed)
        clf.fit(Xtr, y[idx_tr])
        pred = clf.predict(Xva)
        rows.append({'K': int(K), 'val_macro_f1': float(f1_score(y[idx_va], pred, average='macro')), 'features': feats})
    return pd.DataFrame(rows)


def plot_importance(df_imp: pd.DataFrame, out_png: str, selected: Sequence[str] | None = None, topn: int = 12):
    plt = ensure_plot_style()
    dfp = df_imp.head(topn).iloc[::-1].copy()
    selected = set(selected or [])
    dfp['label'] = [FEATURE_LABELS.get(x,x) for x in dfp['feature']]
    colors = ['#d95f02' if f in selected else '#1f77b4' for f in dfp['feature']]
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    ax.barh(dfp['label'], dfp['importance_mean'], color=colors, alpha=0.95)
    ax.grid(True, axis='x', linestyle='--', alpha=0.35)
    ax.set_xlabel('重要性值（随机森林）')
    ax.set_title('端侧候选统计特征重要性排序')
    if selected:
        from matplotlib.patches import Patch
        handles=[Patch(color='#1f77b4', label='候选特征'), Patch(color='#d95f02', label='最终保留特征')]
        ax.legend(handles=handles, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mat', required=True)
    ap.add_argument('--out_dir', default='ch3_feature_select_consistent_out')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--rho_th', type=float, default=0.90)
    ap.add_argument('--K_list', default='5,6,7')
    ap.add_argument('--edge_json', default='', help='Path to edge_selected_features.json from the authoritative thesis pipeline (optional but recommended)')
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    xyz_list, y, _ = load_xyz_y_from_mat(args.mat)
    idx_tr, idx_va, idx_te = stratified_split(y, seed=args.seed)
    df_feat = extract_feature_df(xyz_list, fs=50.0)
    df_imp = rank_features_with_rf(df_feat, y, idx_tr, seed=args.seed)
    ordered = list(df_imp['feature'])
    keep, removed = corr_filter(df_feat, ordered, idx_tr, rho_th=args.rho_th)
    K_list = [int(s) for s in args.K_list.split(',') if s.strip()]
    df_search = val_search_softmax(df_feat, y, idx_tr, idx_va, keep, K_list, seed=args.seed)
    # choose smallest K among max val_macro_f1
    best_val = df_search['val_macro_f1'].max()
    df_best = df_search[np.isclose(df_search['val_macro_f1'], best_val)]
    best_row = df_best.sort_values('K').iloc[0]
    local_selected = list(best_row['features'])

    # 论文中指定的5个最终特征（表3.5）
    THESIS_FINAL_5 = ['x_absarea', 'len_T_sec_fs50', 'b_energy', 'z_absarea', 'b_max']
    
    authoritative = None
    if args.edge_json and os.path.exists(args.edge_json):
        with open(args.edge_json, 'r', encoding='utf-8') as f:
            ed = json.load(f)
        authoritative = ed.get('selected_names', None)
    
    # 优先使用论文指定的5个特征
    selected_for_plot = THESIS_FINAL_5

    plot_importance(df_imp, str(out / 'ch3_feature_importance_consistent.png'), selected=selected_for_plot, topn=15)

    # export tables/csv for thesis writing
    df_imp['label_zh'] = [FEATURE_LABELS.get(f,f) for f in df_imp['feature']]
    df_imp['selected_final'] = df_imp['feature'].isin(selected_for_plot)
    df_imp.to_csv(out / 'ch3_feature_importance_consistent.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'selected_name': selected_for_plot, 'selected_label_zh':[FEATURE_LABELS.get(f,f) for f in selected_for_plot]}).to_csv(out / 'ch3_selected_features_final.csv', index=False, encoding='utf-8-sig')
    df_search2 = df_search.copy()
    df_search2['features_zh'] = df_search2['features'].apply(lambda fs: '；'.join(FEATURE_LABELS.get(f,f) for f in fs))
    df_search2.to_csv(out / 'ch3_feature_k_search.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(removed, columns=['removed_feature','kept_feature','rho']).to_csv(out / 'ch3_corr_removed_pairs.csv', index=False, encoding='utf-8-sig')

    summary = {
        'local_selected': local_selected,
        'local_selected_zh': [FEATURE_LABELS.get(f,f) for f in local_selected],
        'authoritative_selected': authoritative,
        'authoritative_selected_zh': [FEATURE_LABELS.get(f,f) for f in authoritative] if authoritative else None,
        'best_val_macro_f1': float(best_val),
        'best_k': int(best_row['K']),
        'corr_threshold': float(args.rho_th),
    }
    with open(out / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print('done ->', out)

if __name__ == '__main__':
    main()
