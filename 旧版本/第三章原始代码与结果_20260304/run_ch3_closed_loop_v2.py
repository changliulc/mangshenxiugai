import os, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Single-thread safety (avoid BLAS threading stalls in some environments)
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('NUMEXPR_NUM_THREADS','1')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Reuse core DTW/CNN utilities from the existing handoff scripts
import run_dtw_multi_sweep as M
import run_dtw_clsmin_sweep as C
import run_dtw_clsmin_dba as DBA


def load_xyz_y(mat_path: str):
    """Load xyz_list and labels y from .mat.

    Required vars: ProcessedData, targetLength
    - ProcessedData: 3-class cell, each entry is list/array of (T,3)
    - targetLength: 3-class cell, each entry is list/array of valid lengths
    """
    vars = M.load_vars(mat_path, {'ProcessedData','targetLength'})
    PD = vars['ProcessedData']
    TL = vars['targetLength']

    xyz_list = []
    y_list = []
    for c in range(3):
        tl = np.array(TL[c]).reshape(-1)
        for i in range(len(PD[c])):
            T = int(tl[i])
            xyz = np.asarray(PD[c][i], dtype=np.float32)[:T, :]
            xyz_list.append(xyz)
            y_list.append(c)
    y = np.asarray(y_list, np.int64)
    return xyz_list, y


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, Cn: int = 3) -> float:
    f1 = []
    for c in range(Cn):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1.append(f)
    return float(np.mean(f1))


def resample_linear(x: np.ndarray, L: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    T, D = x.shape
    if T == L:
        return x
    t0 = np.linspace(0.0, 1.0, T)
    t1 = np.linspace(0.0, 1.0, L)
    y = np.empty((L, D), np.float32)
    for d in range(D):
        y[:, d] = np.interp(t1, t0, x[:, d]).astype(np.float32)
    return y


def build_edge_feature_matrix(xyz_list, L_feat: int) -> np.ndarray:
    """Edge-friendly feature vector by flattening xyz+mag after resampling.

    Output X shape: (N, 4*L_feat)
    """
    N = len(xyz_list)
    X = np.zeros((N, 4 * L_feat), np.float32)
    for i, xyz in enumerate(xyz_list):
        xr = resample_linear(xyz, L_feat)
        mag = np.sqrt(np.sum(xr * xr, axis=1, dtype=np.float32))
        X[i] = np.concatenate([xr[:, 0], xr[:, 1], xr[:, 2], mag], axis=0)
    return X


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def run_edge_baselines(xyz_list, y, idx_tr, idx_va, idx_te, seed: int, L_feat: int, C_lr: float, C_svm: float):
    X = build_edge_feature_matrix(xyz_list, L_feat)
    mu, sd = standardize_fit(X[idx_tr])
    Xz = standardize_apply(X, mu, sd)

    Xtr, ytr = Xz[idx_tr], y[idx_tr]
    Xva, yva = Xz[idx_va], y[idx_va]
    Xte, yte = Xz[idx_te], y[idx_te]

    # Softmax (multinomial Logistic Regression)
    lr = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=2000,
        C=float(C_lr),
        n_jobs=1,
        random_state=int(seed),
    )
    lr.fit(Xtr, ytr)

    yva_pred = lr.predict(Xva)
    yte_pred = lr.predict(Xte)
    out_lr = dict(
        method='Edge Softmax',
        L_feat=int(L_feat),
        C=float(C_lr),
        val_acc=float(np.mean(yva_pred == yva)),
        val_macroF1=macro_f1(yva, yva_pred, 3),
        test_acc=float(np.mean(yte_pred == yte)),
        test_macroF1=macro_f1(yte, yte_pred, 3),
    )

    # Linear SVM
    svm = LinearSVC(C=float(C_svm), random_state=int(seed))
    svm.fit(Xtr, ytr)
    yva_pred = svm.predict(Xva)
    yte_pred = svm.predict(Xte)
    out_svm = dict(
        method='Edge SVM',
        L_feat=int(L_feat),
        C=float(C_svm),
        val_acc=float(np.mean(yva_pred == yva)),
        val_macroF1=macro_f1(yva, yva_pred, 3),
        test_acc=float(np.mean(yte_pred == yte)),
        test_macroF1=macro_f1(yte, yte_pred, 3),
    )

    return out_lr, out_svm


def run_cnn(X_raw, y, idx_tr, idx_va, idx_te, in_ch: int, seed_train: int):
    mu, sd = M.compute_norm_stats(X_raw, idx_tr)
    X = M.normalize(X_raw, mu, sd)
    M.set_seed(seed_train)
    acc_va, f1_va, acc_te, f1_te = M.train_model(X, y, idx_tr, idx_va, idx_te, in_ch=in_ch, seed=seed_train)
    return dict(val_acc=float(acc_va), val_macroF1=float(f1_va), test_acc=float(acc_te), test_macroF1=float(f1_te))


def sweep_clsmin_mean(X_raw, y, idx_tr, idx_va, idx_te, seed_train: int, legacy_rng: bool):
    configs = [
        (0.10, 0.0),
        (0.15, 0.0),
        (0.20, 0.0),
        (0.10, 0.05),
        (0.15, 0.05),
        (0.20, 0.05),
        (0.15, 0.10),
    ]
    rows = []
    if legacy_rng:
        M.set_seed(seed_train)
    for wR, step in configs:
        t0 = time.time()
        Xal_raw = C.build_dtw_clsmin(X_raw, y, idx_tr, wR=wR, step=step)
        build_s = time.time() - t0
        mu, sd = M.compute_norm_stats(Xal_raw, idx_tr)
        Xal = M.normalize(Xal_raw, mu, sd)
        if not legacy_rng:
            M.set_seed(seed_train)
        acc_va, f1_va, acc_te, f1_te = M.train_model(Xal, y, idx_tr, idx_va, idx_te, in_ch=4, seed=seed_train)
        rows.append(dict(
            wR=float(wR), step=float(step), build_s=float(build_s),
            val_acc=float(acc_va), val_macroF1=float(f1_va),
            test_acc=float(acc_te), test_macroF1=float(f1_te),
        ))
    df = pd.DataFrame(rows)
    best_by_val = df.loc[df['val_macroF1'].idxmax()].to_dict()
    return df, best_by_val


def run_clsmin_dba(X_raw, y, idx_tr, idx_va, idx_te, seed_train: int, wR=0.15, step=0.05, n_iter=5):
    t0 = time.time()
    Xd_raw = DBA.build_dtw_clsmin_dba(X_raw, y, idx_tr, wR=wR, step=step, n_iter=n_iter)
    build_s = time.time() - t0
    mu, sd = M.compute_norm_stats(Xd_raw, idx_tr)
    Xd = M.normalize(Xd_raw, mu, sd)
    M.set_seed(seed_train)
    acc_va, f1_va, acc_te, f1_te = M.train_model(Xd, y, idx_tr, idx_va, idx_te, in_ch=4, seed=seed_train)
    return dict(
        wR=float(wR), step=float(step), n_iter=int(n_iter), build_s=float(build_s),
        val_acc=float(acc_va), val_macroF1=float(f1_va),
        test_acc=float(acc_te), test_macroF1=float(f1_te),
    )


def sweep_multi(X_raw, y, idx_tr, idx_va, idx_te, seed_train: int, legacy_rng: bool):
    configs = [
        (3, 0.10, 0.0),
        (3, 0.15, 0.0),
        (3, 0.20, 0.0),
        (3, 0.15, 0.05),
        (4, 0.15, 0.0),
        (4, 0.15, 0.05),
        (5, 0.15, 0.0),
    ]
    rows = []
    if legacy_rng:
        M.set_seed(seed_train)
    for tpc, wR, step in configs:
        t0 = time.time()
        Xm_raw = M.build_dtw_multi(X_raw, y, idx_tr, templates_per_class=tpc, wR=wR, step=step, seed=seed_train)
        build_s = time.time() - t0
        mu, sd = M.compute_norm_stats(Xm_raw, idx_tr)
        Xm = M.normalize(Xm_raw, mu, sd)
        if not legacy_rng:
            M.set_seed(seed_train)
        acc_va, f1_va, acc_te, f1_te = M.train_model(Xm, y, idx_tr, idx_va, idx_te, in_ch=Xm.shape[1], seed=seed_train)
        rows.append(dict(
            tpc=int(tpc), wR=float(wR), step=float(step), in_ch=int(Xm.shape[1]), build_s=float(build_s),
            val_acc=float(acc_va), val_macroF1=float(f1_va),
            test_acc=float(acc_te), test_macroF1=float(f1_te),
        ))
    df = pd.DataFrame(rows)
    best_by_val = df.loc[df['val_macroF1'].idxmax()].to_dict()
    return df, best_by_val


def replication_control(X_raw, y, idx_tr, idx_va, idx_te, seed_train: int, tpc=4):
    T = 3 * int(tpc)
    Xrep_raw = np.tile(X_raw, (1, T, 1))
    mu, sd = M.compute_norm_stats(Xrep_raw, idx_tr)
    Xrep = M.normalize(Xrep_raw, mu, sd)
    M.set_seed(seed_train)
    acc_va, f1_va, acc_te, f1_te = M.train_model(Xrep, y, idx_tr, idx_va, idx_te, in_ch=Xrep.shape[1], seed=seed_train)
    return dict(
        tpc=int(tpc), in_ch=int(Xrep.shape[1]),
        val_acc=float(acc_va), val_macroF1=float(f1_va),
        test_acc=float(acc_te), test_macroF1=float(f1_te),
    )


def main():
    ap = argparse.ArgumentParser(description='Chapter-3 closed-loop runner (v2): main methods table (5 rows) + ablations.')
    ap.add_argument('--mat', required=True, help='path to .mat')
    ap.add_argument('--L', type=int, default=176)
    ap.add_argument('--seed', type=int, default=42, help='split seed + training seed')
    ap.add_argument('--out_dir', default='.')
    ap.add_argument('--L_feat', type=int, default=128, help='edge feature resample length')
    ap.add_argument('--C_lr', type=float, default=1.0)
    ap.add_argument('--C_svm', type=float, default=1.0)
    ap.add_argument('--legacy_rng', type=int, default=1, help='1=reproduce original sweep behavior (seed set once, no reseed per config)')
    # DBA config
    ap.add_argument('--dba_wR', type=float, default=0.15)
    ap.add_argument('--dba_step', type=float, default=0.05)
    ap.add_argument('--dba_iter', type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz_list, y = load_xyz_y(args.mat)
    idx_tr, idx_va, idx_te = M.stratified_split(y, seed=args.seed)

    # --- Edge baselines (Softmax + SVM) ---
    edge_lr, edge_svm = run_edge_baselines(
        xyz_list, y, idx_tr, idx_va, idx_te,
        seed=args.seed, L_feat=args.L_feat,
        C_lr=args.C_lr, C_svm=args.C_svm,
    )

    # --- Offline baseline data ---
    X_raw = M.build_baseline_data(xyz_list, args.L)

    # O1 baseline CNN
    base = run_cnn(X_raw, y, idx_tr, idx_va, idx_te, in_ch=4, seed_train=args.seed)

    # A1 mean-template clsmin sweep (ablation only)
    df_mean, best_mean = sweep_clsmin_mean(X_raw, y, idx_tr, idx_va, idx_te, seed_train=args.seed, legacy_rng=bool(args.legacy_rng))

    # O2 clsmin DBA (main method)
    dba = run_clsmin_dba(X_raw, y, idx_tr, idx_va, idx_te, seed_train=args.seed, wR=args.dba_wR, step=args.dba_step, n_iter=args.dba_iter)

    # O3 multi sweep
    df_mul, best_mul = sweep_multi(X_raw, y, idx_tr, idx_va, idx_te, seed_train=args.seed, legacy_rng=bool(args.legacy_rng))

    # A2 replication control (ablation only)
    tpc_best = int(best_mul['tpc']) if 'tpc' in best_mul else 4
    rep = replication_control(X_raw, y, idx_tr, idx_va, idx_te, seed_train=args.seed, tpc=tpc_best)

    # ------------------------
    # Main methods table (5 rows)
    # ------------------------
    main_rows = [
        dict(method=edge_lr.pop('method'), config=f"L_feat={edge_lr['L_feat']} C={edge_lr['C']} seed={args.seed}", **edge_lr),
        dict(method=edge_svm.pop('method'), config=f"L_feat={edge_svm['L_feat']} C={edge_svm['C']} seed={args.seed}", **edge_svm),
        dict(method='Baseline CNN (offline, 4ch)', config=f"L={args.L} seed={args.seed}", **base),
        dict(method='DTW-ClsMin-DBA + CNN', config=f"wR={args.dba_wR} step={args.dba_step} iter={args.dba_iter}", **dba),
        dict(method='DTW-MultiTemplate + CNN', config='best by val_macroF1', **best_mul),
    ]

    df_main = pd.DataFrame(main_rows)

    # ------------------------
    # Ablations
    # ------------------------
    ablation_rows = [
        dict(method='DTW-ClsMin-Mean + CNN (ablation)', config='best by val_macroF1', **best_mean),
        dict(method='DTW-ClsMin-DBA + CNN (ablation ref)', config=f"wR={args.dba_wR} step={args.dba_step} iter={args.dba_iter}", **dba),
        dict(method='Replication Control + CNN (ablation)', config=f"tile to tpc={tpc_best} (no DTW)", **rep),
    ]
    df_ab = pd.DataFrame(ablation_rows)

    tag = f'seed{args.seed}_L{args.L}'

    # Save tables
    df_main.to_csv(out_dir / f'ch3_main_methods_{tag}.csv', index=False)
    with open(out_dir / f'ch3_main_methods_{tag}.json', 'w', encoding='utf-8') as f:
        json.dump(main_rows, f, ensure_ascii=False, indent=2)

    df_ab.to_csv(out_dir / f'ch3_ablations_{tag}.csv', index=False)

    # Save sweeps
    df_mean.to_csv(out_dir / f'ch3_clsmin_mean_sweep_{tag}.csv', index=False)
    df_mul.to_csv(out_dir / f'ch3_multi_sweep_{tag}.csv', index=False)

    print('=== MAIN METHODS (5 rows) ===')
    print(df_main.to_string(index=False))
    print('\n=== ABLATIONS ===')
    print(df_ab.to_string(index=False))
    print('\nSaved to', out_dir)


if __name__ == '__main__':
    main()
