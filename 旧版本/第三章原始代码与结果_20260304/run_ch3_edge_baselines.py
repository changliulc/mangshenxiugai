import os, argparse
from pathlib import Path
import numpy as np

# Keep CPU deterministic / avoid BLAS threading stalls in some environments
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('NUMEXPR_NUM_THREADS','1')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# Reuse MAT loader + split util from the DTW/CNN scripts
import run_dtw_multi_sweep as M


def resample_linear(x: np.ndarray, L: int) -> np.ndarray:
    """(T,3) -> (L,3) linear interpolation."""
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


def build_feature_matrix(xyz_list, L_feat: int) -> np.ndarray:
    """Build edge-friendly feature vector by flattening xyz+mag after resampling.

    Output: X shape (N, 4*L_feat)
    """
    N = len(xyz_list)
    X = np.zeros((N, 4 * L_feat), np.float32)
    for i, xyz in enumerate(xyz_list):
        xr = resample_linear(xyz, L_feat)
        mag = np.sqrt(np.sum(xr * xr, axis=1, dtype=np.float32))
        X[i] = np.concatenate([xr[:, 0], xr[:, 1], xr[:, 2], mag], axis=0)
    return X


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, C: int = 3) -> float:
    f1 = []
    for c in range(C):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1.append(f)
    return float(np.mean(f1))


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def main():
    ap = argparse.ArgumentParser(description='Edge baselines (Softmax LR / Linear SVM / Decision Tree) using the SAME stratified split as DTW+CNN experiments.')
    ap.add_argument('--mat', required=True, help='path to .mat (ProcessedData + targetLength)')
    ap.add_argument('--L_feat', type=int, default=128, help='resample length for edge feature vector')
    ap.add_argument('--seed', type=int, default=42, help='split seed (same as offline experiments)')
    ap.add_argument('--out_dir', default='.', help='output directory')
    ap.add_argument('--C_lr', type=float, default=1.0, help='inverse regularization strength for LogisticRegression')
    ap.add_argument('--C_svm', type=float, default=1.0, help='C for LinearSVC')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load xyz_list, y
    vars = M.load_vars(args.mat, {'ProcessedData','targetLength'})
    PD = vars['ProcessedData']; TL = vars['targetLength']

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

    # split (match offline DTW/CNN protocol)
    idx_tr, idx_va, idx_te = M.stratified_split(y, seed=args.seed)

    # build features
    X = build_feature_matrix(xyz_list, args.L_feat)
    mu, sd = standardize_fit(X[idx_tr])
    Xz = standardize_apply(X, mu, sd)

    Xtr, ytr = Xz[idx_tr], y[idx_tr]
    Xva, yva = Xz[idx_va], y[idx_va]
    Xte, yte = Xz[idx_te], y[idx_te]

    # --- Softmax baseline (multinomial logistic regression) ---
    lr = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=2000,
        C=float(args.C_lr),
        n_jobs=1,
        random_state=args.seed,
    )
    lr.fit(Xtr, ytr)

    yva_pred = lr.predict(Xva)
    yte_pred = lr.predict(Xte)
    va_acc = float(np.mean(yva_pred == yva))
    te_acc = float(np.mean(yte_pred == yte))
    va_f1 = macro_f1(yva, yva_pred, 3)
    te_f1 = macro_f1(yte, yte_pred, 3)

    # probabilities (softmax) for confidence analysis if needed
    te_prob = lr.predict_proba(Xte)
    te_conf = te_prob.max(axis=1)

    out_lr = {
        'method': 'Edge Softmax (LogReg)',
        'seed': int(args.seed),
        'L_feat': int(args.L_feat),
        'C_lr': float(args.C_lr),
        'val_acc': va_acc,
        'val_macroF1': va_f1,
        'test_acc': te_acc,
        'test_macroF1': te_f1,
        'test_conf_mean': float(np.mean(te_conf)),
        'test_conf_p05': float(np.quantile(te_conf, 0.05)),
        'test_conf_p50': float(np.quantile(te_conf, 0.50)),
        'test_conf_p95': float(np.quantile(te_conf, 0.95)),
    }

    # --- Linear SVM baseline (no probability by default) ---
    svm = LinearSVC(C=float(args.C_svm), random_state=args.seed)
    svm.fit(Xtr, ytr)

    yva_pred = svm.predict(Xva)
    yte_pred = svm.predict(Xte)
    va_acc2 = float(np.mean(yva_pred == yva))
    te_acc2 = float(np.mean(yte_pred == yte))
    va_f12 = macro_f1(yva, yva_pred, 3)
    te_f12 = macro_f1(yte, yte_pred, 3)

    out_svm = {
        'method': 'Edge LinearSVM',
        'seed': int(args.seed),
        'L_feat': int(args.L_feat),
        'C_svm': float(args.C_svm),
        'val_acc': va_acc2,
        'val_macroF1': va_f12,
        'test_acc': te_acc2,
        'test_macroF1': te_f12,
    }

    # --- Decision Tree control (hyperparams selected on val) ---
    best = None
    for max_depth in [3, 4, 5, 6, 8, None]:
        for min_leaf in [1, 2, 5, 10]:
            dt = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_leaf,
                random_state=args.seed,
            )
            dt.fit(Xtr, ytr)
            yva_pred = dt.predict(Xva)
            f1 = macro_f1(yva, yva_pred, 3)
            if best is None or f1 > best['val_macroF1'] + 1e-9:
                best = {
                    'max_depth': max_depth,
                    'min_samples_leaf': min_leaf,
                    'val_macroF1': float(f1),
                    'model': dt,
                }

    dt = best['model']
    yva_pred = dt.predict(Xva)
    yte_pred = dt.predict(Xte)
    out_dt = {
        'method': 'Edge DecisionTree',
        'seed': int(args.seed),
        'L_feat': int(args.L_feat),
        'max_depth': str(best['max_depth']),
        'min_samples_leaf': int(best['min_samples_leaf']),
        'val_acc': float(np.mean(yva_pred == yva)),
        'val_macroF1': macro_f1(yva, yva_pred, 3),
        'test_acc': float(np.mean(yte_pred == yte)),
        'test_macroF1': macro_f1(yte, yte_pred, 3),
    }

    # save
    import json
    with open(out_dir / f'ch3_edge_baselines_seed{args.seed}_Lfeat{args.L_feat}.json', 'w', encoding='utf-8') as f:
        json.dump({'softmax': out_lr, 'svm': out_svm, 'dt': out_dt}, f, ensure_ascii=False, indent=2)

    print('[Softmax(LogReg)]', out_lr)
    print('[LinearSVM]      ', out_svm)
    print('[DecisionTree]   ', out_dt)
    print('Saved to', out_dir)


if __name__ == '__main__':
    main()
