import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd

# keep CPU deterministic / avoid threading stalls
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import run_dtw_multi_sweep as M
import run_dtw_clsmin_dba as DBA


def load_xyz_y(mat_path: str):
    vars = M.load_vars(mat_path, {'ProcessedData','targetLength'})
    PD=vars['ProcessedData']; TL=vars['targetLength']
    xyz_list=[]; y_list=[]
    for c in range(3):
        tl=np.array(TL[c]).reshape(-1)
        for i in range(len(PD[c])):
            T=int(tl[i])
            xyz=np.asarray(PD[c][i],dtype=np.float32)[:T,:]
            xyz_list.append(xyz)
            y_list.append(c)
    y=np.asarray(y_list,np.int64)
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
    N = len(xyz_list)
    X = np.zeros((N, 4 * L_feat), np.float32)
    for i, xyz in enumerate(xyz_list):
        xr = resample_linear(xyz, L_feat)
        mag = np.sqrt(np.sum(xr * xr, axis=1, dtype=np.float32))
        X[i] = np.concatenate([xr[:,0], xr[:,1], xr[:,2], mag], axis=0)
    return X


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def run_edge(xyz_list, y, idx_tr, idx_va, idx_te, seed: int, L_feat: int, C_lr: float, C_svm: float):
    X = build_edge_feature_matrix(xyz_list, L_feat)
    mu, sd = standardize_fit(X[idx_tr])
    Xz = standardize_apply(X, mu, sd)

    Xtr, ytr = Xz[idx_tr], y[idx_tr]
    Xva, yva = Xz[idx_va], y[idx_va]
    Xte, yte = Xz[idx_te], y[idx_te]

    lr = LogisticRegression(
        multi_class='multinomial', solver='lbfgs', max_iter=2000,
        C=float(C_lr), n_jobs=1, random_state=int(seed)
    )
    lr.fit(Xtr, ytr)
    yva_pred = lr.predict(Xva)
    yte_pred = lr.predict(Xte)
    lr_row = dict(method='Edge Softmax', seed=int(seed),
                  val_acc=float(np.mean(yva_pred==yva)), val_macroF1=macro_f1(yva,yva_pred,3),
                  test_acc=float(np.mean(yte_pred==yte)), test_macroF1=macro_f1(yte,yte_pred,3),
                  L_feat=int(L_feat), C=float(C_lr))

    svm = LinearSVC(C=float(C_svm), random_state=int(seed))
    svm.fit(Xtr, ytr)
    yva_pred = svm.predict(Xva)
    yte_pred = svm.predict(Xte)
    svm_row = dict(method='Edge SVM', seed=int(seed),
                   val_acc=float(np.mean(yva_pred==yva)), val_macroF1=macro_f1(yva,yva_pred,3),
                   test_acc=float(np.mean(yte_pred==yte)), test_macroF1=macro_f1(yte,yte_pred,3),
                   L_feat=int(L_feat), C=float(C_svm))

    return lr_row, svm_row


def run_train_eval(X_raw, y, idx_tr, idx_va, idx_te, in_ch, seed_train):
    mu,sd=M.compute_norm_stats(X_raw, idx_tr)
    X=M.normalize(X_raw, mu, sd)
    M.set_seed(seed_train)
    acc_va,f1_va,acc_te,f1_te=M.train_model(X,y,idx_tr,idx_va,idx_te,in_ch=in_ch, seed=seed_train)
    return float(acc_va), float(f1_va), float(acc_te), float(f1_te)


def main():
    ap=argparse.ArgumentParser(description='Multi-seed stability (v2): only MAIN methods (Edge Softmax/SVM + Offline CNN + ClsMin-DBA + MultiTemplate).')
    ap.add_argument('--mat', required=True)
    ap.add_argument('--L', type=int, default=176)
    ap.add_argument('--L_feat', type=int, default=128)
    ap.add_argument('--C_lr', type=float, default=1.0)
    ap.add_argument('--C_svm', type=float, default=1.0)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9])
    ap.add_argument('--out_dir', default='.')
    # fixed configs
    ap.add_argument('--dba_wR', type=float, default=0.15)
    ap.add_argument('--dba_step', type=float, default=0.05)
    ap.add_argument('--dba_iter', type=int, default=5)
    ap.add_argument('--tpc', type=int, default=4)
    ap.add_argument('--wR', type=float, default=0.15)
    ap.add_argument('--step', type=float, default=0.05)
    args=ap.parse_args()

    out_dir=Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz_list,y=load_xyz_y(args.mat)
    X_raw=M.build_baseline_data(xyz_list, args.L)

    rows=[]
    for s in args.seeds:
        idx_tr, idx_va, idx_te = M.stratified_split(y, seed=s)

        # edge baselines
        lr_row, svm_row = run_edge(xyz_list,y,idx_tr,idx_va,idx_te,seed=s,L_feat=args.L_feat,C_lr=args.C_lr,C_svm=args.C_svm)
        rows.append(lr_row)
        rows.append(svm_row)

        # offline baseline
        va_acc,va_f1,te_acc,te_f1 = run_train_eval(X_raw,y,idx_tr,idx_va,idx_te,in_ch=4, seed_train=s)
        rows.append(dict(seed=s, method='Baseline CNN (offline, 4ch)', val_acc=va_acc, val_macroF1=va_f1, test_acc=te_acc, test_macroF1=te_f1, L=args.L))

        # clsmin DBA
        Xd = DBA.build_dtw_clsmin_dba(X_raw,y,idx_tr, wR=args.dba_wR, step=args.dba_step, n_iter=args.dba_iter)
        va_acc,va_f1,te_acc,te_f1 = run_train_eval(Xd,y,idx_tr,idx_va,idx_te,in_ch=4, seed_train=s)
        rows.append(dict(seed=s, method='DTW-ClsMin-DBA + CNN', val_acc=va_acc, val_macroF1=va_f1, test_acc=te_acc, test_macroF1=te_f1,
                         wR=args.dba_wR, step=args.dba_step, n_iter=args.dba_iter))

        # multi
        Xm = M.build_dtw_multi(X_raw,y,idx_tr, templates_per_class=args.tpc, wR=args.wR, step=args.step, seed=s)
        va_acc,va_f1,te_acc,te_f1 = run_train_eval(Xm,y,idx_tr,idx_va,idx_te,in_ch=Xm.shape[1], seed_train=s)
        rows.append(dict(seed=s, method='DTW-MultiTemplate + CNN', val_acc=va_acc, val_macroF1=va_f1, test_acc=te_acc, test_macroF1=te_f1,
                         tpc=args.tpc, wR=args.wR, step=args.step, in_ch=int(Xm.shape[1])))

    df=pd.DataFrame(rows)

    agg=df.groupby('method').agg(
        test_acc_mean=('test_acc','mean'), test_acc_std=('test_acc','std'),
        test_f1_mean=('test_macroF1','mean'), test_f1_std=('test_macroF1','std'),
        n=('seed','count')
    ).reset_index()

    tag=f'L{args.L}_seeds{len(args.seeds)}'
    df.to_csv(out_dir/f'ch3_multiseed_raw_{tag}.csv', index=False)
    agg.to_csv(out_dir/f'ch3_multiseed_summary_{tag}.csv', index=False)
    with open(out_dir/f'ch3_multiseed_summary_{tag}.json','w',encoding='utf-8') as f:
        json.dump(agg.to_dict(orient='records'),f,ensure_ascii=False,indent=2)

    print(agg.to_string(index=False))
    print('\nSaved to', out_dir)


if __name__=='__main__':
    main()
