import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')

import run_dtw_multi_sweep as M
import run_dtw_clsmin_sweep as C
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


def run_train_eval(X_raw, y, idx_tr, idx_va, idx_te, in_ch, seed_train):
    mu,sd=M.compute_norm_stats(X_raw, idx_tr)
    X=M.normalize(X_raw, mu, sd)
    M.set_seed(seed_train)
    acc_va,f1_va,acc_te,f1_te=M.train_model(X,y,idx_tr,idx_va,idx_te,in_ch=in_ch, seed=seed_train)
    return float(acc_va), float(f1_va), float(acc_te), float(f1_te)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mat', required=True)
    ap.add_argument('--L', type=int, default=176)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0,1,2,3,4,5,6,7,8,9])
    ap.add_argument('--out_dir', default='.')
    args=ap.parse_args()

    out_dir=Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz_list,y=load_xyz_y(args.mat)
    X_raw=M.build_baseline_data(xyz_list, args.L)

    # fixed configs picked from seed=42 sweeps
    cfg_cls_mean=dict(wR=0.10, step=0.05)
    cfg_cls_dba=dict(wR=0.15, step=0.05, n_iter=5)
    cfg_multi=dict(tpc=4, wR=0.15, step=0.05)

    rows=[]
    for s in args.seeds:
        idx_tr, idx_va, idx_te = M.stratified_split(y, seed=s)

        # baseline
        va_acc,va_f1,te_acc,te_f1 = run_train_eval(X_raw,y,idx_tr,idx_va,idx_te,in_ch=4, seed_train=s)
        rows.append(dict(seed=s, method='Baseline CNN', val_acc=va_acc, val_f1=va_f1, test_acc=te_acc, test_f1=te_f1))

        # clsmin mean
        Xc=C.build_dtw_clsmin(X_raw,y,idx_tr, wR=cfg_cls_mean['wR'], step=cfg_cls_mean['step'])
        va_acc,va_f1,te_acc,te_f1 = run_train_eval(Xc,y,idx_tr,idx_va,idx_te,in_ch=4, seed_train=s)
        rows.append(dict(seed=s, method='DTW-ClsMin-Mean', val_acc=va_acc, val_f1=va_f1, test_acc=te_acc, test_f1=te_f1,
                         wR=cfg_cls_mean['wR'], step=cfg_cls_mean['step']))

        # clsmin DBA
        Xd=DBA.build_dtw_clsmin_dba(X_raw,y,idx_tr, wR=cfg_cls_dba['wR'], step=cfg_cls_dba['step'], n_iter=cfg_cls_dba['n_iter'])
        va_acc,va_f1,te_acc,te_f1 = run_train_eval(Xd,y,idx_tr,idx_va,idx_te,in_ch=4, seed_train=s)
        rows.append(dict(seed=s, method='DTW-ClsMin-DBA', val_acc=va_acc, val_f1=va_f1, test_acc=te_acc, test_f1=te_f1,
                         wR=cfg_cls_dba['wR'], step=cfg_cls_dba['step'], n_iter=cfg_cls_dba['n_iter']))

        # multi
        Xm=M.build_dtw_multi(X_raw,y,idx_tr, templates_per_class=cfg_multi['tpc'], wR=cfg_multi['wR'], step=cfg_multi['step'], seed=s)
        va_acc,va_f1,te_acc,te_f1 = run_train_eval(Xm,y,idx_tr,idx_va,idx_te,in_ch=Xm.shape[1], seed_train=s)
        rows.append(dict(seed=s, method='DTW-MultiTemplate', val_acc=va_acc, val_f1=va_f1, test_acc=te_acc, test_f1=te_f1,
                         tpc=cfg_multi['tpc'], wR=cfg_multi['wR'], step=cfg_multi['step'], in_ch=int(Xm.shape[1])))

    df=pd.DataFrame(rows)

    # mean±std summary per method
    agg=df.groupby('method').agg(
        test_acc_mean=('test_acc','mean'), test_acc_std=('test_acc','std'),
        test_f1_mean=('test_f1','mean'), test_f1_std=('test_f1','std'),
        val_f1_mean=('val_f1','mean'), val_f1_std=('val_f1','std'),
        n=('seed','count')
    ).reset_index()

    tag=f'L{args.L}_seeds{len(args.seeds)}'
    df.to_csv(out_dir/f'ch3_multiseed_raw_{tag}.csv', index=False)
    agg.to_csv(out_dir/f'ch3_multiseed_summary_{tag}.csv', index=False)
    with open(out_dir/f'ch3_multiseed_summary_{tag}.json','w') as f:
        json.dump(agg.to_dict(orient='records'),f,ensure_ascii=False,indent=2)

    print(agg.to_string(index=False))
    print('\nSaved to', out_dir)


if __name__=='__main__':
    main()
