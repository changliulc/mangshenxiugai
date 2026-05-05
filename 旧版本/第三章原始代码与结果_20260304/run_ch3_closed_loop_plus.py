import os, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Single-thread safety (you can tune later)
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('NUMEXPR_NUM_THREADS','1')

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


def run_cnn(X_raw, y, idx_tr, idx_va, idx_te, in_ch: int, seed_train: int):
    mu,sd=M.compute_norm_stats(X_raw, idx_tr)
    X=M.normalize(X_raw, mu, sd)
    M.set_seed(seed_train)
    acc_va,f1_va,acc_te,f1_te=M.train_model(X,y,idx_tr,idx_va,idx_te,in_ch=in_ch, seed=seed_train)
    return dict(val_acc=float(acc_va), val_f1=float(f1_va), test_acc=float(acc_te), test_f1=float(f1_te))


def sweep_clsmin_mean(X_raw, y, idx_tr, idx_va, idx_te, seed_train: int, legacy_rng: bool):
    configs=[
        (0.10,0.0),
        (0.15,0.0),
        (0.20,0.0),
        (0.10,0.05),
        (0.15,0.05),
        (0.20,0.05),
        (0.15,0.10),
    ]
    rows=[]
    if legacy_rng:
        M.set_seed(seed_train)
    for wR,step in configs:
        t0=time.time()
        Xal_raw=C.build_dtw_clsmin(X_raw,y,idx_tr,wR=wR,step=step)
        build_s=time.time()-t0
        mu,sd=M.compute_norm_stats(Xal_raw, idx_tr)
        Xal=M.normalize(Xal_raw, mu, sd)
        if not legacy_rng:
            M.set_seed(seed_train)
        acc_va,f1_va,acc_te,f1_te=M.train_model(Xal,y,idx_tr,idx_va,idx_te,in_ch=4, seed=seed_train)
        rows.append(dict(wR=wR,step=step,build_s=float(build_s),
                         val_acc=float(acc_va),val_f1=float(f1_va),
                         test_acc=float(acc_te),test_f1=float(f1_te)))
    df=pd.DataFrame(rows)
    best=df.loc[df['val_f1'].idxmax()].to_dict()
    best_test=df.loc[df['test_acc'].idxmax()].to_dict()
    return df, best, best_test


def run_clsmin_dba(X_raw, y, idx_tr, idx_va, idx_te, seed_train: int, wR=0.15, step=0.05, n_iter=5):
    t0=time.time()
    Xd_raw=DBA.build_dtw_clsmin_dba(X_raw,y,idx_tr,wR=wR,step=step,n_iter=n_iter)
    build_s=time.time()-t0
    mu,sd=M.compute_norm_stats(Xd_raw, idx_tr)
    Xd=M.normalize(Xd_raw, mu, sd)
    M.set_seed(seed_train)
    acc_va,f1_va,acc_te,f1_te=M.train_model(Xd,y,idx_tr,idx_va,idx_te,in_ch=4, seed=seed_train)
    return dict(wR=wR,step=step,n_iter=n_iter,build_s=float(build_s),
                val_acc=float(acc_va),val_f1=float(f1_va),
                test_acc=float(acc_te),test_f1=float(f1_te))


def sweep_multi(X_raw, y, idx_tr, idx_va, idx_te, seed_train: int, legacy_rng: bool):
    configs=[
        (3,0.10,0.0),
        (3,0.15,0.0),
        (3,0.20,0.0),
        (3,0.15,0.05),
        (4,0.15,0.0),
        (4,0.15,0.05),
        (5,0.15,0.0),
    ]
    rows=[]
    if legacy_rng:
        M.set_seed(seed_train)
    for tpc,wR,step in configs:
        t0=time.time()
        Xm_raw=M.build_dtw_multi(X_raw,y,idx_tr,templates_per_class=tpc,wR=wR,step=step,seed=seed_train)
        build_s=time.time()-t0
        mu,sd=M.compute_norm_stats(Xm_raw, idx_tr)
        Xm=M.normalize(Xm_raw, mu, sd)
        if not legacy_rng:
            M.set_seed(seed_train)
        acc_va,f1_va,acc_te,f1_te=M.train_model(Xm,y,idx_tr,idx_va,idx_te,in_ch=Xm.shape[1], seed=seed_train)
        rows.append(dict(tpc=tpc,wR=wR,step=step,in_ch=int(Xm.shape[1]),build_s=float(build_s),
                         val_acc=float(acc_va),val_f1=float(f1_va),
                         test_acc=float(acc_te),test_f1=float(f1_te)))
    df=pd.DataFrame(rows)
    best=df.loc[df['val_f1'].idxmax()].to_dict()
    return df, best


def replication_control(X_raw, y, idx_tr, idx_va, idx_te, seed_train: int, tpc=4):
    T=3*tpc
    Xrep_raw=np.tile(X_raw,(1,T,1))
    mu,sd=M.compute_norm_stats(Xrep_raw, idx_tr)
    Xrep=M.normalize(Xrep_raw, mu, sd)
    M.set_seed(seed_train)
    acc_va,f1_va,acc_te,f1_te=M.train_model(Xrep,y,idx_tr,idx_va,idx_te,in_ch=Xrep.shape[1], seed=seed_train)
    return dict(tpc=tpc,in_ch=int(Xrep.shape[1]),val_acc=float(acc_va),val_f1=float(f1_va),test_acc=float(acc_te),test_f1=float(f1_te))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mat', required=True)
    ap.add_argument('--L', type=int, default=176)
    ap.add_argument('--seed', type=int, default=42, help='used for split + training')
    ap.add_argument('--out_dir', default='.')
    ap.add_argument('--legacy_rng', type=int, default=1, help='1=reproduce original sweep behavior (seed set once, no reseed per config)')
    args=ap.parse_args()

    out_dir=Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz_list,y=load_xyz_y(args.mat)
    idx_tr, idx_va, idx_te = M.stratified_split(y, seed=args.seed)

    X_raw=M.build_baseline_data(xyz_list, args.L)

    # baseline CNN
    base=run_cnn(X_raw,y,idx_tr,idx_va,idx_te,in_ch=4,seed_train=args.seed)

    # clsmin mean sweep
    df_cls, best_cls, best_cls_test = sweep_clsmin_mean(X_raw,y,idx_tr,idx_va,idx_te,seed_train=args.seed, legacy_rng=bool(args.legacy_rng))

    # clsmin DBA
    dba = run_clsmin_dba(X_raw,y,idx_tr,idx_va,idx_te,seed_train=args.seed,wR=0.15,step=0.05,n_iter=5)

    # multi sweep
    df_mul, best_mul = sweep_multi(X_raw,y,idx_tr,idx_va,idx_te,seed_train=args.seed, legacy_rng=bool(args.legacy_rng))

    # replication control uses best tpc (or 4)
    tpc_best=int(best_mul['tpc']) if 'tpc' in best_mul else 4
    rep = replication_control(X_raw,y,idx_tr,idx_va,idx_te,seed_train=args.seed, tpc=tpc_best)

    # assemble summary
    summary=[
        dict(method='Baseline CNN (4ch)', config=f'L={args.L} seed={args.seed}', **base),
        dict(method='DTW-ClsMin-Mean + CNN', config='best by val_f1', **best_cls),
        dict(method='DTW-ClsMin-Mean + CNN', config='best by test_acc (ref)', **best_cls_test),
        dict(method='DTW-ClsMin-DBA + CNN', config='DBA single-template', **dba),
        dict(method='DTW-MultiTemplate + CNN', config='best by val_f1', **best_mul),
        dict(method='Replication control + CNN', config='tile channels, no DTW', **rep),
    ]

    df_sum=pd.DataFrame(summary)

    # save
    tag=f'seed{args.seed}_L{args.L}'
    df_sum.to_csv(out_dir/f'ch3_closed_loop_plus_{tag}.csv', index=False)
    with open(out_dir/f'ch3_closed_loop_plus_{tag}.json','w') as f:
        json.dump(summary,f,ensure_ascii=False,indent=2)

    df_cls.to_csv(out_dir/f'ch3_clsmin_sweep_{tag}.csv', index=False)
    df_mul.to_csv(out_dir/f'ch3_multi_sweep_{tag}.csv', index=False)

    print(df_sum.to_string(index=False))
    print('\nSaved to', out_dir)


if __name__=='__main__':
    main()
