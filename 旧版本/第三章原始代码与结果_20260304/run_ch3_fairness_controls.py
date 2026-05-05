import os, json, argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import run_dtw_multi_sweep as M


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


def count_params(model: nn.Module) -> int:
    return sum(int(p.numel()) for p in model.parameters())


def train_one_custom(model: nn.Module, X, y, idx_tr, idx_va, idx_te, seed_train=42, max_epochs=60, patience=10):
    torch.set_num_threads(1)
    M.set_seed(seed_train)

    ds_tr=M.SeqDataset(X[idx_tr], y[idx_tr])
    ds_va=M.SeqDataset(X[idx_va], y[idx_va])
    ds_te=M.SeqDataset(X[idx_te], y[idx_te])
    dl_tr=DataLoader(ds_tr, batch_size=64, shuffle=True)
    dl_va=DataLoader(ds_va, batch_size=256, shuffle=False)
    dl_te=DataLoader(ds_te, batch_size=256, shuffle=False)

    opt=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit=nn.CrossEntropyLoss()

    best=-1
    best_state=None
    bad=0

    def eval_model_local(m, loader):
        m.eval()
        ys=[]; ps=[]
        with torch.no_grad():
            for xb,yb in loader:
                logits=m(xb)
                pred=torch.argmax(logits, dim=1).cpu().numpy()
                ys.append(yb.numpy()); ps.append(pred)
        yt=np.concatenate(ys); yp=np.concatenate(ps)
        acc=float(np.mean(yt==yp))
        f1=M.macro_f1(yt, yp, 3)
        return acc,f1

    for ep in range(1, max_epochs+1):
        model.train()
        for xb,yb in dl_tr:
            opt.zero_grad(set_to_none=True)
            loss=crit(model(xb), yb)
            loss.backward()
            opt.step()
        if ep%2==0 or ep==1:
            acc_va,f1_va=eval_model_local(model, dl_va)
            if f1_va>best+1e-6:
                best=f1_va
                best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
                bad=0
            else:
                bad+=1
                if bad>=patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    acc_va,f1_va=eval_model_local(model, dl_va)
    acc_te,f1_te=eval_model_local(model, dl_te)
    return float(acc_va), float(f1_va), float(acc_te), float(f1_te)


class CompressCNN(nn.Module):
    def __init__(self, in_ch, compress_ch=8, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, compress_ch, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(compress_ch, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )
    def forward(self,x):
        return self.net(x)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mat', required=True)
    ap.add_argument('--L', type=int, default=176)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tpc', type=int, default=4)
    ap.add_argument('--wR', type=float, default=0.15)
    ap.add_argument('--step', type=float, default=0.05)
    ap.add_argument('--compress_ch', type=int, default=8)
    ap.add_argument('--out_dir', default='.')
    args=ap.parse_args()

    out_dir=Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz_list,y=load_xyz_y(args.mat)
    idx_tr, idx_va, idx_te = M.stratified_split(y, seed=args.seed)
    X_raw=M.build_baseline_data(xyz_list, args.L)

    rows=[]

    # baseline params
    base_model=M.SimpleCNN(in_ch=4)
    rows.append(dict(name='BaselineCNN', params=count_params(base_model), in_ch=4))

    multi_model=M.SimpleCNN(in_ch=4*3*args.tpc)
    rows.append(dict(name='MultiCNN', params=count_params(multi_model), in_ch=4*3*args.tpc))

    # replication control performance
    T=3*args.tpc
    Xrep_raw=np.tile(X_raw,(1,T,1))
    mu,sd=M.compute_norm_stats(Xrep_raw, idx_tr)
    Xrep=M.normalize(Xrep_raw, mu, sd)
    M.set_seed(args.seed)
    acc_va,f1_va,acc_te,f1_te=M.train_model(Xrep,y,idx_tr,idx_va,idx_te,in_ch=Xrep.shape[1], seed=args.seed)
    rows.append(dict(name='ReplicationControl', params=count_params(M.SimpleCNN(in_ch=Xrep.shape[1])), in_ch=int(Xrep.shape[1]),
                     val_acc=float(acc_va), val_f1=float(f1_va), test_acc=float(acc_te), test_f1=float(f1_te)))

    # multi + 1x1 compress (optional)
    Xm_raw=M.build_dtw_multi(X_raw,y,idx_tr, templates_per_class=args.tpc, wR=args.wR, step=args.step, seed=args.seed)
    mu,sd=M.compute_norm_stats(Xm_raw, idx_tr)
    Xm=M.normalize(Xm_raw, mu, sd)

    model_c=CompressCNN(in_ch=int(Xm.shape[1]), compress_ch=args.compress_ch)
    t0=time.time()
    acc_va,f1_va,acc_te,f1_te=train_one_custom(model_c, Xm, y, idx_tr, idx_va, idx_te, seed_train=args.seed)
    rows.append(dict(name=f'Multi+1x1(compress={args.compress_ch})', params=count_params(model_c), in_ch=int(Xm.shape[1]),
                     val_acc=acc_va, val_f1=f1_va, test_acc=acc_te, test_f1=f1_te, train_s=round(time.time()-t0,3)))

    df=pd.DataFrame(rows)
    tag=f'seed{args.seed}_L{args.L}_tpc{args.tpc}'
    df.to_csv(out_dir/f'ch3_fairness_{tag}.csv', index=False)
    with open(out_dir/f'ch3_fairness_{tag}.json','w') as f:
        json.dump(df.to_dict(orient='records'),f,ensure_ascii=False,indent=2)

    print(df.to_string(index=False))
    print('\nSaved to', out_dir)


if __name__=='__main__':
    main()
