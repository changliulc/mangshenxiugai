import os, random, time, zlib
from pathlib import Path
import numpy as np

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ.setdefault('NUMEXPR_NUM_THREADS','1')

# patch coverage for numba (same as your scripts)
import coverage, types as _types
if not hasattr(coverage, 'types'):
    coverage.types = _types.SimpleNamespace()
for _name in ['Tracer','TTraceData','TShouldTraceFn','TFileDisposition','TShouldStartContextFn','TWarnFn','TTraceFn']:
    if not hasattr(coverage.types, _name):
        setattr(coverage.types, _name, object)

from numba import njit

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------- seed ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ---------------- MAT reader (same as your code) ----------------
MAT_DTYPES=set([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
MX_CELL_CLASS=1
DTYPE_MAP={1:np.int8,2:np.uint8,3:np.int16,4:np.uint16,5:np.int32,6:np.uint32,7:np.float32,9:np.float64,12:np.int64,13:np.uint64}

def read_tag(buf, off):
    a=int.from_bytes(buf[off:off+4],'little')
    b=int.from_bytes(buf[off+4:off+8],'little')
    off2=off+8
    dt=a; nb=b
    if dt not in MAT_DTYPES:
        dt=a & 0xFFFF
        nb=a>>16
        dinl=buf[off+4:off+8]
        return dt,nb,True,dinl,off2
    return dt,nb,False,None,off2

def skip_pad(off, nb):
    return off + ((8-(nb%8))%8)

def read_data(buf, off, nb, small, dinl):
    if small:
        return dinl[:nb], off
    data=buf[off:off+nb]
    off=skip_pad(off+nb, nb)
    return data, off

def parse_mimatrix(payload, off=0):
    dt,nb,small,dinl,off = read_tag(payload, off)
    d,off = read_data(payload, off, nb, small, dinl)
    flags=int.from_bytes(d[:4],'little')
    mx_class=flags & 0xFF
    is_logical=bool(flags & (1<<10))
    dt,nb,small,dinl,off = read_tag(payload, off)
    d,off = read_data(payload, off, nb, small, dinl)
    dims=tuple(int.from_bytes(d[i:i+4],'little', signed=True) for i in range(0,nb,4))
    dt,nb,small,dinl,off = read_tag(payload, off)
    d,off = read_data(payload, off, nb, small, dinl)
    name=bytes(d).decode('utf-8','ignore')
    if mx_class==MX_CELL_CLASS:
        numel=int(np.prod(dims)) if len(dims)>0 else 0
        cells=[]
        for _ in range(numel):
            dt2,nb2,small2,dinl2,off = read_tag(payload, off)
            sub,off = read_data(payload, off, nb2, small2, dinl2)
            if dt2==15:
                sub=zlib.decompress(sub)
                dt3,nb3,small3,dinl3,off3 = read_tag(sub,0)
                p3,_ = read_data(sub, off3, nb3, small3, dinl3)
                nm,_,_,val=parse_mimatrix(p3,0)
                cells.append(val)
            else:
                nm,_,_,val=parse_mimatrix(sub,0)
                cells.append(val)
        return name,mx_class,dims,cells
    dt2,nb2,small2,dinl2,off = read_tag(payload, off)
    d,off = read_data(payload, off, nb2, small2, dinl2)
    arr=np.frombuffer(d, dtype=DTYPE_MAP[dt2])
    val=arr.reshape(dims, order='F') if len(dims)>0 else arr
    if is_logical:
        val=val.astype(bool)
    return name,mx_class,dims,val

def load_vars(path, wanted):
    buf=Path(path).read_bytes()
    out={}
    off=128
    while off+8<=len(buf) and len(out)<len(wanted):
        dt,nb,small,dinl,off = read_tag(buf, off)
        payload,off = read_data(buf, off, nb, small, dinl)
        if dt==15:
            payload=zlib.decompress(payload)
            sub_off=0
            while sub_off+8<=len(payload):
                dt2,nb2,small2,dinl2,sub_off2 = read_tag(payload, sub_off)
                sub,sub_off = read_data(payload, sub_off2, nb2, small2, dinl2)
                if dt2!=14:
                    continue
                name,_,_,val=parse_mimatrix(sub,0)
                if name in wanted:
                    out[name]=val
        elif dt==14:
            name,_,_,val=parse_mimatrix(payload,0)
            if name in wanted:
                out[name]=val
    return out

# ---------------- DTW (same as your code) ----------------
@njit
def dtw_distance_1d(x, y, w, step_penalty):
    N=x.shape[0]
    M=y.shape[0]
    if w < abs(N-M):
        w = abs(N-M)
    inf=1e30
    D=np.full((N+1,M+1), inf, np.float32)
    D[0,0]=0.0
    for i in range(1,N+1):
        j0=1
        if i-w>j0:
            j0=i-w
        j1=M
        if i+w<j1:
            j1=i+w
        xi=x[i-1]
        for j in range(j0,j1+1):
            dist=(xi-y[j-1])*(xi-y[j-1])
            d0=D[i-1,j-1]
            d1=D[i-1,j] + step_penalty
            d2=D[i,j-1] + step_penalty
            if d0<=d1 and d0<=d2:
                D[i,j]=d0+dist
            elif d1<=d2:
                D[i,j]=d1+dist
            else:
                D[i,j]=d2+dist
    return float(D[N,M])

@njit
def dtw_warp_mv(X, Y, w, step_penalty):
    N=X.shape[0]
    M=Y.shape[0]
    Dch=X.shape[1]
    if w < abs(N-M):
        w = abs(N-M)
    inf=1e30
    cost=np.full((N+1,M+1), inf, np.float32)
    ptr=np.full((N+1,M+1), -1, np.int8)
    cost[0,0]=0.0
    for i in range(1,N+1):
        j0=1
        if i-w>j0:
            j0=i-w
        j1=M
        if i+w<j1:
            j1=i+w
        for j in range(j0,j1+1):
            # distance on magnitude channel only (last channel)
            k = Dch - 1
            diff = X[i-1,k]-Y[j-1,k]
            dist = diff*diff
            d0=cost[i-1,j-1]
            d1=cost[i-1,j] + step_penalty
            d2=cost[i,j-1] + step_penalty
            if d0<=d1 and d0<=d2:
                cost[i,j]=d0+dist
                ptr[i,j]=0
            elif d1<=d2:
                cost[i,j]=d1+dist
                ptr[i,j]=1
            else:
                cost[i,j]=d2+dist
                ptr[i,j]=2

    out=np.zeros((M,Dch), np.float32)
    cnt=np.zeros((M,), np.int32)
    i=N
    j=M
    while i>0 and j>0:
        jj=j-1
        ii=i-1
        for k in range(Dch):
            out[jj,k] += X[ii,k]
        cnt[jj] += 1
        p=ptr[i,j]
        if p==0:
            i-=1; j-=1
        elif p==1:
            i-=1
        elif p==2:
            j-=1
        else:
            break

    last=-1
    for jj in range(M):
        if cnt[jj] > 0:
            inv = 1.0 / cnt[jj]
            for k in range(Dch):
                out[jj,k] *= inv
            last = jj
        elif last >= 0:
            for k in range(Dch):
                out[jj,k] = out[last,k]

    if cnt[0] == 0:
        first=-1
        for jj in range(M):
            if cnt[jj] > 0:
                first=jj
                break
        if first >= 0:
            for jj in range(first):
                for k in range(Dch):
                    out[jj,k] = out[first,k]
    return out

# ---------------- data utils ----------------
def resample_linear(x, L):
    T,D=x.shape
    if T==L:
        return x.astype(np.float32, copy=True)
    t0=np.linspace(0,1,T)
    t1=np.linspace(0,1,L)
    y=np.empty((L,D), np.float32)
    for d in range(D):
        y[:,d]=np.interp(t1,t0,x[:,d]).astype(np.float32)
    return y

def make_4ch_seq(xyz, L):
    xr=resample_linear(xyz, L)
    m=np.sqrt(np.sum(xr*xr, axis=1, dtype=np.float32))
    out=np.empty((L,4), np.float32)
    out[:,0:3]=xr
    out[:,3]=m
    return out

def build_baseline_data(xyz_list, L):
    N=len(xyz_list)
    X=np.zeros((N,4,L), np.float32)
    for i,xyz in enumerate(xyz_list):
        seq=make_4ch_seq(xyz,L)
        X[i]=seq.T
    return X

def stratified_split(y, seed=42, train_ratio=0.70, val_ratio=0.15):
    rng=np.random.default_rng(seed)
    y=np.asarray(y)
    idx_tr=[]; idx_va=[]; idx_te=[]
    for c in np.unique(y):
        idx=np.where(y==c)[0]
        rng.shuffle(idx)
        n=len(idx)
        n_tr=int(round(train_ratio*n))
        n_va=int(round(val_ratio*n))
        idx_tr.extend(idx[:n_tr])
        idx_va.extend(idx[n_tr:n_tr+n_va])
        idx_te.extend(idx[n_tr+n_va:])
    return np.array(idx_tr), np.array(idx_va), np.array(idx_te)

def compute_norm_stats(X, idx_tr):
    mu = X[idx_tr].mean(axis=(0,2), keepdims=True)
    sd = X[idx_tr].std(axis=(0,2), keepdims=True)
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)

def normalize(X, mu, sd):
    return (X - mu) / sd

def macro_f1(y_true, y_pred, C=3):
    f1=[]
    for c in range(C):
        tp=np.sum((y_true==c)&(y_pred==c))
        fp=np.sum((y_true!=c)&(y_pred==c))
        fn=np.sum((y_true==c)&(y_pred!=c))
        p=tp/(tp+fp) if (tp+fp)>0 else 0
        r=tp/(tp+fn) if (tp+fn)>0 else 0
        f=2*p*r/(p+r) if (p+r)>0 else 0
        f1.append(f)
    return float(np.mean(f1))

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X=torch.from_numpy(X)
        self.y=torch.from_numpy(y.astype(np.int64))
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleCNN(nn.Module):
    def __init__(self, in_ch, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3),
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
    def forward(self, x):
        return self.net(x)

def eval_model(model, loader, device='cpu'):
    model.eval()
    ys=[]; ps=[]
    with torch.no_grad():
        for xb,yb in loader:
            xb=xb.to(device)
            logits=model(xb)
            pred=torch.argmax(logits, dim=1).cpu().numpy()
            ys.append(yb.numpy())
            ps.append(pred)
    y=np.concatenate(ys)
    p=np.concatenate(ps)
    acc=float(np.mean(y==p))
    f1=macro_f1(y,p,3)
    return acc,f1

def train_model(X, y, idx_tr, idx_va, idx_te, in_ch, max_epochs=60, patience=10):
    torch.set_num_threads(1)
    device='cpu'
    ds_tr=SeqDataset(X[idx_tr], y[idx_tr])
    ds_va=SeqDataset(X[idx_va], y[idx_va])
    ds_te=SeqDataset(X[idx_te], y[idx_te])
    dl_tr=DataLoader(ds_tr, batch_size=64, shuffle=True)
    dl_va=DataLoader(ds_va, batch_size=256, shuffle=False)
    dl_te=DataLoader(ds_te, batch_size=256, shuffle=False)
    model=SimpleCNN(in_ch=in_ch).to(device)
    opt=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit=nn.CrossEntropyLoss()
    best_va=-1
    best_state=None
    bad=0
    for ep in range(1, max_epochs+1):
        model.train()
        for xb,yb in dl_tr:
            xb=xb.to(device)
            yb=yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits=model(xb)
            loss=crit(logits, yb)
            loss.backward()
            opt.step()
        if ep % 2 == 0 or ep==1:
            acc_va,f1_va=eval_model(model, dl_va, device)
            if f1_va > best_va + 1e-6:
                best_va=f1_va
                best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
                bad=0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    acc_va,f1_va=eval_model(model, dl_va, device)
    acc_te,f1_te=eval_model(model, dl_te, device)
    return acc_va,f1_va,acc_te,f1_te

# ---------------- DBA template (key change) ----------------
def dba_template_class(X_raw, ids, wR=0.15, step=0.05, n_iter=5):
    ids=np.asarray(ids, dtype=np.int64)
    L=X_raw.shape[2]
    w=max(1,int(round(wR*L)))
    tpl=X_raw[ids].mean(axis=0).astype(np.float32)  # init
    _=dtw_warp_mv(X_raw[ids[0]].T, tpl.T, w, step)  # warmup
    for _ in range(n_iter):
        acc=np.zeros_like(tpl)
        for idx in ids:
            warped=dtw_warp_mv(X_raw[idx].T, tpl.T, w, step)  # (L,4)
            acc += warped.T
        tpl=(acc/len(ids)).astype(np.float32)
    return tpl

def build_dtw_clsmin_dba(X_raw, y, idx_tr, wR=0.15, step=0.05, n_iter=5):
    C=int(y.max())+1
    L=X_raw.shape[2]
    w=max(1,int(round(wR*L)))
    templates=np.zeros((C,4,L), np.float32)
    for c in range(C):
        ids = idx_tr[y[idx_tr]==c]
        templates[c] = dba_template_class(X_raw, ids, wR=wR, step=step, n_iter=n_iter)

    # warmup
    _=dtw_distance_1d(X_raw[0,3], templates[0,3], w, 0.0)
    _=dtw_warp_mv(X_raw[0].T, templates[0].T, w, step)

    N=X_raw.shape[0]
    X_al=np.zeros_like(X_raw)
    for i in range(N):
        mi=X_raw[i,3]
        bestc=0
        bestd=1e30
        for c in range(C):
            d=dtw_distance_1d(mi, templates[c,3], w, 0.0)
            if d < bestd:
                bestd=d
                bestc=c
        aligned=dtw_warp_mv(X_raw[i].T, templates[bestc].T, w, step)
        X_al[i]=aligned.T
    return X_al

def main():
    set_seed(42)
    mat_path='/mnt/data/ac949708-72b7-4e9b-be27-bb6c78f1f5b6.mat'  # 修改为你的数据路径
    L=176

    vars=load_vars(mat_path, {'ProcessedData','targetLength'})
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

    idx_tr, idx_va, idx_te = stratified_split(y, seed=42)

    X_raw=build_baseline_data(xyz_list, L)

    # baseline CNN
    mu,sd=compute_norm_stats(X_raw, idx_tr)
    X=normalize(X_raw,mu,sd)
    set_seed(42)
    acc_va,f1_va,acc_te,f1_te=train_model(X,y,idx_tr,idx_va,idx_te,in_ch=4)
    print('BASE', 'val', round(acc_va,4), round(f1_va,4), 'test', round(acc_te,4), round(f1_te,4), flush=True)

    # ClsMin mean (original) for reference
    # NOTE: original code uses mean template. Here we mimic it quickly:
    def build_clsmin_mean(X_raw, y, idx_tr, wR=0.10, step=0.05):
        C=int(y.max())+1
        L=X_raw.shape[2]
        w=max(1,int(round(wR*L)))
        templates=np.zeros((C,4,L), np.float32)
        for c in range(C):
            ids=idx_tr[y[idx_tr]==c]
            templates[c]=X_raw[ids].mean(axis=0)
        _=dtw_distance_1d(X_raw[0,3], templates[0,3], w, 0.0)
        _=dtw_warp_mv(X_raw[0].T, templates[0].T, w, step)
        N=X_raw.shape[0]
        X_al=np.zeros_like(X_raw)
        for i in range(N):
            mi=X_raw[i,3]
            bestc=0; bestd=1e30
            for c in range(C):
                d=dtw_distance_1d(mi, templates[c,3], w, 0.0)
                if d < bestd:
                    bestd=d; bestc=c
            aligned=dtw_warp_mv(X_raw[i].T, templates[bestc].T, w, step)
            X_al[i]=aligned.T
        return X_al

    X_mean=build_clsmin_mean(X_raw,y,idx_tr,wR=0.10,step=0.05)
    mu2,sd2=compute_norm_stats(X_mean, idx_tr)
    X_mean=normalize(X_mean, mu2, sd2)
    set_seed(42)
    acc_va,f1_va,acc_te,f1_te=train_model(X_mean,y,idx_tr,idx_va,idx_te,in_ch=4)
    print('ClsMin-mean', 'val', round(acc_va,4), round(f1_va,4), 'test', round(acc_te,4), round(f1_te,4), flush=True)

    # ClsMin-DBA (proposed simple improvement)
    X_dba=build_dtw_clsmin_dba(X_raw,y,idx_tr,wR=0.15,step=0.05,n_iter=5)
    mu3,sd3=compute_norm_stats(X_dba, idx_tr)
    X_dba=normalize(X_dba, mu3, sd3)
    set_seed(42)
    acc_va,f1_va,acc_te,f1_te=train_model(X_dba,y,idx_tr,idx_va,idx_te,in_ch=4)
    print('ClsMin-DBA', 'val', round(acc_va,4), round(f1_va,4), 'test', round(acc_te,4), round(f1_te,4), flush=True)

if __name__=='__main__':
    main()
