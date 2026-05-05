import struct, zlib, time
from pathlib import Path
import numpy as np

# ---- patch coverage for numba import ----
import coverage, types as _types
if not hasattr(coverage, 'types'):
    coverage.types = _types.SimpleNamespace()
for _name in ['Tracer','TTraceData','TShouldTraceFn','TFileDisposition','TShouldStartContextFn','TWarnFn','TTraceFn']:
    if not hasattr(coverage.types, _name):
        setattr(coverage.types, _name, object)

from numba import njit

# ---------- MAT reader ----------
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
    # flags
    dt,nb,small,dinl,off = read_tag(payload, off)
    d,off = read_data(payload, off, nb, small, dinl)
    flags=int.from_bytes(d[:4],'little')
    mx_class=flags & 0xFF
    is_logical=bool(flags & (1<<10))
    # dims
    dt,nb,small,dinl,off = read_tag(payload, off)
    d,off = read_data(payload, off, nb, small, dinl)
    dims=tuple(int.from_bytes(d[i:i+4],'little', signed=True) for i in range(0,nb,4))
    # name
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
    # numeric
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

# ---------- preprocessing ----------

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

def mag_xyz(xyz):
    return np.sqrt(np.sum(xyz*xyz, axis=1, dtype=np.float32))

# ---------- split ----------

def split_by_source(y,source,seed,val_ratio=0.15,test_ratio=0.15):
    rng=np.random.default_rng(seed)
    N=len(y)
    tr=np.zeros(N,bool); va=np.zeros(N,bool); te=np.zeros(N,bool)
    for c in range(int(y.max())+1):
        idx=np.where(y==c)[0]
        uniq=np.unique(source[idx])
        rng.shuffle(uniq)
        n=len(uniq)
        nte=max(1,int(round(test_ratio*n)))
        nva=max(1,int(round(val_ratio*n)))
        te_src=set(uniq[:nte]); va_src=set(uniq[nte:nte+nva])
        for i in idx:
            s=source[i]
            if s in te_src: te[i]=True
            elif s in va_src: va[i]=True
            else: tr[i]=True
    return tr,va,te

# ---------- DTW numba ----------
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
            dist=0.0
            for k in range(Dch):
                diff = X[i-1,k]-Y[j-1,k]
                dist += diff*diff
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

# ---------- classifier ----------

def softmax(z):
    z=z-np.max(z,axis=1,keepdims=True)
    ez=np.exp(z)
    return ez/np.sum(ez,axis=1,keepdims=True)

def macro_f1(y_true,y_pred,C):
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

def train_softmax(Xtr,ytr,Xva,yva,lam=1e-3,lr=0.3,max_iter=200,patience=20,seed=0):
    rng=np.random.default_rng(seed)
    N,D=Xtr.shape
    C=int(np.max(ytr))+1
    W=(rng.standard_normal((D,C)).astype(np.float32))*0.01
    b=np.zeros((C,),np.float32)
    Y=np.eye(C,dtype=np.float32)[ytr]
    best=-1
    best_state=None
    bad=0
    for it in range(max_iter):
        P=softmax(Xtr@W+b)
        G=(P-Y)/N
        W-=lr*(Xtr.T@G + lam*W)
        b-=lr*np.sum(G,axis=0)
        if it%5==0:
            Pv=softmax(Xva@W+b)
            ypv=np.argmax(Pv,axis=1)
            f1=macro_f1(yva,ypv,C)
            if f1>best+1e-6:
                best=f1
                best_state=(W.copy(),b.copy(),it,f1)
                bad=0
            else:
                bad+=1
                if bad>=patience:
                    break
    return best_state

def eval_model(X,y,W,b):
    P=softmax(X@W+b)
    yp=np.argmax(P,axis=1)
    return float(np.mean(yp==y)), macro_f1(y,yp,int(np.max(y))+1)

def standardize(X, tr_mask):
    mu=X[tr_mask].mean(axis=0)
    sd=X[tr_mask].std(axis=0)
    sd[sd<1e-6]=1.0
    return (X-mu)/sd

# ---------- experiment ----------

def build_seq_arrays(PD, TL, SI, L0):
    xyz_list=[]; y_list=[]; src_list=[]
    for c in range(3):
        tl=np.array(TL[c]).reshape(-1)
        si=np.array(SI[c]).reshape(-1)
        for i in range(len(PD[c])):
            T=int(tl[i])
            xyz=np.asarray(PD[c][i],dtype=np.float32)[:T,:]
            xyz_list.append(xyz)
            y_list.append(c)
            src_list.append(int(si[i]))
    y=np.asarray(y_list,np.int64)
    source=np.asarray(src_list,np.int64)

    seq4=np.zeros((len(y),L0,4), np.float32)
    mag=np.zeros((len(y),L0), np.float32)
    for i,xyz in enumerate(xyz_list):
        xr=resample_linear(xyz,L0)
        m=mag_xyz(xr)
        seq4[i,:,:3]=xr
        seq4[i,:,3]=m
        mag[i]=m
    return y, source, seq4, mag


def run_clsmin(seq4, mag, y, tr, va, te, wR=0.15, step=0.0, lam=1e-3):
    L0=seq4.shape[1]
    w=max(1,int(round(wR*L0)))

    tmpl4=np.zeros((3,L0,4), np.float32)
    tmpl_mag=np.zeros((3,L0), np.float32)
    for c in range(3):
        mc=(y==c)&tr
        tmpl4[c]=seq4[mc].mean(axis=0)
        tmpl_mag[c]=tmpl4[c,:,3]

    # JIT warmup
    _ = dtw_distance_1d(mag[0], tmpl_mag[0], w, 0.0)
    _ = dtw_warp_mv(seq4[0], tmpl4[0], w, step)

    t0=time.time()
    Xfeat=np.zeros((len(y), 4*L0), np.float32)
    for i in range(len(y)):
        bestc=0
        bestd=1e30
        mi=mag[i]
        d0=dtw_distance_1d(mi, tmpl_mag[0], w, 0.0)
        d1=dtw_distance_1d(mi, tmpl_mag[1], w, 0.0)
        d2=dtw_distance_1d(mi, tmpl_mag[2], w, 0.0)
        if d1<d0:
            bestc=1; bestd=d1
        else:
            bestc=0; bestd=d0
        if d2<bestd:
            bestc=2
        aligned=dtw_warp_mv(seq4[i], tmpl4[bestc], w, step)
        Xfeat[i]=aligned.T.reshape(-1)
    build_s=time.time()-t0

    Xz=standardize(Xfeat,tr)
    Xtr,ytr=Xz[tr],y[tr]
    Xva,yva=Xz[va],y[va]
    Xte,yte=Xz[te],y[te]

    t1=time.time()
    W,b,it,f1v=train_softmax(Xtr,ytr,Xva,yva,lam=lam,lr=0.3,max_iter=250,patience=25,seed=42)
    train_s=time.time()-t1

    acc,f1=eval_model(Xte,yte,W,b)
    return build_s, train_s, f1v, acc, f1


def run_multi(seq4, y, tr, va, te, wR=0.15, step=0.0, lam=1e-3):
    L0=seq4.shape[1]
    w=max(1,int(round(wR*L0)))

    tmpl4=np.zeros((3,L0,4), np.float32)
    for c in range(3):
        mc=(y==c)&tr
        tmpl4[c]=seq4[mc].mean(axis=0)

    _ = dtw_warp_mv(seq4[0], tmpl4[0], w, step)

    t0=time.time()
    Xfeat=np.zeros((len(y), 12*L0), np.float32)
    for i in range(len(y)):
        o0=dtw_warp_mv(seq4[i], tmpl4[0], w, step)
        o1=dtw_warp_mv(seq4[i], tmpl4[1], w, step)
        o2=dtw_warp_mv(seq4[i], tmpl4[2], w, step)
        Xfeat[i]=np.concatenate([o0.T.reshape(-1), o1.T.reshape(-1), o2.T.reshape(-1)], axis=0)
    build_s=time.time()-t0

    Xz=standardize(Xfeat,tr)
    Xtr,ytr=Xz[tr],y[tr]
    Xva,yva=Xz[va],y[va]
    Xte,yte=Xz[te],y[te]

    t1=time.time()
    W,b,it,f1v=train_softmax(Xtr,ytr,Xva,yva,lam=lam,lr=0.25,max_iter=250,patience=25,seed=42)
    train_s=time.time()-t1

    acc,f1=eval_model(Xte,yte,W,b)
    return build_s, train_s, f1v, acc, f1


def main(mat_path):
    vars=load_vars(mat_path, {'ProcessedData','targetLength','sourceIndex'})
    PD=vars['ProcessedData']; TL=vars['targetLength']; SI=vars['sourceIndex']

    L0=96
    t0=time.time()
    y, source, seq4, mag = build_seq_arrays(PD, TL, SI, L0)
    print('loaded+resampled_s', round(time.time()-t0,3), 'N', len(y), 'class', np.bincount(y, minlength=3))

    tr,va,te=split_by_source(y, source, seed=42)
    print('split', tr.sum(), va.sum(), te.sum())

    # baseline: flatten seq4
    Xbase = seq4.reshape(len(y), -1)
    Xz = standardize(Xbase, tr)
    Xtr,ytr=Xz[tr],y[tr]
    Xva,yva=Xz[va],y[va]
    Xte,yte=Xz[te],y[te]
    W,b,it,f1v=train_softmax(Xtr,ytr,Xva,yva,lam=1e-3,lr=0.3,max_iter=200,patience=20,seed=42)
    acc,f1=eval_model(Xte,yte,W,b)
    print('BASE(flat) val_f1', round(f1v,4), 'test_acc', round(acc,4), 'test_f1', round(f1,4))

    build_s, train_s, f1v, acc, f1 = run_clsmin(seq4, mag, y, tr, va, te, wR=0.15, step=0.0, lam=1e-3)
    print('DTW-ClsMin build_s', round(build_s,2), 'train_s', round(train_s,2), 'val_f1', round(f1v,4), 'test_acc', round(acc,4), 'test_f1', round(f1,4))

    build_s, train_s, f1v, acc, f1 = run_multi(seq4, y, tr, va, te, wR=0.15, step=0.0, lam=1e-3)
    print('DTW-Multi build_s', round(build_s,2), 'train_s', round(train_s,2), 'val_f1', round(f1v,4), 'test_acc', round(acc,4), 'test_f1', round(f1,4))


if __name__=='__main__':
    main('/mnt/data/3f82e08c-9373-4134-b811-015b645b3a9e.mat')
