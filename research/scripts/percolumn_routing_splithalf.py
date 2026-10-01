import numpy as np, glob, os
from sklearn.metrics import roc_auc_score
MODELS="D:/data/volumetric/mri/models/rsna_knee_abnormality"
canon=None; rows=[]
for f in sorted(glob.glob(os.path.join(MODELS,"**","*.holdout.npz"),recursive=True)):
    try:
        d=np.load(f,allow_pickle=True)
        if int(d["epoch"])<0: continue
        st,tr,pr=d["studies"],(d["truth"]>0.5).astype(int),d["predictions"]
        if st.shape[0]!=909: continue
        if canon is None: canon=(st,tr)
        if not np.array_equal(st,canon[0]) or not np.array_equal(tr,canon[1]): continue
        rows.append((os.path.basename(f).replace(".holdout.npz",""),pr.astype(np.float64)))
    except Exception: pass
names=[r[0] for r in rows]; P=np.stack([r[1] for r in rows]); T=canon[1]
n,S,C=P.shape; print(f"POPULATION {n} honest solos, 909 studies, 12 cols")
def auc(y,s): return roc_auc_score(y,s) if 0<y.sum()<len(y) else np.nan
def macro(idx_sel, idx_eval):
    # idx_sel: studies to SELECT on; idx_eval: studies to SCORE on. per-column argmax member.
    out=[]
    for c in range(C):
        a=np.array([auc(T[idx_sel,c],P[m,idx_sel,c]) for m in range(n)])
        best=int(np.nanargmax(a))
        out.append(auc(T[idx_eval,c],P[best,idx_eval,c]))
    return np.nanmean(out)
def macro_fixed(m,idx): return np.nanmean([auc(T[idx,c],P[m,idx,c]) for c in range(C)])
rng=np.random.default_rng(2026); res=[]
for rep in range(20):
    perm=rng.permutation(S); A,B=perm[:S//2],perm[S//2:]
    # honest routing: select on A, score on B (and vice versa)
    r=0.5*(macro(A,B)+macro(B,A))
    # control: best SINGLE member picked on A, scored on B
    bA=int(np.argmax([macro_fixed(m,A) for m in range(n)])); bB=int(np.argmax([macro_fixed(m,B) for m in range(n)]))
    ctl=0.5*(macro_fixed(bA,B)+macro_fixed(bB,A))
    res.append((r,ctl))
res=np.array(res)
print(f"HONEST per-column routing  : {res[:,0].mean():.4f} +- {res[:,0].std():.4f}")
print(f"HONEST best-single control : {res[:,1].mean():.4f} +- {res[:,1].std():.4f}")
print(f"DELTA routing - single     : {(res[:,0]-res[:,1]).mean():+.4f} +- {(res[:,0]-res[:,1]).std():.4f}  (n=20 half-splits)")
