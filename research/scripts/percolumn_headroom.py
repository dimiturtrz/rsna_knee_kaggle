import numpy as np, glob, os, json
from kneemri.evaluation.scores import LabelScores
from sklearn.metrics import roc_auc_score
MODELS = "D:/data/volumetric/mri/models/rsna_knee_abnormality"
files = sorted(glob.glob(os.path.join(MODELS, "**", "*.holdout.npz"), recursive=True))
canon = None
rows = []
for f in files:
    try:
        d = np.load(f, allow_pickle=True)
        if int(d["epoch"]) < 0: continue
        st, tr, pr = d["studies"], (d["truth"] > 0.5).astype(int), d["predictions"]
        if st.shape[0] != 909: continue
        if canon is None: canon = (st, tr)
        if not np.array_equal(st, canon[0]) or not np.array_equal(tr, canon[1]): continue
        aucs = []
        for c in range(tr.shape[1]):
            y = tr[:, c]
            aucs.append(roc_auc_score(y, pr[:, c]) if 0 < y.sum() < len(y) else np.nan)
        rows.append((os.path.basename(f).replace(".holdout.npz",""), np.array(aucs)))
    except Exception: pass
A = np.vstack([r[1] for r in rows])
names = [r[0] for r in rows]
LAB = ["ACL","MCL","MedMen","LatMen","MedOA","LatOA","PFOA","Effus","Synov","Baker","Contus","Fract"]
print(f"POPULATION: {len(rows)} honest solos @holdout909, per-column AUC (cut-free)")
print(f"{'col':8} {'median':>8} {'p90':>8} {'max':>8} {'best member':>34} {'spread':>8}")
for c in range(12):
    col = A[:, c]
    best = names[int(np.nanargmax(col))]
    print(f"{LAB[c]:8} {np.nanmedian(col):8.4f} {np.nanpercentile(col,90):8.4f} {np.nanmax(col):8.4f} {best[:34]:>34} {np.nanmax(col)-np.nanmedian(col):8.4f}")
med = np.nanmedian(A, axis=0); mx = np.nanmax(A, axis=0)
print(f"\nmacro of per-column MEDIANS: {med.mean():.4f}")
print(f"macro of per-column MAXIMA (oracle column-wise pick): {mx.mean():.4f}")
print(f"HEADROOM if each column served by its own best member: +{mx.mean()-med.mean():.4f}")
best_macro = np.nanmax(A.mean(axis=1)); print(f"best SINGLE member macro: {best_macro:.4f}  -> oracle gain over it: +{mx.mean()-best_macro:.4f}")
