"""Paired bootstrap SE of a 12-column macro-AUC delta at n=909 (M315)."""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "D:/personal_projects/rsna_knee_kaggle")
from core.findings import TARGETS
from kneemri.runs.holdout_record import HoldoutRecord

CUT = 0.5
BASE = Path("D:/data/volumetric/mri/models/rsna_knee_abnormality")
rng = np.random.default_rng(20260911)


def macro_auc(pred: np.ndarray, truth: np.ndarray) -> float:
    vals = []
    for c in range(pred.shape[1]):
        t = truth[:, c]
        ok = ~np.isnan(t)
        y = (t[ok] >= CUT).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        vals.append(roc_auc_score(y, pred[ok, c]))
    return float(np.mean(vals))


def paired(name_a: str, name_b: str, n_boot: int = 2000) -> None:
    ra = HoldoutRecord(BASE / f"{name_a}.holdout.npz").read()
    rb = HoldoutRecord(BASE / f"{name_b}.holdout.npz").read()
    pa, pb, truth = ra.aligned(rb)
    n = len(truth)
    t0 = time.time()
    idx = rng.integers(0, n, size=(n_boot, n))
    da = np.empty(n_boot)
    db = np.empty(n_boot)
    for i in range(n_boot):
        ii = idx[i]
        da[i] = macro_auc(pa[ii], truth[ii])
        db[i] = macro_auc(pb[ii], truth[ii])
    dd = da - db
    point = macro_auc(pa, truth) - macro_auc(pb, truth)
    se_paired = float(dd.std())
    se_unpaired = float(np.sqrt(da.var() + db.var()))
    lo, hi = np.percentile(dd, [2.5, 97.5])
    p_flip = float(np.mean(np.sign(dd) != np.sign(point)))
    print(f"{name_a} vs {name_b}: n={n} point={point:+.4f}")
    print(f"  paired SE={se_paired:.4f}  2sd={2*se_paired:.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  P(sign flip)={p_flip:.3f}")
    print(f"  unpaired SE={se_unpaired:.4f}  2sd={2*se_unpaired:.4f}  paired shrinks {se_unpaired/se_paired:.1f}x")
    print(f"  ({time.time()-t0:.0f}s, {n_boot} resamples)")


pairs = [
    ("band_dense_s2026", "focal_dense_s2026"),
    ("band_dense_s2026", "attend_dense_s2026"),
    ("band_dense_s2026", "calib_dense_hold_s2026"),
    ("resnet_224_short_s2026", "resnet_224_long_s2026"),
    ("band_span_s2026", "calib_span_s2026"),
]
for a, b in pairs:
    paired(a, b)
