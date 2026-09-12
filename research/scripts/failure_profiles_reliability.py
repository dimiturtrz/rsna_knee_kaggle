"""Split-half reliability of the centered failure profiles: are the 302
member-specific profiles real signal or per-column AUC sampling noise?

Frame identical to failure_profiles.py (302 honest members @holdout909,
midpoint-binarised v2 truth). Split: seeded permutation of the 909 studies
(seed base 2026, 25 splits; stated, not index parity). For each split each
member's profile is centered by the pool median recomputed WITHIN that half,
then the member's own half-A vs half-B profiles are Spearman-correlated.

Run: CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe research/scripts/failure_profiles_reliability.py
"""

import glob
import os

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = "D:/models/rsna_knee_abnormality"
REF = os.path.join(ROOT, "band_span_s2026.holdout.npz")
VOID = {
    "raptor_v4_swa", "raptor_v4", "raptor_v9", "raptor_effv2l480",
    "raptor_cnv2b336", "raptor_coat384", "dinov3f0_solo",
    "foreign_dinov3_matched", "foreign_dinov3_band1189",
    "foreign_dinov3_band1387", "foreign_dinov3_band1486",
    "foreign_dinov3_band1090", "foreign_dinov3_tta4", "foreign_dinov3_tta3",
    "foreign_dinov3_band1684", "foreign_dinov3_band0892",
    "foreign_dinov3_band208",
}
N_SPLITS = 25
SEED_BASE = 2026


def per_column_auc(truth: np.ndarray, preds: np.ndarray, idx: np.ndarray) -> np.ndarray | None:
    out = np.empty(12)
    for j in range(12):
        y = truth[idx, j]
        if len(np.unique(y)) < 2:
            return None
        out[j] = roc_auc_score(y, preds[idx, j])
    return out


def main() -> None:
    ref = np.load(REF)
    ref_studies = [str(s) for s in ref["studies"].tolist()]
    pos = {s: i for i, s in enumerate(ref_studies)}
    t0 = (np.asarray(ref["truth"], float) >= 0.5).astype(int)

    names, preds_all = [], []
    n_scored = 0
    for p in sorted(glob.glob(os.path.join(ROOT, "*.holdout.npz"))):
        name = os.path.basename(p)[:-len(".holdout.npz")]
        z = np.load(p)
        st = [str(s) for s in z["studies"].tolist()]
        if set(st) != set(ref_studies) or len(st) != 909:
            continue
        order = np.array([pos[s] for s in st])
        pr = np.empty((909, 12)); pr[order] = z["predictions"]
        tr = np.empty((909, 12)); tr[order] = z["truth"]
        if not np.allclose(tr, np.asarray(ref["truth"], float)):
            continue
        n_scored += 1
        if name in VOID:
            continue
        names.append(name)
        preds_all.append(pr)

    preds = np.asarray(preds_all)  # (m, 909, 12)
    m = preds.shape[0]
    print(f"CUT: 302-frame — scored {n_scored} minus 17 VOID -> honest n={m}; "
          f"per-column AUC, midpoint-binarised v2 truth")
    print(f"SPLIT: seeded permutation of the 909 studies (numpy default_rng, "
          f"seed {SEED_BASE}+split), halves 455/454, {N_SPLITS} splits; "
          f"centering = pool median recomputed within each half")

    within = []          # all member x split correlations
    split_medians = []
    skipped = 0
    for s in range(N_SPLITS):
        rng = np.random.default_rng(SEED_BASE + s)
        perm = rng.permutation(909)
        half_a, half_b = perm[:455], perm[455:]
        auc_a = np.empty((m, 12)); auc_b = np.empty((m, 12))
        ok = True
        for i in range(m):
            a = per_column_auc(t0, preds[i], half_a)
            b = per_column_auc(t0, preds[i], half_b)
            if a is None or b is None:
                ok = False
                break
            auc_a[i], auc_b[i] = a, b
        if not ok:
            skipped += 1
            continue
        ca = auc_a - np.median(auc_a, axis=0)
        cb = auc_b - np.median(auc_b, axis=0)
        rs = np.array([spearmanr(ca[i], cb[i]).statistic for i in range(m)])
        within.append(rs)
        split_medians.append(float(np.median(rs)))

    within = np.asarray(within)          # (splits, m)
    flat = within.ravel()
    member_med = np.median(within, axis=0)   # per-member median across splits
    print(f"\n=== within-member split-half Spearman (own profile A vs B) ===")
    print(f"  splits used: {len(split_medians)} (skipped {skipped}, single-class column)")
    print(f"  pooled {flat.size} member-x-split correlations: "
          f"median {np.median(flat):+.4f} | p10 {np.percentile(flat, 10):+.4f} | "
          f"p90 {np.percentile(flat, 90):+.4f}")
    print(f"  per-split median: mean {np.mean(split_medians):+.4f} "
          f"sd {np.std(split_medians):.4f} "
          f"[{min(split_medians):+.4f}, {max(split_medians):+.4f}]")
    print(f"  per-member median-across-splits distribution: "
          f"median {np.median(member_med):+.4f} | "
          f"p10 {np.percentile(member_med, 10):+.4f} | "
          f"p90 {np.percentile(member_med, 90):+.4f} | "
          f"members with median r < 0: {(member_med < 0).sum()}/{m}")

    rel = float(np.mean(split_medians))
    print(f"\n=== attenuation correction ===")
    print(f"  mean split-half reliability (per-split medians): {rel:+.4f}")
    print(f"  observed between-member mean pairwise Spearman (full data): +0.0144")
    if rel > 0:
        print(f"  noise-adjusted between-member agreement: {0.0144 / rel:+.4f}")
    print(f"  Spearman-Brown estimate of full-length profile reliability: "
          f"{2 * rel / (1 + rel):+.4f}")


if __name__ == "__main__":
    main()
