"""Do honest-pool members share one failure PROFILE across the 12 label columns,
or only one mean? Per-member per-column AUC @holdout909, midpoint cut, v2 ruler;
column-centered profiles; pairwise Spearman agreement with a permutation null;
bottom-3 set census; Kendall's W.

Run: CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe research/scripts/failure_profiles.py
"""

import glob
import os

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = "D:/models/rsna_knee_abnormality"
REF = os.path.join(ROOT, "band_span_s2026.holdout.npz")
COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
VOID = {
    "raptor_v4_swa", "raptor_v4", "raptor_v9", "raptor_effv2l480",
    "raptor_cnv2b336", "raptor_coat384", "dinov3f0_solo",
    "foreign_dinov3_matched", "foreign_dinov3_band1189",
    "foreign_dinov3_band1387", "foreign_dinov3_band1486",
    "foreign_dinov3_band1090", "foreign_dinov3_tta4", "foreign_dinov3_tta3",
    "foreign_dinov3_band1684", "foreign_dinov3_band0892",
    "foreign_dinov3_band208",
}
TARGET_BOTTOM3 = frozenset({"MCL", "Fracture", "Lateral OA"})
N_PERM = 500
SEED = 2026


def pairwise_spearman_mean(ranked: np.ndarray) -> tuple[float, np.ndarray]:
    """ranked: (m members, 12) per-row ranks. Spearman == Pearson on ranks."""
    z = ranked - ranked.mean(axis=1, keepdims=True)
    denom = np.sqrt((z ** 2).sum(axis=1))
    corr = (z @ z.T) / np.outer(denom, denom)
    iu = np.triu_indices(corr.shape[0], k=1)
    pairs = corr[iu]
    return float(pairs.mean()), pairs


def main() -> None:
    ref = np.load(REF)
    ref_studies = [str(s) for s in ref["studies"].tolist()]
    pos = {s: i for i, s in enumerate(ref_studies)}
    t0 = (np.asarray(ref["truth"], float) >= 0.5).astype(int)

    names, aucs = [], []
    n_scored = n_void = 0
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
            n_void += 1
            continue
        names.append(name)
        aucs.append([roc_auc_score(t0[:, j], pr[:, j]) for j in range(12)])

    auc = np.asarray(aucs)                      # (m, 12)
    m = auc.shape[0]
    col_median = np.median(auc, axis=0)
    centered = auc - col_median                 # member-specific weakness
    ranked = np.apply_along_axis(rankdata, 1, centered)

    print(f"CUT: per-column AUC @holdout909, midpoint binarised v2 truth; "
          f"scored {n_scored}, minus {n_void} VOID in-sample -> honest pool n={m} "
          f"(licence-tainted kept; profiles, not shipping)")
    print(f"\ncolumn pool medians (difficulty, subtracted):")
    for j, c in enumerate(COLUMNS):
        print(f"  {c:17s} {col_median[j]:.4f}")

    mean_r, pairs = pairwise_spearman_mean(ranked)
    print(f"\n=== pairwise Spearman of centered profiles ({m * (m - 1) // 2} pairs) ===")
    print(f"  mean {mean_r:+.4f} | p10 {np.percentile(pairs, 10):+.4f} | "
          f"median {np.percentile(pairs, 50):+.4f} | p90 {np.percentile(pairs, 90):+.4f} | "
          f"min {pairs.min():+.4f} | max {pairs.max():+.4f}")
    print(f"  pairs < 0: {(pairs < 0).mean() * 100:.1f}%")

    rng = np.random.default_rng(SEED)
    null_means = []
    for _ in range(N_PERM):
        perm = np.take_along_axis(
            centered, np.argsort(rng.random(centered.shape), axis=1), axis=1)
        pr_ranked = np.apply_along_axis(rankdata, 1, perm)
        null_means.append(pairwise_spearman_mean(pr_ranked)[0])
    null_means = np.asarray(null_means)
    print(f"  null (columns permuted within member, {N_PERM} reps): "
          f"mean {null_means.mean():+.4f}, "
          f"95% interval [{np.percentile(null_means, 2.5):+.4f}, "
          f"{np.percentile(null_means, 97.5):+.4f}], "
          f"observed vs null z = "
          f"{(mean_r - null_means.mean()) / null_means.std():.1f}")

    rank_sums = ranked.sum(axis=0)
    s_dev = ((rank_sums - rank_sums.mean()) ** 2).sum()
    kendall_w = 12 * s_dev / (m ** 2 * (12 ** 3 - 12))
    print(f"\n=== Kendall's W over {m} members x 12 columns: {kendall_w:.4f} "
          f"(0 = no agreement, 1 = identical rankings) ===")

    print(f"\n=== bottom-3 (weakest centered columns) census ===")
    from collections import Counter
    bottom3 = Counter()
    n_target = 0
    for i in range(m):
        weakest = frozenset(np.array(COLUMNS)[np.argsort(centered[i])[:3]])
        bottom3[weakest] += 1
        if weakest == TARGET_BOTTOM3:
            n_target += 1
    print(f"  exactly {{MCL, Fracture, Lateral OA}}: {n_target}/{m} "
          f"({n_target / m * 100:.1f}%)")
    print(f"  distinct bottom-3 sets: {len(bottom3)} of "
          f"{12 * 11 * 10 // 6} possible")
    for combo, cnt in bottom3.most_common(10):
        print(f"  {cnt:4d} ({cnt / m * 100:4.1f}%)  {sorted(combo)}")

    print(f"\n=== per-column: how often in a member's bottom-3 ===")
    counts = np.zeros(12, int)
    for i in range(m):
        counts[np.argsort(centered[i])[:3]] += 1
    for j in np.argsort(-counts):
        print(f"  {COLUMNS[j]:17s} {counts[j]:4d}/{m} ({counts[j] / m * 100:.1f}%)")


if __name__ == "__main__":
    main()
