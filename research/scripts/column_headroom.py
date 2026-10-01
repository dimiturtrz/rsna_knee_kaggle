"""Per-column headroom: modelling gap or label gap? (M388)

Gold: audit400/holdout469 — the 469 blind-adjudicated cells = EVERY v2-vs-cluster
dissent cell inside holdout909 (verified per-column against
labels/v2_lone_dissent.csv ∩ the 909: exact match, so the per-column dissent base
IS the gold n; no sampling scale-up — stated in the output). These are SELECTED
dissent cells, not a random column sample: contradiction rates describe the
dissent stratum only.

Cuts: per-column AUC of honest-pool members @holdout909, midpoint-binarised v2
truth. Error estimates: (i) gold-only lower bound contr/909; (ii) strata-weighted
dissent wrong at the gold rate + non-dissent wrong at the measured unanimous
0.033. Ceiling ~= 1 - eps (symmetric-flip approximation — v2's measured error is
directional over-call, so approximate; stated).

Run: CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe research/scripts/column_headroom.py
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = "D:/data/volumetric/mri/processed/rsna_knee_abnormality"
MODELS = "D:/data/volumetric/mri/models/rsna_knee_abnormality"
COLS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
        "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
        "Contusion", "Fracture"]
VOID = {
    "raptor_v4_swa", "raptor_v4", "raptor_v9", "raptor_effv2l480",
    "raptor_cnv2b336", "raptor_coat384", "dinov3f0_solo",
    "foreign_dinov3_matched", "foreign_dinov3_band1189",
    "foreign_dinov3_band1387", "foreign_dinov3_band1486",
    "foreign_dinov3_band1090", "foreign_dinov3_tta4", "foreign_dinov3_tta3",
    "foreign_dinov3_band1684", "foreign_dinov3_band0892",
    "foreign_dinov3_band208",
}
UNANIMOUS_WRONG = 0.033   # measured v2-wrong rate on the v2=cluster stratum
MIN_GOLD = 30             # main's explicit small-n rule


def per_col_auc(t, s):
    ranks = rankdata(s, axis=0)
    out = np.empty(12)
    for j in range(12):
        pos = t[:, j] == 1
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        out[j] = (ranks[pos, j].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return out


def main() -> None:
    key = pd.read_csv(f"{ROOT}/audit400/holdout469/adjudication_key.csv")
    frames = []
    for f in sorted(glob.glob(f"{ROOT}/audit400/holdout469/answers_*of10.csv")):
        a = pd.read_csv(f)
        col = "verdict" if "verdict" in a.columns else "answer"
        frames.append(a[["row_id", col]].rename(columns={col: "verdict"}))
    df = key.merge(pd.concat(frames), on="row_id")
    df["contr"] = (df.verdict != df.v2_value).astype(int)
    sel = pd.read_csv(f"{ROOT}/labels/v2_lone_dissent.csv")

    ref = np.load(f"{MODELS}/band_span_s2026.holdout.npz")
    ref_studies = [str(s) for s in ref["studies"].tolist()]
    pos = {s: i for i, s in enumerate(ref_studies)}
    t0 = (np.asarray(ref["truth"], float) >= 0.5).astype(int)
    sel909 = sel[sel.study_uid.isin(set(ref_studies))]
    match = (df.groupby("column").size()
             == sel909.groupby("column").size()).all() and len(df) == len(sel909)
    print(f"GOLD: {len(df)} adjudicated cells; selector cells inside holdout909: "
          f"{len(sel909)}; per-column match: {match} -> the audit adjudicated "
          f"EVERY dissent cell in the holdout, so the per-column dissent base "
          f"IS the gold n (no scale-up; the 2,315 in v2_lone_dissent.csv is the "
          f"corpus-wide count). Rates describe the DISSENT stratum, not the "
          f"column at large.")
    print(f"overall v2 contradiction in gold: {df.contr.mean():.4f} (record: 0.891); "
          f"tear dissent contradicted: "
          f"{int(df[df.column.isin(['ACL','MCL','Medial Meniscus','Lateral Meniscus'])].contr.sum())}/277 "
          f"(record: 0/277 for v2); verdict agrees with cluster_value on "
          f"{(df.verdict == df.cluster_value).mean():.4f} of cells")

    names, aucs = [], []
    for p in sorted(glob.glob(f"{MODELS}/*.holdout.npz")):
        name = os.path.basename(p)[:-len(".holdout.npz")]
        if name in VOID:
            continue
        z = np.load(p)
        st_ = [str(s) for s in z["studies"].tolist()]
        if set(st_) != set(ref_studies) or len(st_) != 909:
            continue
        order = np.array([pos[s] for s in st_])
        pr = np.empty((909, 12)); pr[order] = z["predictions"]
        tr = np.empty((909, 12)); tr[order] = z["truth"]
        if not np.allclose(tr, np.asarray(ref["truth"], float)):
            continue
        names.append(name)
        aucs.append(per_col_auc(t0, pr))
    A = np.asarray(aucs)
    best = A.max(axis=0)
    best_name = [names[int(i)] for i in A.argmax(axis=0)]
    med = np.median(A, axis=0)
    print(f"\nmember frame: honest pool n={len(names)} @holdout909-midpoint-v2 "
          f"(void foreign excluded)")

    print(f"\n{'column':17s} {'gold':>4s} {'contr':>5s} {'rate':>6s} "
          f"{'eps_i':>6s} {'eps_ii':>6s} {'ceil_i':>6s} {'ceil_ii':>7s} "
          f"{'medAUC':>6s} {'bestAUC':>7s} {'gap':>6s}  class")
    rows = []
    for j, c in enumerate(COLS):
        sub = df[df.column == c]
        n_gold, contr = len(sub), int(sub.contr.sum())
        rate = contr / n_gold if n_gold else float("nan")
        eps_i = contr / 909
        eps_ii = (contr + (909 - n_gold) * UNANIMOUS_WRONG) / 909
        ceil_i, ceil_ii = 1 - eps_i, 1 - eps_ii
        gap = ceil_ii - best[j]
        if n_gold < MIN_GOLD:
            rate_s = f"n too small (gold {n_gold})"
            cls = ("B?" if gap > 0.02 else "A?")
            cls += " [strata estimate only — no gold]"
        else:
            rate_s = f"{rate:.3f}"
            if ceil_ii <= 0.93 or best[j] >= ceil_ii - 0.005:
                cls = "C label-limited" if ceil_ii <= 0.93 else "A at ceiling"
                if best[j] >= ceil_ii - 0.005 and ceil_ii <= 0.93:
                    cls = "A/C at a LOW ceiling (labels)"
            elif gap <= 0.010:
                cls = "A near ceiling"
            elif ceil_ii >= 0.95:
                cls = "B modelling hole"
            else:
                cls = "B/C mixed"
        rows.append((c, n_gold, contr, rate_s, eps_i, eps_ii, ceil_i, ceil_ii,
                     med[j], best[j], gap, cls, best_name[j]))
        print(f"{c:17s} {n_gold:4d} {contr:5d} {rate_s:>6s} "
              f"{eps_i:6.4f} {eps_ii:6.4f} {ceil_i:6.3f} {ceil_ii:7.3f} "
              f"{med[j]:6.4f} {best[j]:7.4f} {gap:+6.3f}  {cls}")

    print(f"\nbest member per column: "
          + ", ".join(f"{r[0]}={r[12]}" for r in rows))
    macro_ceil = float(np.mean([r[7] for r in rows]))
    macro_best = float(np.mean([r[9] for r in rows]))
    macro_med = float(np.mean([r[8] for r in rows]))
    print(f"\nmacro: ceiling(ii) {macro_ceil:.4f} | best-member macro {macro_best:.4f} "
          f"| pool-median macro {macro_med:.4f} | best->ceiling "
          f"{macro_ceil - macro_best:+.4f}, median->ceiling "
          f"{macro_ceil - macro_med:+.4f}")
    dirn = df[df.contr == 1]
    print(f"direction of contradicted cells: v2=1->verdict 0 on "
          f"{(dirn.v2_value == 1).mean():.4f} (over-call), n={len(dirn)}")


if __name__ == "__main__":
    main()
