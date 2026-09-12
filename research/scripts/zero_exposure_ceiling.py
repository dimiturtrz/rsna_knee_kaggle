"""M383 ask 2: the 0%-exposure pool ceiling.

Population: ALL *.holdout.npz under MODELS ROOT passing the three honesty
filters (epoch >= 0, exact 909-study set, binarised truth cell-identical to
the reference) -- the roster parse misses several shipped members
(seat_clean_distill, clean_calib, chainfour are not .pt tokens in
comp/members.roster), so this script scans npz directly.

Exposure classes from experiments/arms/*.sh --teacher-targets:
- 0%: teacher_clean / teacher_clean_full (--exclude-fold 0 refits),
  teacher_fold0 (six fold-0-holdout members), no distill.
- ~80%: teacher_foreign_corpus (mattiaangeli 5 folds, each fitted on 4/5 of
  ALL studies incl. fold 0).
- 100%: teacher_oof_corpus/oof5/gen2 (fold 0 in training), raptor_teacher
  (published fold -1 weights).

Run: CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe research/scripts/zero_exposure_ceiling.py
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from greedy_halfsplit import ROOT, fast_macro_auc  # noqa: E402

EXPOSED_100 = {
    "oof_teacher_s2026": "teacher_oof_corpus",
    "seat_distill_lr4e5_s2026": "teacher_oof_corpus",
    "distill_v3base_oof_s2026": "teacher_oof_corpus",
    "compose_v4b_oof_s2026": "teacher_oof_corpus",
    "seat_oof5_distill_s2026": "teacher_oof5",
    "seat_distill_gen2_s2026": "teacher_gen2_full",
    "stack_increments_s2026": "raptor_teacher",
    "stack_increments_base_s2026": "raptor_teacher",
    "stack_increments_fx_s2026": "raptor_teacher",
    "stack_increments_ff": "raptor_teacher",
    "raptor_distill_s2026": "raptor_teacher",
    "cnx_distill_warm_s2026": "raptor_teacher",
    "swin_distill_pre_s2026": "raptor_teacher",
    "merged_distill_cf": "raptor_teacher",
    "retrain_six_distill_s2026": "teacher_oof5",
}
EXPOSED_80 = {
    "foreign_distill_mix07_s2026": "teacher_foreign_corpus",
    "foreign_distill_mix30_s2026": "teacher_foreign_corpus",
    "foreign_distill_calib_s2026": "teacher_foreign_corpus",
    "foreign_dissent_s2026": "teacher_foreign_corpus",
    "distill_stack_max_s2026": "teacher_foreign_corpus",
}
SHIPPED_SIX = ["clean_calib_s2026", "chainfour_s2026", "stack_increments_s2026",
               "stack_increments_base_s2026", "calib_dense_hold_s2026",
               "seat_clean_distill_s2026"]


def main() -> None:
    ref = np.load(os.path.join(ROOT, "band_span_s2026.holdout.npz"))
    ref_studies = [str(s) for s in ref["studies"].tolist()]
    pos = {s: i for i, s in enumerate(ref_studies)}
    t0 = (np.asarray(ref["truth"], float) >= 0.5).astype(int)

    names, logits, solos = [], [], []
    for path in sorted(glob.glob(os.path.join(ROOT, "*.holdout.npz"))):
        name = os.path.basename(path)[: -len(".holdout.npz")]
        a = np.load(path)
        st = [str(s) for s in a["studies"].tolist()]
        if set(st) != set(ref_studies) or len(st) != 909:
            continue
        if "epoch" in a.files and int(np.asarray(a["epoch"])) < 0:
            continue
        order = np.array([pos[s] for s in st])
        pred = np.empty((909, 12)); pred[order] = a["predictions"]
        truth = np.empty((909, 12)); truth[order] = a["truth"]
        if not np.array_equal((truth >= 0.5).astype(int), t0):
            continue
        p = np.clip(pred.astype(float), 1e-6, 1 - 1e-6)
        names.append(name)
        logits.append(np.log(p / (1 - p)))
        solos.append(fast_macro_auc(t0, pred))
    logits = np.array(logits)
    solos = np.array(solos)
    print(f"honest npz pool n={len(names)}")

    exposed = {**EXPOSED_100, **EXPOSED_80}
    present = sorted([(n, t) for n, t in exposed.items() if n in names],
                     key=lambda x: -solos[names.index(x[0])])
    print(f"\nexposed members in pool ({len(present)}):")
    for n, t in present:
        print(f"  {solos[names.index(n)]:.4f} {n:34s} {t}")

    zero = [i for i, n in enumerate(names) if n not in exposed]
    print(f"\n0%-exposure sub-pool n={len(zero)} of {len(names)}")

    ship_idx = [names.index(n) for n in SHIPPED_SIX]
    ship_score = fast_macro_auc(t0, logits[ship_idx].mean(axis=0))
    print(f"\nshipped honestsix (recomputed, midpoint cut): {ship_score:.4f}")
    for i in ship_idx:
        print(f"  {solos[i]:.4f} {names[i]:34s} exposure: "
              f"{exposed.get(names[i], '0%')}")

    order0 = sorted(zero, key=lambda i: -solos[i])
    top6 = order0[:6]
    zero_score = fast_macro_auc(t0, logits[top6].mean(axis=0))
    print(f"\n0%-exposure no-search top-six: {zero_score:.4f}")
    for i in top6:
        print(f"  {solos[i]:.4f} {names[i]}")
    print(f"cost vs shipped six: {zero_score - ship_score:+.4f}")

    keep80 = [i for i, n in enumerate(names) if n not in EXPOSED_100]
    top80 = sorted(keep80, key=lambda i: -solos[i])[:6]
    s80 = fast_macro_auc(t0, logits[top80].mean(axis=0))
    print(f"\ntop-six excluding only the 100% line (foreign ~80% kept): {s80:.4f}")
    for i in top80:
        print(f"  {solos[i]:.4f} {names[i]:34s} {exposed.get(names[i], '0%')}")


if __name__ == "__main__":
    main()
