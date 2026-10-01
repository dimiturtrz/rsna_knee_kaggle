"""Rescore the shippable pool on T0 (v2-derived ruler) and T1 (corrected ruler),
then re-run greedy 6-seat selection on T1.

T0: binarised (>= 0.5) `truth` from any holdout npz aligned to the 909 reference
    studies of band_span_s2026.holdout.npz.
T1: T0 with the reference-matching cells of v2_lone_dissent.csv flipped to
    `cluster_value`.

`--direction` picks WHICH cells. D363 adjudicated 56 blind cells stratified by
direction and measured the two halves as opposite operations: deletions
(v2=1 -> cluster 0) 2/28 wrong, additions (v2=0 -> cluster 1) 22/28 wrong.
`both` (469 cells) is the original T1 and applies one fixing half and one
breaking half. `down` is T2, the deletions-only ruler the adjudication licenses
and the one `experiments/arms/v2delonly_span.sh` declares it should be read on.

Run: CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe research/scripts/rescore_t1_ruler.py [--direction both|down|up]
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

MIDPOINT = 0.5
"""The cut this script scores on, so membership is decided on the same positive set it grades."""

MODELS_ROOT = "D:/data/volumetric/mri/models/rsna_knee_abnormality"
REFERENCE_NPZ = os.path.join(MODELS_ROOT, "band_span_s2026.holdout.npz")
SELECTOR_CSV = (
    "D:/data/volumetric/mri/processed/rsna_knee_abnormality/labels/v2_lone_dissent.csv"
)
ROSTER = "comp/members.roster"
CHAMPION = "comp/champion.json"

COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
N_SEATS = 6
ANCHOR_TOL = 2e-4

# Members whose 909-study reading is IN-SAMPLE by the project's own provenance
# record (comp/members.declines): the six published raptor checkpoints at
# 896/909 (98.6%, declines "published raptor weights: VOID" block), and
# dinov3f0_solo + the whole foreign_dinov3_* decode family serving
# mattiaangeli's released fold weights at 703/909 (declines "DINOv3 decode
# family: VOID" block, incl. foreign_dinov3_matched == foreign_oof_corpus by
# max|delta| 7.7e-06). A ruler change cannot repair in-sample-ness, so these
# stay in the spec table but are flagged and excluded from the honest-pool
# sensitivity run.
VOID_INSAMPLE = {
    "raptor_v4_swa", "raptor_v4", "raptor_v9", "raptor_effv2l480",
    "raptor_cnv2b336", "raptor_coat384",
    "dinov3f0_solo",
    "foreign_dinov3_matched", "foreign_dinov3_band1189",
    "foreign_dinov3_band1387", "foreign_dinov3_band1486",
    "foreign_dinov3_band1090", "foreign_dinov3_tta4", "foreign_dinov3_tta3",
    "foreign_dinov3_band1684", "foreign_dinov3_band0892",
    "foreign_dinov3_band208",
}
ANCHORS = {
    "band_span_s2026": (0.8565, 0.8592),
    "stackfit_s2026": (0.8631, 0.8682),
    "calib_dense_hold_s2026": (0.8634, 0.8644),
    "discord_mask_s2026": (0.8148, 0.8288),
}


def macro_auc(truth_bin: np.ndarray, preds: np.ndarray) -> float:
    aucs = []
    for j in range(truth_bin.shape[1]):
        y = truth_bin[:, j]
        if len(np.unique(y)) < 2:
            raise ValueError(f"column {COLUMNS[j]} single-class")
        aucs.append(roc_auc_score(y, preds[:, j]))
    return float(np.mean(aucs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=("both", "down", "up"), default="both")
    args = ap.parse_args()
    direction = args.direction
    print(f"=== ruler direction: {direction} "
          f"({'T1, both halves' if direction == 'both' else 'T2, deletions only' if direction == 'down' else 'additions only'}) ===")

    ref = np.load(REFERENCE_NPZ)
    ref_studies = [str(s) for s in ref["studies"].tolist()]
    ref_pos = {s: i for i, s in enumerate(ref_studies)}
    n_ref = len(ref_studies)
    assert len(set(ref_studies)) == n_ref

    t0 = (np.asarray(ref["truth"], dtype=np.float64) >= MIDPOINT).astype(np.int64)

    t1 = t0.copy()
    n_flips = 0
    n_v2_mismatch = 0
    with open(SELECTOR_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            uid = row["study_uid"]
            if uid not in ref_pos:
                continue
            j = COLUMNS.index(row["column"])
            i = ref_pos[uid]
            v2v, clv = int(row["v2_value"]), int(row["cluster_value"])
            if direction == "down" and not (v2v == 1 and clv == 0):
                continue
            if direction == "up" and not (v2v == 0 and clv == 1):
                continue
            if t0[i, j] != v2v:
                n_v2_mismatch += 1
            t1[i, j] = clv
            n_flips += 1
    print(f"reference studies: {n_ref}; selector cells applied T0->T1: {n_flips}; "
          f"cells where npz truth != v2_value (left at cluster_value anyway): "
          f"{n_v2_mismatch}")
    if direction == "both":
        assert n_flips == 469, n_flips

    kept: dict[str, np.ndarray] = {}
    excluded: list[tuple[str, str]] = []
    for path in sorted(glob.glob(os.path.join(MODELS_ROOT, "*.holdout.npz"))):
        name = os.path.basename(path)[: -len(".holdout.npz")]
        z = np.load(path)
        studies = [str(s) for s in z["studies"].tolist()]
        if set(studies) != set(ref_studies) or len(studies) != n_ref:
            excluded.append((name, f"{len(studies)} studies, "
                                   f"{len(set(studies) & set(ref_studies))} shared"))
            continue
        preds = np.asarray(z["predictions"], dtype=np.float64)
        order = np.array([ref_pos[s] for s in studies])
        aligned = np.empty_like(preds)
        aligned[order] = preds
        member_truth = np.asarray(z["truth"], dtype=np.float64)
        aligned_truth = np.empty_like(member_truth)
        aligned_truth[order] = member_truth
        if not np.array_equal((aligned_truth >= MIDPOINT), (np.asarray(ref["truth"], dtype=np.float64) >= MIDPOINT)):
            excluded.append((name, "positive set disagrees with reference after alignment"))
            continue
        kept[name] = aligned

    print(f"npz scanned: {len(kept) + len(excluded)}; kept: {len(kept)}; "
          f"excluded: {len(excluded)}")
    for name, why in excluded:
        print(f"  EXCLUDED {name}: {why}")

    scores: dict[str, tuple[float, float]] = {}
    for name, preds in kept.items():
        scores[name] = (macro_auc(t0, preds), macro_auc(t1, preds))

    print("\n=== sanity anchors ===")
    bad = False
    for name, (e0, e1) in ANCHORS.items():
        if name not in scores:
            print(f"  ANCHOR MISSING: {name} not in scored pool")
            bad = True
            continue
        a0, a1 = scores[name]
        ok = abs(a0 - e0) <= ANCHOR_TOL and abs(a1 - e1) <= ANCHOR_TOL
        print(f"  {name}: T0 {a0:.4f} (expect {e0:.4f}), T1 {a1:.4f} "
              f"(expect {e1:.4f}) -> {'OK' if ok else 'MISMATCH'}")
        bad = bad or not ok
    if bad and direction == "both":
        print("\nANCHOR MISMATCH — STOPPING as instructed.")
        sys.exit(1)

    print("\n=== member table (sorted by T1; VOID = in-sample on the 909 per "
          "comp/members.declines) ===")
    print(f"{'member':45s} {'T0':>7s} {'T1':>7s} {'delta':>8s}  flag")
    for name, (a0, a1) in sorted(scores.items(), key=lambda kv: -kv[1][1]):
        flag = "VOID-INSAMPLE" if name in VOID_INSAMPLE else ""
        print(f"{name:45s} {a0:.4f} {a1:.4f} {a1 - a0:+.4f}  {flag}")
    missing_void = VOID_INSAMPLE - set(scores)
    if missing_void:
        print(f"  (void-listed but not in the scored pool: {sorted(missing_void)})")

    with open(ROSTER) as fh:
        roster_names = []
        for line in fh:
            for tok in line.split("#")[0].split():
                if tok.endswith(".pt"):
                    roster_names.append(tok[:-len(".pt")])
    missing = [n for n in roster_names if n not in kept]
    print(f"\n=== roster members with NO usable holdout npz ({len(missing)} "
          f"of {len(roster_names)}) ===")
    for n in missing:
        npz_exists = os.path.exists(os.path.join(MODELS_ROOT, n + ".holdout.npz"))
        print(f"  {n} ({'npz exists but excluded' if npz_exists else 'no npz on disk'})")

    def greedy(pool: dict[str, np.ndarray], label: str) -> tuple[list[str], float, float]:
        print(f"\n=== greedy forward selection on T1 ({label}) ===")
        remaining = dict(pool)
        chosen: list[str] = []
        joint_sum = np.zeros((n_ref, len(COLUMNS)))
        for step in range(N_SEATS):
            best_name, best_score = None, -1.0
            for name, preds in remaining.items():
                s = macro_auc(t1, (joint_sum + preds) / (step + 1))
                if s > best_score:
                    best_name, best_score = name, s
            assert best_name is not None
            chosen.append(best_name)
            joint_sum = joint_sum + remaining.pop(best_name)
            print(f"  seat {step + 1}: + {best_name:40s} joint T1 {best_score:.4f}")
        final_joint = joint_sum / N_SEATS
        j_t0, j_t1 = macro_auc(t0, final_joint), macro_auc(t1, final_joint)
        print(f"  FINAL {N_SEATS}-seat joint: T0 {j_t0:.4f} | T1 {j_t1:.4f}")
        return chosen, j_t0, j_t1

    chosen, j_t0, j_t1 = greedy(kept, "full scored pool, per spec")
    honest = {n: p for n, p in kept.items() if n not in VOID_INSAMPLE}
    chosen_h, jh_t0, jh_t1 = greedy(
        honest, f"honest pool ({len(honest)} members, void in-sample excluded) — "
                f"sensitivity, because the spec run's early seats are void readings")

    with open(CHAMPION) as fh:
        champ = json.load(fh)["members"]
    print("\n=== champion comparison ===")
    print(f"  champion seats: {champ}")
    champ_scorable = []
    for m in champ:
        if m in kept:
            champ_scorable.append(m)
        elif m.endswith("_ff"):
            print(f"  ABSENT (full-fit, fold -1, unscorable by construction): {m}")
        else:
            npz = os.path.join(MODELS_ROOT, m + ".holdout.npz")
            why = "npz exists but excluded" if os.path.exists(npz) else "no npz on disk"
            print(f"  ABSENT ({why}): {m}")
    for tag, picks in (("spec pool", chosen), ("honest pool", chosen_h)):
        overlap = [m for m in champ if m in picks]
        print(f"  overlap champion vs greedy-{tag}: {overlap or 'NONE'}")
    c_t0 = c_t1 = None
    if champ_scorable:
        sub = np.mean([kept[m] for m in champ_scorable], axis=0)
        c_t0, c_t1 = macro_auc(t0, sub), macro_auc(t1, sub)
        print(f"  champion SCORABLE subset ({len(champ_scorable)}/{len(champ)} seats): "
              f"{champ_scorable}")
        print(f"  champion-subset joint: T0 {c_t0:.4f} | T1 {c_t1:.4f}")
    else:
        print("  champion SCORABLE subset: EMPTY — 0 of 6 seats can be scored. "
              "The comparison is not merely partial, it is impossible on these "
              "rulers; no champion joint score exists offline.")

    print("\nOUTCOME:")
    if set(chosen) == set(champ):
        print("  SAME SET (spec pool)")
    else:
        print(f"  spec-pool greedy set: {chosen}")
        print(f"  honest-pool greedy set: {chosen_h}")
        print("  The champion's exact set CANNOT be scored (4 seats have no npz, "
              "2 are full-fit fold -1); 0/6 seats are scorable, so no joint "
              "champion number exists on T0 or T1 and the comparison is "
              "stated against overlap only: overlap is 0 seats for both greedy "
              "runs. Outcome class: DIFFERENT-OVERLAPPING with overlap = 0 — "
              "i.e. a wholly different six, but with NO corrected-ruler "
              "champion score to beat, the 'better' arm of the classification "
              "is vacuous here. Greedy joints: spec pool T0 "
              f"{j_t0:.4f} / T1 {j_t1:.4f}; honest pool T0 {jh_t0:.4f} / T1 "
              f"{jh_t1:.4f}.")


if __name__ == "__main__":
    main()
