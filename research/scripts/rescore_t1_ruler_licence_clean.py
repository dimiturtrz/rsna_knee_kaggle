"""Licence-clean greedy 6-seat selection on the T1 (corrected) ruler.

Same T0/T1 construction and same honest pool as rescore_t1_ruler.py (17
in-sample void members excluded), PLUS every member whose provenance touches
Rijeka, fastMRI, MRNet or any merged corpus/scheme, sourced from
comp/members.roster `# licence:` tags and comp/members.declines.

Run: CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe research/scripts/rescore_t1_ruler_licence_clean.py
"""

import csv
import glob
import os

import numpy as np
from sklearn.metrics import roc_auc_score

MODELS_ROOT = "D:/models/rsna_knee_abnormality"
REFERENCE_NPZ = os.path.join(MODELS_ROOT, "band_span_s2026.holdout.npz")
SELECTOR_CSV = (
    "D:/data/volumetric/mri/processed/rsna_knee_abnormality/labels/v2_lone_dissent.csv"
)

COLUMNS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
N_SEATS = 6

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

# name -> (corpus, source of the exclusion). Line numbers are of this read.
LICENCE_DROP = {
    "merged_blend_cf": ("rijeka", "roster :406 `# licence: rijeka`"),
    "merged_blend_s2026": ("rijeka", "roster :446 `# licence: rijeka`"),
    "merged_dense_cf": ("rijeka", "roster :495 `# licence: rijeka`"),
    "merged_span_cf": ("rijeka", "roster :516 `# licence: rijeka`"),
    "merged_corpus_s2026": ("rijeka", "roster :585 `# licence: rijeka`"),
    "merged3_corpus_s2026": ("mrnet", "roster :639 `# licence: mrnet`"),
    "merged_distill_cf": ("rijeka (merged_blend_v1)",
                          "declines :115 - one variable off stack_increments is "
                          "the corpus, blend_three_v1 -> merged_blend_v1"),
    "merged_labelwise_cf": ("rijeka (merged_blend_v1)",
                            "declines :128 - 'the corpus again', one variable off "
                            "labelwise_dense_s2026"),
    "merged_blend_cf_r224": ("rijeka", "declines :650 - resize price for merged_blend_cf"),
    "merged_blend_cf_r280": ("rijeka", "declines :651 - resize price for merged_blend_cf"),
    "merged_blend_cf_r392": ("rijeka", "declines :685 - resize price for merged_blend_cf"),
    "merged_blend_cf_n12b0.1-0.9": ("rijeka", "declines :649 - regroup price for merged_blend_cf"),
    "merged_dense_cf_run1": ("rijeka", "declines :921 - one-epoch repeat of merged_dense_cf"),
    "merged3_oversample_s2026": ("mrnet (merged3)",
                                 "declines :956 - oversampling variant of the merged3 "
                                 "corpus, 'its corpus is contested'"),
    "ortho_fastmri_s2026": ("fastmri (mergedf)",
                            "declines :1044 - DECLINED ON LICENCE, fitted on the mergedf "
                            "cache (our corpus + NYU fastMRI)"),
    "ortho_fastmri_masked_s2026": ("fastmri (mergedf)",
                                   "declines :1068 - DECLINED ON LICENCE, fitted on mergedf"),
    "mrnet_warm_s2026": ("mrnet",
                         "declines :298 - trunk warm-started from MRNet pretrain; "
                         "CC/RUA copyright question knee-ib4t unresolved"),
    "mrnet_warm_s2026_run1": ("mrnet", "declines :940 - the higher MRNet run of the pair"),
}


def macro_auc(truth_bin: np.ndarray, preds: np.ndarray) -> float:
    return float(np.mean([
        roc_auc_score(truth_bin[:, j], preds[:, j])
        for j in range(truth_bin.shape[1])
    ]))


def main() -> None:
    ref = np.load(REFERENCE_NPZ)
    ref_studies = [str(s) for s in ref["studies"].tolist()]
    ref_pos = {s: i for i, s in enumerate(ref_studies)}
    n_ref = len(ref_studies)

    t0 = (np.asarray(ref["truth"], dtype=np.float64) >= 0.5).astype(np.int64)
    t1 = t0.copy()
    n_flips = 0
    with open(SELECTOR_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            uid = row["study_uid"]
            if uid not in ref_pos:
                continue
            t1[ref_pos[uid], COLUMNS.index(row["column"])] = int(row["cluster_value"])
            n_flips += 1
    assert n_flips == 469

    kept: dict[str, np.ndarray] = {}
    for path in sorted(glob.glob(os.path.join(MODELS_ROOT, "*.holdout.npz"))):
        name = os.path.basename(path)[: -len(".holdout.npz")]
        z = np.load(path)
        studies = [str(s) for s in z["studies"].tolist()]
        if set(studies) != set(ref_studies) or len(studies) != n_ref:
            continue
        preds = np.asarray(z["predictions"], dtype=np.float64)
        order = np.array([ref_pos[s] for s in studies])
        aligned = np.empty_like(preds)
        aligned[order] = preds
        truth = np.empty_like(aligned)
        truth[order] = np.asarray(z["truth"], dtype=np.float64)
        if not np.allclose(truth, np.asarray(ref["truth"], dtype=np.float64)):
            continue
        kept[name] = aligned

    honest = {n: p for n, p in kept.items() if n not in VOID_INSAMPLE}
    dropped_present = sorted(set(honest) & set(LICENCE_DROP))
    clean = {n: p for n, p in honest.items() if n not in LICENCE_DROP}

    print(f"scored pool {len(kept)} -> honest {len(honest)} "
          f"-> licence-clean {len(clean)}")
    print(f"\n=== dropped members ({len(dropped_present)} of "
          f"{len(LICENCE_DROP)} listed are in the scored pool) ===")
    for name in dropped_present:
        corpus, source = LICENCE_DROP[name]
        print(f"  {name:35s} {corpus:25s} {source}")
    not_in_pool = sorted(set(LICENCE_DROP) - set(kept))
    if not_in_pool:
        print(f"  (listed but not in the scored pool: {not_in_pool})")

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

    chosen, c_t0, c_t1 = greedy(clean, f"licence-clean pool, {len(clean)} members")
    print(f"\n=== constraint cost ===")
    print(f"  licence-clean T1 joint {c_t1:.4f} vs unconstrained honest-pool "
          f"T1 joint 0.8850 -> cost {c_t1 - 0.8850:+.4f}")
    print(f"  licence-clean T0 joint {c_t0:.4f} vs unconstrained honest-pool "
          f"T0 joint 0.8762 -> {c_t0 - 0.8762:+.4f}")
    print(f"\n  selected seats: {chosen}")
    print(f"  foreign_labels_dense_s2026 survives the filter: "
          f"{'foreign_labels_dense_s2026' in clean}; selected: "
          f"{'foreign_labels_dense_s2026' in chosen}")


if __name__ == "__main__":
    main()
