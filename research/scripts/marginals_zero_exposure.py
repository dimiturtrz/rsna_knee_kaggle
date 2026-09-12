"""Re-fit solo_rank per-attribute marginals on the 0%-exposure sub-pool (M385).

Exposure map from letter M383/M385 (authoritative, research/scripts/zero_exposure_ceiling.py):
100% fold-0-exposure teachers and ~80% foreign-corpus teachers, applied by NAME
to the honest pool; marginals re-fitted on what remains. Same estimator and cut
as solo_rank.py: median macro AUC with vs without the attribute, @holdout909,
midpoint-binarised v2 truth. Also: zero-joint pairs/triples among the seven
non-null attributes, and the two-roster packaged population rebuild.

Run: CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe research/scripts/marginals_zero_exposure.py
"""

import glob
import os
import statistics as st
import sys
from itertools import combinations

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solo_rank  # noqa: E402  (attribute machinery, identical to the original fit)

ROOT = solo_rank.ROOT
REF = solo_rank.REF

EXPOSURE_100 = {
    "oof_teacher_s2026", "seat_distill_lr4e5_s2026", "distill_v3base_oof_s2026",
    "compose_v4b_oof_s2026", "seat_oof5_distill_s2026", "seat_distill_gen2_s2026",
    "stack_increments_s2026", "stack_increments_base_s2026",
    "stack_increments_fx_s2026", "stack_increments_ff", "raptor_distill_s2026",
    "cnx_distill_warm_s2026", "swin_distill_pre_s2026", "merged_distill_cf",
    "retrain_six_distill_s2026",
}
EXPOSURE_80 = {
    "foreign_distill_mix07_s2026", "foreign_distill_mix30_s2026",
    "foreign_distill_calib_s2026", "foreign_dissent_s2026",
    "distill_stack_max_s2026",
}


def load_pool():
    ref = np.load(REF)
    ref_studies = [str(s) for s in ref["studies"].tolist()]
    pos = {s: i for i, s in enumerate(ref_studies)}
    t0 = (np.asarray(ref["truth"], float) >= 0.5).astype(int)

    import json
    recipes = {}
    for p in glob.glob(os.path.join(ROOT, "*.metrics.jsonl")):
        name = os.path.basename(p)[:-len(".metrics.jsonl")]
        try:
            recipes[name] = json.loads(open(p).readline())["recipe"]
        except (json.JSONDecodeError, KeyError):
            pass

    scores = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "*.holdout.npz"))):
        name = os.path.basename(p)[:-len(".holdout.npz")]
        z = np.load(p)
        st_ = [str(s) for s in z["studies"].tolist()]
        if set(st_) != set(ref_studies) or len(st_) != 909:
            continue
        order = np.array([pos[s] for s in st_])
        pr = np.empty((909, 12)); pr[order] = z["predictions"]
        tr = np.empty((909, 12)); tr[order] = z["truth"]
        if not np.allclose(tr, np.asarray(ref["truth"], float)):
            continue
        scores[name] = float(np.mean([roc_auc_score(t0[:, j], pr[:, j])
                                      for j in range(12)]))

    attrs, no_recipe = {}, set()
    for name in scores:
        rec = recipes.get(name) or recipes.get(solo_rank.parent_stem(name))
        attrs[name] = solo_rank.build_attributes(name, rec, no_recipe)
    honest = {n: s for n, s in scores.items() if n not in solo_rank.VOID}
    return honest, attrs


def make_tests():
    return [
        ("distill", lambda a: "distill" in a["loss"]),
        ("blend-three-llm", lambda a: a["labels"] == "blend-three-llm"),
        ("labelwise head", lambda a: a["head"].startswith("labelwise")),
        ("calib-tgt", lambda a: "calib-tgt" in a["loss"]),
        ("hold-annot", lambda a: "hold-annot" in a["loss"]),
        ("ema", lambda a: "ema" in a["loss"]),
        ("slices>=20", lambda a: a["slices"].isdigit() and int(a["slices"]) >= 20),
        ("merged-rijeka", lambda a: a["labels"] == "merged-rijeka"),
        ("grade_ladder", lambda a: "grade_ladder" in a["loss"]),
        ("mixup", lambda a: "mixup" in a["loss"]),
        ("res 336", lambda a: a["res"] == "336"),
        ("res 224", lambda a: a["res"] == "224"),
        ("span>=0.70", lambda a: a["span"] != "?" and float(a["span"]) >= 0.70),
        ("dinov2-small", lambda a: a["family"] == "dinov2" and a["size"] == "small"),
        ("dinov3", lambda a: a["family"] == "dinov3"),
        ("external extra-corpus", lambda a: a["external"].startswith("extra-corpus")),
        ("external medical-pretrain", lambda a: a["external"] == "medical-pretrain"),
        ("epochs>=24", lambda a: a["epochs"].isdigit() and int(a["epochs"]) >= 24),
    ]


def marginals(pool, attrs, tests):
    out = {}
    for label, f in tests:
        w = [s for n, s in pool.items() if f(attrs[n])]
        wo = [s for n, s in pool.items() if not f(attrs[n])]
        out[label] = (len(w), st.median(w) if w else float("nan"),
                      len(wo), st.median(wo) if wo else float("nan"),
                      (st.median(w) - st.median(wo)) if w and wo else float("nan"))
    return out


def main():
    honest, attrs = load_pool()
    tests = make_tests()

    in100 = sorted(set(honest) & EXPOSURE_100)
    in80 = sorted(set(honest) & EXPOSURE_80)
    zero = {n: s for n, s in honest.items()
            if n not in EXPOSURE_100 and n not in EXPOSURE_80}
    print(f"POPULATION: honest pool {len(honest)} @holdout909-midpoint-v2 "
          f"(scored npz minus {len(solo_rank.VOID)} void); "
          f"dropped 100%-exposure {len(in100)}: {in100}")
    print(f"          dropped ~80%-exposure {len(in80)}: {in80}")
    print(f"          -> 0%-exposure sub-pool {len(zero)}")
    missing = sorted((EXPOSURE_100 | EXPOSURE_80) - set(honest))
    print(f"          map names not in tonight's honest pool ({len(missing)}): {missing}")

    tagged = sorted(n for n in honest
                    if "distill" in attrs[n]["loss"] or attrs[n]["warm"] == "raptor-features")
    drop = set(in100) | set(in80)
    print(f"\nreconciliation: my attributes tag {len(tagged)} members "
          f"distill/raptor-teacher; of the drop set, "
          f"{len([n for n in drop if n in honest and n not in tagged])} are NOT "
          f"tagged by my attributes: "
          f"{sorted(n for n in drop if n in honest and n not in tagged)}")
    print(f"  tagged-but-clean (stay in 0% pool, zero fold-0 exposure by the map): "
          f"{sorted(n for n in tagged if n not in drop)}")

    full_m = marginals(honest, attrs, tests)
    zero_m = marginals(zero, attrs, tests)
    print(f"\n=== MARGINALS: full honest (n={len(honest)}) vs 0%-exposure (n={len(zero)}) ===")
    print(f"{'attribute':26s} {'d_full':>7s} {'n_with':>6s} | {'d_zero':>7s} "
          f"{'n_with':>6s} | {'move':>7s}")
    for label, _ in tests:
        nf, mf, nwf, mwf, df = full_m[label]
        nz, mz, nwz, mwz, dz = zero_m[label]
        note = ""
        if nf == nz and abs(df - dz) > 0.002:
            note = "  <-- moved >0.002 with NO member loss: stacking composition"
        print(f"{label:26s} {df:+.4f} {nf:6d} | {dz:+.4f} {nz:6d} | {dz - df:+.4f}{note}")

    rng = np.random.default_rng(2026)
    dw = [s for n, s in zero.items() if "distill" in attrs[n]["loss"]]
    dwo = [s for n, s in zero.items() if "distill" not in attrs[n]["loss"]]
    boots = []
    dw_a, dwo_a = np.array(dw), np.array(dwo)
    for _ in range(1000):
        a = rng.choice(dw_a, size=len(dw_a), replace=True)
        b = rng.choice(dwo_a, size=len(dwo_a), replace=True)
        boots.append(np.median(a) - np.median(b))
    print(f"\ndistill on 0% pool: n={len(dw)} (was "
          f"{sum(1 for n in honest if 'distill' in attrs[n]['loss'])}), "
          f"marginal {zero_m['distill'][4]:+.4f}, "
          f"bootstrap 95% [{np.percentile(boots, 2.5):+.4f}, "
          f"{np.percentile(boots, 97.5):+.4f}] over members (1000 reps)")

    print(f"\n=== TASK 2: zero-joint pairs/triples among the seven non-null attributes "
          f"(on the 0% sub-pool, n={len(zero)}) ===")
    seven = tests[:7]
    pos_attrs = [(label, f) for label, f in seven if zero_m[label][4] > 0]
    print(f"  positive-marginal components on 0% pool: "
          f"{[(l, round(zero_m[l][4], 4)) for l, _ in pos_attrs]}")
    for r in (2, 3):
        for combo in combinations(pos_attrs, r):
            joint = [n for n in zero
                     if all(f(attrs[n]) for _, f in combo)]
            if not joint:
                labels = " + ".join(l for l, _ in combo)
                print(f"  NEVER COMBINED ({r}): {labels}")

    print(f"\n=== TASK 3: two-roster packaged population ===")
    names_all = []
    for rf in ("comp/members.roster", "comp/members2.roster"):
        names = []
        for line in open(rf):
            for tok in line.split("#")[0].split():
                if tok.endswith(".pt"):
                    names.append(tok[:-3])
        print(f"  {rf}: {len(names)} names")
        names_all += names
    dupes = [n for n in set(names_all) if names_all.count(n) > 1]
    print(f"  union {len(names_all)}, duplicates across the two files: {dupes or 'none'}")

    ref = np.load(REF)
    ref_studies = [str(s) for s in ref["studies"].tolist()]
    pos = {s: i for i, s in enumerate(ref_studies)}
    t0 = (np.asarray(ref["truth"], float) >= 0.5).astype(int)
    ok, no_npz, bad_st, bad_tr, bad_ep = 0, [], [], [], []
    for name in dict.fromkeys(names_all):
        p = os.path.join(ROOT, name + ".holdout.npz")
        if not os.path.exists(p):
            no_npz.append(name)
            continue
        z = np.load(p)
        st_ = [str(s) for s in z["studies"].tolist()]
        if set(st_) != set(ref_studies) or len(st_) != 909:
            bad_st.append(name)
            continue
        order = np.array([pos[s] for s in st_])
        tr = np.empty((909, 12)); tr[order] = z["truth"]
        if not np.array_equal((tr >= 0.5).astype(int), t0):
            bad_tr.append(name)
            continue
        if int(np.asarray(z["epoch"])) < 0:
            bad_ep.append(name)
            continue
        ok += 1
    print(f"  packaged-honest from BOTH rosters: {ok} "
          f"(excluded: {len(no_npz)} no npz {sorted(no_npz)}, "
          f"{len(bad_st)} studies, {len(bad_tr)} truth, {len(bad_ep)} epoch<0 {bad_ep})")
    for nm in ("seat_clean_distill_s2026", "clean_calib_s2026", "chainfour_s2026"):
        print(f"  {nm}: rostered={nm in set(names_all)}")
    print(f"  vs main's 133: delta {133 - ok}")


if __name__ == "__main__":
    main()
