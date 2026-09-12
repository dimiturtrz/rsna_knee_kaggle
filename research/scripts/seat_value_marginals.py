"""M387, two tasks on the 133 packaged-honest pool (BOTH rosters).

TASK 1: seat-value marginals. Fixed base = the shipped honest six (contains two
raptor-teacher members by design — the question is seat value ADDED TO THE
SHIPPED SIX). Candidates = 133 minus the six minus the exposure map (stated n).
Per attribute X of the seven positives (merged-rijeka INCLUDED, consistent with
marginal_additivity.py): 20 seeded splits x 2 directions; on the train half pick
the best-solo candidate carrying X vs the best-solo candidate not carrying X;
score each seven (LogitAverage) on the test half; delta = carrying - not.

TASK 2: the marginal_additivity saturation curve, 0% pool unchanged, with the
k=5 bin augmented by distill_stack_max_s2026 (marked ~80%-exposed).

Run: CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe research/scripts/seat_value_marginals.py
"""

import os
import statistics as st
import sys
from collections import Counter

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marginals_zero_exposure as mze  # noqa: E402
import solo_rank  # noqa: E402

ROOT = solo_rank.ROOT
REF = solo_rank.REF
SIX = ["clean_calib_s2026", "chainfour_s2026", "stack_increments_s2026",
       "stack_increments_base_s2026", "calib_dense_hold_s2026",
       "seat_clean_distill_s2026"]
ATTRS = [
    ("labelwise", lambda a: a["head"].startswith("labelwise")),
    ("distill", lambda a: "distill" in a["loss"]),
    ("hold-annot", lambda a: "hold-annot" in a["loss"]),
    ("calib-tgt", lambda a: "calib-tgt" in a["loss"]),
    ("blend-three-llm", lambda a: a["labels"] == "blend-three-llm"),
    ("ema", lambda a: "ema" in a["loss"]),
    ("merged-rijeka", lambda a: a["labels"] == "merged-rijeka"),
]
N_SPLITS = 20
CLIP = 1e-6


def per_col_auc(y, s):
    ranks = rankdata(s, axis=0)
    out = np.empty(12)
    for j in range(12):
        pos = y[:, j] == 1
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        out[j] = (ranks[pos, j].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return out


def load_133():
    ref = np.load(REF)
    ref_studies = [str(s) for s in ref["studies"].tolist()]
    pos = {s: i for i, s in enumerate(ref_studies)}
    t0 = (np.asarray(ref["truth"], float) >= 0.5).astype(int)
    names_all = []
    for rf in ("comp/members.roster", "comp/members2.roster"):
        for line in open(rf):
            for tok in line.split("#")[0].split():
                if tok.endswith(".pt"):
                    names_all.append(tok[:-3])
    names, preds = [], []
    for name in dict.fromkeys(names_all):
        p = os.path.join(ROOT, name + ".holdout.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        st_ = [str(s) for s in z["studies"].tolist()]
        if set(st_) != set(ref_studies) or len(st_) != 909:
            continue
        order = np.array([pos[s] for s in st_])
        pr = np.empty((909, 12)); pr[order] = z["predictions"]
        tr = np.empty((909, 12)); tr[order] = z["truth"]
        if not np.array_equal((tr >= 0.5).astype(int), t0):
            continue
        if int(np.asarray(z["epoch"])) < 0:
            continue
        names.append(name)
        preds.append(pr)
    return names, np.asarray(preds), t0


def main() -> None:
    names, P, t0 = load_133()
    _, attrs = mze.load_pool()
    # the two-roster loader admits members on BINARISED truth equality (the M386
    # spec); mze's pool requires raw-truth allclose, so a member can be in the
    # 133 but lack attributes — build them from the recipe line directly.
    import json
    for n in names:
        if n not in attrs:
            rec = None
            p = os.path.join(ROOT, n + ".metrics.jsonl")
            if os.path.exists(p):
                try:
                    rec = json.loads(open(p).readline())["recipe"]
                except (json.JSONDecodeError, KeyError):
                    pass
            attrs[n] = solo_rank.build_attributes(n, rec, set())
            print(f"  attr fallback (binarised-truth admit, raw-truth differ): {n}")
    L = np.log(np.clip(P, CLIP, 1 - CLIP) / (1 - np.clip(P, CLIP, 1 - CLIP)))
    six = [names.index(n) for n in SIX]
    drop = mze.EXPOSURE_100 | mze.EXPOSURE_80
    exposed_in = sorted(set(names) & drop - set(SIX))
    cands = [i for i, n in enumerate(names)
             if n not in SIX and n not in drop]
    print(f"POPULATION: {len(names)} packaged-honest (both rosters); the six are "
          f"seated as shipped (two raptor-teacher members inside); candidates = "
          f"133 - 6 - {len(exposed_in)} exposed {exposed_in} -> n={len(cands)}")
    print(f"CUT: LogitAverage macro AUC, midpoint-binarised v2 truth @holdout909; "
          f"{N_SPLITS} splits x 2 = 40 readings per attribute")
    print(f"merged-rijeka INCLUDED as a seat attribute (consistent with "
          f"marginal_additivity.py); its candidates are licence-tagged for "
          f"shipping but 0%-exposure")

    def macro(idx_array, members, idx):
        return float(per_col_auc(t0[idx], L[members][:, idx].mean(axis=0)).mean())

    results = {l: [] for l, _ in ATTRS}
    abs_car = {l: [] for l, _ in ATTRS}
    abs_not = {l: [] for l, _ in ATTRS}
    control_picks = {l: Counter() for l, _ in ATTRS}
    attr_picks = {l: Counter() for l, _ in ATTRS}
    for sp in range(N_SPLITS):
        rng = np.random.default_rng(2026 + sp)
        perm = rng.permutation(909)
        for train, test in ((perm[:455], perm[455:]), (perm[455:], perm[:455])):
            solo = {i: float(per_col_auc(t0[train], L[i][train]).mean())
                    for i in cands}
            for label, f in ATTRS:
                carry = [i for i in cands if f(attrs[names[i]])]
                notcarry = [i for i in cands if not f(attrs[names[i]])]
                pc = max(carry, key=lambda i: solo[i])
                pn = max(notcarry, key=lambda i: solo[i])
                sc = macro(None, six + [pc], test)
                sn = macro(None, six + [pn], test)
                results[label].append(sc - sn)
                abs_car[label].append(sc)
                abs_not[label].append(sn)
                attr_picks[label][names[pc]] += 1
                control_picks[label][names[pn]] += 1

    print(f"\n=== TASK 1: seat-value delta (carrying X - not carrying X), "
          f"added to the shipped six ===")
    for label, _ in ATTRS:
        d = np.array(results[label])
        print(f"  {label:16s} delta {d.mean():+.4f}±{d.std():.4f} | "
              f"carry {np.mean(abs_car[label]):.4f} vs not "
              f"{np.mean(abs_not[label]):.4f} | win {(d > 0).sum()}/40 loss "
              f"{(d < 0).sum()}/40")
        print(f"      picks: {attr_picks[label].most_common(3)}")
        print(f"      control: {control_picks[label].most_common(3)}")

    print(f"\n=== TASK 2: saturation curve, 0% pool + augmented k=5 ===")
    honest, attrs_full = mze.load_pool()
    zero = {n: s for n, s in honest.items() if n not in drop}
    print(f"  0% pool n={len(zero)} (honest {len(honest)}): "
          f"distill_stack_clean_s2026 ENTERED since D380 — its npz landed during "
          f"this session, TRUNCATED at epoch 3 of 12 (one_cycle anneal never "
          f"ran): its 0.8426 is a FLOOR by the convergence_check rule, not a "
          f"realised k=5 point")
    marg = {"labelwise": 0.0138, "distill": 0.0125, "merged-rijeka": 0.0117,
            "hold-annot": 0.0112, "calib-tgt": 0.0109, "blend-three-llm": 0.0109,
            "ema": 0.0105}
    ks = {n: sum(1 for l, f in ATTRS if f(attrs_full[n])) for n in zero}
    base = st.median([s for n, s in zero.items() if ks[n] == 0])
    d380 = {0: (222, 0.8439, 0.0000), 1: (31, 0.8545, -0.0004),
            2: (23, 0.8575, -0.0085), 3: (11, 0.8607, -0.0191),
            4: (2, 0.8609, -0.0293), 5: (1, 0.8648, -0.0364)}
    print(f"  base (k=0 median) = {base:.4f}")
    for k in range(6):
        mem = [(n, s) for n, s in zero.items() if ks[n] == k]
        if not mem:
            continue
        resid = [s - (base + sum(marg[l] for l, f in ATTRS if f(attrs_full[n])))
                 for n, s in mem]
        med_s, med_r = st.median(s for _, s in mem), st.median(resid)
        on, os_, or_ = d380[k]
        flag = ""
        if k == 5:
            flag = ("  [n=2 incl. distill_stack_clean TRUNCATED-FLOOR; "
                    "converged-only: n=1 chainfour 0.8648 residual -0.0364]")
        print(f"  k={k}: n={len(mem):3d} median solo {med_s:.4f} "
              f"median residual {med_r:+.4f} "
              f"(D380: n={on}, {os_:.4f}, {or_:+.4f}; "
              f"d_residual {med_r - or_:+.4f}){flag}")
    dsm = honest.get("distill_stack_max_s2026")
    ch = zero.get("chainfour_s2026")
    add5 = base + sum(marg[l] for l in
                      ("distill", "labelwise", "blend-three-llm", "calib-tgt",
                       "hold-annot"))
    print(f"  augmented k=5 (EXPOSED member marked):")
    print(f"    chainfour_s2026          solo {ch:.4f} residual "
          f"{ch - 0.9012:+.4f} [clean, converged]")
    print(f"    distill_stack_max_s2026  solo {dsm:.4f} residual {dsm - add5:+.4f} "
          f"[~80% EXPOSED, converged e8/12; additive prediction {add5:.4f}]")
    print(f"    augmented converged k=5 median residual "
          f"{st.median([ch - 0.9012, dsm - add5]):+.4f} (n=2)")


if __name__ == "__main__":
    main()
