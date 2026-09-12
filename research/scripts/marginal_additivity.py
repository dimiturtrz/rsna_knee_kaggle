"""Are the 0%-exposure marginals additive or saturating? (M386)

Pool and exposure map reused from marginals_zero_exposure.py (0% sub-pool,
n=290, @holdout909, midpoint-binarised v2 truth). k = number of the seven
positive attributes a member carries. Residual = solo - (base + own marginals),
base = median solo of the k=0 members (stated in the output).

merged-rijeka is INCLUDED as one of the seven, per the letter's list: the
question is additivity of the measured terms, and the 0% pool already handles
the leak; licence is a shipping constraint, not an additivity one. A variant
excluding it is printed for comparison.

Run: CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe research/scripts/marginal_additivity.py
"""

import os
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marginals_zero_exposure as mze  # noqa: E402

ATTRS = [
    ("labelwise head", lambda a: a["head"].startswith("labelwise"), 0.0138),
    ("distill", lambda a: "distill" in a["loss"], 0.0125),
    ("merged-rijeka", lambda a: a["labels"] == "merged-rijeka", 0.0117),
    ("hold-annot", lambda a: "hold-annot" in a["loss"], 0.0112),
    ("calib-tgt", lambda a: "calib-tgt" in a["loss"], 0.0109),
    ("blend-three-llm", lambda a: a["labels"] == "blend-three-llm", 0.0109),
    ("ema", lambda a: "ema" in a["loss"], 0.0105),
]
# marginals above are the 0%-pool values from marginals_zero_exposure.py's own
# output; recomputed below as a consistency check rather than trusted.


def main() -> None:
    honest, attrs = mze.load_pool()
    drop = mze.EXPOSURE_100 | mze.EXPOSURE_80
    zero = {n: s for n, s in honest.items() if n not in drop}
    print(f"POPULATION: 0%-exposure sub-pool n={len(zero)} "
          f"(honest {len(honest)} minus {len(set(honest) & drop)} exposed); "
          f"@holdout909, midpoint-binarised v2 truth")

    # recompute the marginals on this pool (consistency check vs the letter)
    recomputed = {}
    for label, f, _ in ATTRS:
        w = [s for n, s in zero.items() if f(attrs[n])]
        wo = [s for n, s in zero.items() if not f(attrs[n])]
        recomputed[label] = st.median(w) - st.median(wo)
    print("marginals recomputed vs carried: "
          + ", ".join(f"{l} {recomputed[l]:+.4f}/{m:+.4f}" for l, _, m in ATTRS))
    marg = {l: recomputed[l] for l, _, _ in ATTRS}

    def run(attr_set, tag):
        ks = {}
        for n, s in zero.items():
            k = sum(1 for l, f, _ in attr_set if f(attrs[n]))
            ks[n] = k
        k0 = [s for n, s in zero.items() if ks[n] == 0]
        base = st.median(k0)
        print(f"\n=== {tag}: base = median solo of k=0 members = {base:.4f} "
              f"(n={len(k0)}) ===")
        maxk = max(ks.values())
        rows = {}
        for k in range(maxk + 1):
            mem = [(n, s) for n, s in zero.items() if ks[n] == k]
            if not mem:
                continue
            resid = [s - (base + sum(marg[l] for l, f, _ in attr_set if f(attrs[n])))
                     for n, s in mem]
            rows[k] = (len(mem), st.median(s for _, s in mem), st.median(resid))
            print(f"  k={k}: n={len(mem):3d}  median solo {rows[k][1]:.4f}  "
                  f"median residual {rows[k][2]:+.4f}")
        thin = [k for k, (n_, _, _) in rows.items() if n_ < 5]
        if thin:
            print(f"  (thin bins n<5 at k={thin})")
        lo = [r for k, (n_, _, r) in rows.items() if k <= 1 for _ in range(n_)]
        hi = [r for k, (n_, _, r) in rows.items() if k >= 3 for _ in range(n_)]
        lo_all = [s - (base + sum(marg[l] for l, f, _ in attr_set if f(attrs[n])))
                  for n, s in zero.items() if ks[n] <= 1]
        hi_all = [s - (base + sum(marg[l] for l, f, _ in attr_set if f(attrs[n])))
                  for n, s in zero.items() if ks[n] >= 3]
        print(f"  coarse: k<=1 n={len(lo_all)} median residual "
              f"{st.median(lo_all):+.4f} | k>=3 n={len(hi_all)} median residual "
              f"{st.median(hi_all):+.4f}")
        top = sorted(zero.items(), key=lambda kv: -kv[1])[:8]
        print("  top-8 solos with their k: "
              + ", ".join(f"{n}(k={ks[n]})" for n, _ in top))
        return base, ks

    base7, ks7 = run(ATTRS, "PRIMARY: seven attributes (merged-rijeka included)")
    six = [a for a in ATTRS if a[0] != "merged-rijeka"]
    run(six, "VARIANT: six shippable attributes (merged-rijeka excluded)")

    print(f"\n=== TASK 2: distill_stack_max_s2026 under strict additivity "
          f"(context only — ~80% exposure, NOT in the 0% pool) ===")
    carried = ["distill", "labelwise head", "blend-three-llm", "calib-tgt",
               "hold-annot"]
    s = sum(marg[l] for l in carried)
    print(f"  recipe carries: {carried} (k=5 of the seven; not ema, not merged-rijeka)")
    print(f"  additive prediction = base {base7:.4f} + "
          f"{' + '.join(f'{marg[l]:.4f}' for l in carried)} = "
          f"{base7 + s:.4f}")
    print(f"  (realised-combo check: the full-pool realised top stack "
          f"oof_teacher 0.8685 / calib_hold_clean_distill 0.8641 carried 2-3 of "
          f"these terms, so additivity has never been observed at k=5)")


if __name__ == "__main__":
    main()
