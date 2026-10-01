"""Seventh-member choice for a FIXED honest six: next-best-by-solo vs
least-profile-correlated (column-centred AUC profile, Spearman vs the six's
ensemble profile). Half-split transfer test, M383.

Frame identical to greedy_halfsplit.py: 117 packaged-honest @holdout909,
LogitAverage (clip 1e-6), macro AUC vs midpoint-binarised v2 truth,
20 seeded splits (default_rng(2026+split), 455/454), both directions = 40
readings.

k=8 extension construction: seventh fixed by the SOLO rule on the train half;
eighth picked by each rule (SOLO: next best solo not already seated;
DIVERSITY: least profile-correlated with the SEVEN's ensemble profile,
tie-break solo).

Run: CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe research/scripts/diversity_seventh.py
"""

import os
from collections import Counter

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = "D:/data/volumetric/mri/models/rsna_knee_abnormality"
REF = os.path.join(ROOT, "band_span_s2026.holdout.npz")
ROSTER = "comp/members.roster"
FIXED_SIX = [
    "clean_calib_s2026", "chainfour_s2026", "stack_increments_s2026",
    "stack_increments_base_s2026", "calib_dense_hold_s2026",
    "seat_clean_distill_s2026",
]
N_SPLITS = 20
SEED_BASE = 2026
CLIP = 1e-6


def per_col_auc(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    ranks = rankdata(s, axis=0)
    out = np.empty(12)
    for j in range(12):
        pos = y[:, j] == 1
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        out[j] = (ranks[pos, j].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return out


def main() -> None:
    ref = np.load(REF)
    ref_studies = [str(s) for s in ref["studies"].tolist()]
    pos = {s: i for i, s in enumerate(ref_studies)}
    t0 = (np.asarray(ref["truth"], float) >= 0.5).astype(int)

    roster = []
    for line in open(ROSTER):
        for tok in line.split("#")[0].split():
            if tok.endswith(".pt"):
                roster.append(tok[:-3])

    names, preds = [], []
    for name in roster:
        p = os.path.join(ROOT, name + ".holdout.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        st = [str(s) for s in z["studies"].tolist()]
        if set(st) != set(ref_studies) or len(st) != 909:
            continue
        order = np.array([pos[s] for s in st])
        pr = np.empty((909, 12)); pr[order] = z["predictions"]
        tr = np.empty((909, 12)); tr[order] = z["truth"]
        if not np.array_equal((tr >= 0.5).astype(int), t0):
            continue
        if int(np.asarray(z["epoch"])) < 0:
            continue
        names.append(name)
        preds.append(pr)

    # main's fixed six includes seat_clean_distill_s2026, which is NOT named in
    # comp/members.roster (a roster_check finding: scoreable, neither rostered
    # nor declined). Load it from its npz as a seated member; the CANDIDATE
    # pool for the seventh stays the 117 packaged-honest.
    extra = []
    for nm in FIXED_SIX:
        if nm in names:
            continue
        z = np.load(os.path.join(ROOT, nm + ".holdout.npz"))
        order = np.array([pos[str(s)] for s in z["studies"].tolist()])
        pr = np.empty((909, 12)); pr[order] = z["predictions"]
        extra.append((nm, pr))
    for nm, pr in extra:
        names.append(nm)
        preds.append(pr)

    P = np.asarray(preds)
    L = np.log(np.clip(P, CLIP, 1 - CLIP) / (1 - np.clip(P, CLIP, 1 - CLIP)))
    six = [names.index(nm) for nm in FIXED_SIX]
    rest = [i for i in range(len(names)) if i not in six]
    print(f"POPULATION: {len(names) - len(extra)} packaged-honest @holdout909 "
          f"(as greedy_halfsplit.py); fixed six loaded "
          f"({len(extra)} of the six are NOT rostered: "
          f"{[nm for nm, _ in extra] or 'none'}); candidates = packaged minus "
          f"the rostered five of the six")
    print(f"CUT: LogitAverage, macro AUC vs midpoint-binarised v2 truth; "
          f"{N_SPLITS} splits x 2 = {2 * N_SPLITS} readings; "
          f"pool medians for centring computed over all {len(names)} members "
          f"within the train half")

    def rules(train: np.ndarray):
        col_auc = np.array([per_col_auc(t0[train], L[i][train]) for i in range(len(names))])
        solo = col_auc.mean(axis=1)
        med = np.median(col_auc, axis=0)
        centered = col_auc - med

        def ensemble_profile(members: list[int]) -> np.ndarray:
            joint = L[members][:, train].mean(axis=0)
            return per_col_auc(t0[train], joint) - med

        def solo_pick(seated: list[int]) -> int:
            cands = [i for i in rest if i not in seated]
            return max(cands, key=lambda i: solo[i])

        def div_pick(seated: list[int]) -> int:
            ep = ensemble_profile(seated)
            cands = [i for i in rest if i not in seated]
            return min(cands,
                       key=lambda i: (spearmanr(centered[i], ep).statistic, -solo[i]))

        return solo_pick, div_pick

    def score(members: list[int], test: np.ndarray) -> float:
        return float(per_col_auc(t0[test], L[members][:, test].mean(axis=0)).mean())

    d7, s7, v7, d8, s8, v8 = [], [], [], [], [], []
    pick_solo7, pick_div7, pick_solo8, pick_div8 = Counter(), Counter(), Counter(), Counter()
    for sp in range(N_SPLITS):
        rng = np.random.default_rng(SEED_BASE + sp)
        perm = rng.permutation(909)
        for train, test in ((perm[:455], perm[455:]), (perm[455:], perm[:455])):
            solo_pick, div_pick = rules(train)
            p_s, p_d = solo_pick([]), div_pick(six)
            pick_solo7[names[p_s]] += 1
            pick_div7[names[p_d]] += 1
            s7.append(score(six + [p_s], test))
            v7.append(score(six + [p_d], test))
            d7.append(v7[-1] - s7[-1])
            # k=8: seventh = solo pick; eighth by each rule
            p_s8, p_d8 = solo_pick([p_s]), div_pick(six + [p_s])
            pick_solo8[names[p_s8]] += 1
            pick_div8[names[p_d8]] += 1
            s8.append(score(six + [p_s, p_s8], test))
            v8.append(score(six + [p_s, p_d8], test))
            d8.append(v8[-1] - s8[-1])

    def report(k, d, s, v, ps, pd):
        d, s, v = np.array(d), np.array(s), np.array(v)
        print(f"\n=== k={k} (40 readings) ===")
        print(f"  SOLO-pick seven     {s.mean():.4f}±{s.std():.4f}")
        print(f"  DIVERSITY-pick seven {v.mean():.4f}±{v.std():.4f}")
        print(f"  delta diversity-solo {d.mean():+.4f}±{d.std():.4f} | "
              f"win {(d > 0).sum()}/40, loss {(d < 0).sum()}/40, tie {(d == 0).sum()}")
        print(f"  solo rule picked: {ps.most_common(6)}")
        print(f"  diversity rule picked: {pd.most_common(6)}")

    report(7, d7, s7, v7, pick_solo7, pick_div7)
    report(8, d8, s8, v8, pick_solo8, pick_div8)

    # in-sample reference: same rules on the full 909
    solo_pick, div_pick = rules(np.arange(909))
    p_s, p_d = solo_pick([]), div_pick(six)
    full = np.arange(909)
    print(f"\nfull-909 in-sample reference: solo rule picks {names[p_s]} "
          f"-> seven {score(six + [p_s], full):.4f}; diversity rule picks "
          f"{names[p_d]} -> seven {score(six + [p_d], full):.4f}; "
          f"six alone {score(six, full):.4f}")


if __name__ == "__main__":
    main()
