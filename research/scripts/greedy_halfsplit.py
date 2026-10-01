"""Does greedy seat selection transfer? Half-split test (M381), the same
protocol that killed per-column routing.

Population: PACKAGED (comp/members.roster) HONEST members — npz with exactly
the canonical holdout909 studies, binarised truth identical cell-for-cell to
the reference, npz epoch >= 0. Ensemble: LogitAverage (mean of logits of
clipped probabilities, clip [1e-6, 1-1e-6]); score = macro AUC over 12 columns
vs midpoint-binarised v2 truth.

Protocol: 20 seeded half-splits (numpy default_rng seed 2026+split; 455/454).
Greedy up to 8 seats on the train half; no-search control = top-k solo. Score
both on the test half at k=4,6,8; symmetric swap -> 40 readings per k.
Secondary: coverage-greedy (selection criterion = mean(MCL, Fracture AUC))
scored on macro AUC, same frame.

Run: CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe research/scripts/greedy_halfsplit.py
"""

import glob
import os
from collections import Counter

import numpy as np
from scipy.stats import rankdata

ROOT = "D:/data/volumetric/mri/models/rsna_knee_abnormality"
REF = os.path.join(ROOT, "band_span_s2026.holdout.npz")
ROSTER = "comp/members.roster"
N_SPLITS = 20
SEED_BASE = 2026
KS = (4, 6, 8)
MAX_SEATS = 8
CLIP = 1e-6
MCL, FRACTURE = 1, 11  # column indices


def _aucs(y: np.ndarray, s: np.ndarray, cols: tuple[int, ...] | None = None) -> list[float]:
    """y (n, 12) 0/1 truth; s (n, 12) scores. Mann-Whitney AUC per column."""
    ranks = rankdata(s, axis=0)
    aucs = []
    for j in (cols if cols is not None else range(s.shape[1])):
        pos = y[:, j] == 1
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        aucs.append((ranks[pos, j].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return aucs


def fast_macro_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(np.mean(_aucs(y, s)))


def fast_pair_auc(y: np.ndarray, s: np.ndarray, cols: tuple[int, ...]) -> float:
    return float(np.mean(_aucs(y, s, cols)))


def greedy(logits: np.ndarray, y: np.ndarray, idx: np.ndarray,
           criterion: str) -> list[int]:
    """logits (m, 909, 12). Greedy up to MAX_SEATS on study subset idx."""
    remaining = set(range(logits.shape[0]))
    chosen: list[int] = []
    joint = np.zeros((909, 12))
    for _ in range(MAX_SEATS):
        best, best_s = None, -1.0
        for c in remaining:
            s = (joint[idx] + logits[c][idx]) / (len(chosen) + 1)
            v = (fast_macro_auc(y[idx], s) if criterion == "macro"
                 else fast_pair_auc(y[idx], s, (MCL, FRACTURE)))
            if v > best_s:
                best, best_s = c, v
        assert best is not None
        chosen.append(best)
        joint += logits[best]
        remaining.discard(best)
    return chosen


def joint_score(logits: np.ndarray, picks: list[int], y: np.ndarray,
                idx: np.ndarray) -> float:
    return fast_macro_auc(y[idx], logits[picks][:, idx].mean(axis=0))


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
    n_no_npz, n_studies, n_truth, n_epoch = 0, 0, 0, 0
    for name in roster:
        p = os.path.join(ROOT, name + ".holdout.npz")
        if not os.path.exists(p):
            n_no_npz += 1
            continue
        z = np.load(p)
        st = [str(s) for s in z["studies"].tolist()]
        if set(st) != set(ref_studies) or len(st) != 909:
            n_studies += 1
            continue
        order = np.array([pos[s] for s in st])
        pr = np.empty((909, 12)); pr[order] = z["predictions"]
        tr = np.empty((909, 12)); tr[order] = z["truth"]
        if not np.array_equal((tr >= 0.5).astype(int), t0):
            n_truth += 1
            continue
        if int(np.asarray(z["epoch"])) < 0:
            n_epoch += 1
            continue
        names.append(name)
        preds.append(pr)

    P = np.asarray(preds)                       # (m, 909, 12)
    L = np.log(np.clip(P, CLIP, 1 - CLIP) / (1 - np.clip(P, CLIP, 1 - CLIP)))
    m = len(names)
    from sklearn.metrics import roc_auc_score
    assert abs(fast_macro_auc(t0, L[0])
               - float(np.mean([roc_auc_score(t0[:, j], L[0][:, j])
                                for j in range(12)]))) < 1e-12
    print(f"POPULATION: roster {len(roster)} names -> packaged+honest {m} "
          f"(excluded: {n_no_npz} no npz [all ff/cf/df dark], {n_studies} wrong "
          f"studies, {n_truth} truth mismatch, {n_epoch} epoch<0 probe)")
    print("  NOTE: main's letter said 133; this count is 117 with the breakdown "
          "above — definition: roster names with a same-stem holdout npz, exact "
          "909-study set, binarised truth cell-identical, npz epoch >= 0.")
    print(f"CUT: LogitAverage (clip {CLIP}), macro AUC vs midpoint-binarised v2 "
          f"truth @holdout909; {N_SPLITS} splits x 2 directions = "
          f"{2 * N_SPLITS} readings per k")

    # full-909 reference on THIS population and cut
    solo_full = np.array([fast_macro_auc(t0, L[i]) for i in range(m)])
    top_full = np.argsort(-solo_full)
    greedy_full = greedy(L, t0, np.arange(909), "macro")
    print("\nfull-909 reference (this population/cut):")
    for k in (1, 2, 3, 4, 5, 6, 7, 8):
        print(f"  greedy@{k} {joint_score(L, greedy_full[:k], t0, np.arange(909)):.4f}"
              + (f"  | top-{k} solo {joint_score(L, list(top_full[:k]), t0, np.arange(909)):.4f}"
                 if k in KS else ""))

    results = {k: {"greedy": [], "topk": [], "cov": []} for k in KS}
    picks6, picks8 = [], []
    for s in range(N_SPLITS):
        rng = np.random.default_rng(SEED_BASE + s)
        perm = rng.permutation(909)
        halves = [(perm[:455], perm[455:]), (perm[455:], perm[:455])]
        for train, test in halves:
            solo = np.array([fast_macro_auc(t0[train], L[i][train]) for i in range(m)])
            top_order = np.argsort(-solo)
            g_macro = greedy(L, t0, train, "macro")
            g_cov = greedy(L, t0, train, "coverage")
            picks6.append(tuple(names[i] for i in g_macro[:6]))
            picks8.append(tuple(names[i] for i in g_macro[:8]))
            for k in KS:
                results[k]["greedy"].append(joint_score(L, g_macro[:k], t0, test))
                results[k]["topk"].append(joint_score(L, list(top_order[:k]), t0, test))
                results[k]["cov"].append(joint_score(L, g_cov[:k], t0, test))

    print("\n=== PRIMARY: greedy vs top-k solo, out-of-half (40 readings each) ===")
    for k in KS:
        g, t, c = (np.array(results[k][x]) for x in ("greedy", "topk", "cov"))
        d = g - t
        print(f"  k={k}: greedy {g.mean():.4f}±{g.std():.4f} | "
              f"top-k {t.mean():.4f}±{t.std():.4f} | "
              f"delta {d.mean():+.4f}±{d.std():.4f} "
              f"(win {(d > 0).sum()}/40, loss {(d < 0).sum()}/40)")

    print("\n=== selection stability across the 40 greedy runs ===")
    for kk, picks in ((6, picks6), (8, picks8)):
        cnt = Counter(nm for p in picks for nm in p)
        jac = []
        for i in range(len(picks)):
            for j in range(i + 1, len(picks)):
                a, b = set(picks[i]), set(picks[j])
                jac.append(len(a & b) / len(a | b))
        print(f"  k={kk}: mean pairwise Jaccard {np.mean(jac):.3f} of {kk}-sets; "
              f"most picked: {cnt.most_common(8)}")

    print("\n=== SECONDARY: coverage-greedy (MCL+Fracture criterion) vs macro-greedy, "
          "scored on macro AUC ===")
    for k in KS:
        g, c = np.array(results[k]["greedy"]), np.array(results[k]["cov"])
        d = c - g
        print(f"  k={k}: coverage {c.mean():.4f}±{c.std():.4f} vs macro "
              f"{g.mean():.4f}±{g.std():.4f} | delta {d.mean():+.4f}±{d.std():.4f} "
              f"(win {(d > 0).sum()}/40)")


if __name__ == "__main__":
    main()
