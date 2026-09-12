"""Fit the board itself (M375 as corrected by M376: board is quantized to 0.001).

1. OLS board ~ member presence + seat count, coefficients + SEs (84 ensemble
   rows with members_named).
2'. Residual SD of the fit vs the 0.001 quantization step.
3. Leave-one-entry-out CV R2 (hat-matrix shortcut).
4'. Minimum member effect the board can detect: quantization floor and the
   statistical floor (2*SE per member over the observed design).
5. Count of row pairs (all 120 numeric-board rows) differing by exactly 0.001.

Usage: python research/scripts/fit_board.py
"""
import json
import numpy as np


def load():
    rows = [json.loads(l) for l in open("comp/board.jsonl") if l.strip()]
    data = [r for r in rows if "note" not in r]
    numeric = [r for r in data if isinstance(r.get("board"), (int, float))]
    ens = [r for r in numeric if r.get("scope") == "ensemble"]
    unnamed = [r["id"] for r in ens if not r.get("members_named")]
    fit = [r for r in ens if r.get("members_named")]
    return data, numeric, ens, unnamed, fit


def main():
    data, numeric, ens, unnamed, fit = load()
    print("== population ==")
    print(f"rows total={len(data)} numeric_board={len(numeric)} ensemble={len(ens)}")
    print(f"dropped no members_named: n={len(unnamed)} ids={unnamed}")
    print(f"fit n={len(fit)}")

    # --- quantization facts (all numeric rows) ---
    vals = sorted({round(r["board"], 3) for r in numeric})
    gaps = [round(b - a, 6) for a, b in zip(vals, vals[1:])]
    print(f"\ndistinct board values={len(vals)} min adjacent gap={min(gaps)}")
    boards = np.array([r["board"] for r in numeric])
    step_pairs = 0
    for i in range(len(boards)):
        step_pairs += int((np.abs(boards[i + 1:] - boards[i]) - 0.001 < 1e-9).sum())
    print(f"pairs differing by exactly 0.001 (ask 5): {step_pairs} "
          f"of {len(boards)*(len(boards)-1)//2} pairs over {len(boards)} rows")

    # --- design ---
    members = sorted({m for r in fit for m in r["members_named"]})
    midx = {m: i for i, m in enumerate(members)}
    n, p = len(fit), len(members)
    X = np.zeros((n, p + 2))
    y = np.array([r["board"] for r in fit])
    for i, r in enumerate(fit):
        X[i, 0] = 1.0
        X[i, 1] = r["members"]
        for m in r["members_named"]:
            X[i, 2 + midx[m]] = 1.0
    names = ["(intercept)", "seat_count"] + members
    support = {m: sum(1 for r in fit if m in r["members_named"]) for m in members}
    print(f"\ndistinct members={p}  support: min={min(support.values())} "
          f"median={int(np.median(list(support.values())))} max={max(support.values())}")

    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(n - rank, 1)
    sigma2 = float(resid @ resid / dof)
    resid_sd = float(np.std(resid))
    se = np.sqrt(np.maximum(np.diag(np.linalg.pinv(X.T @ X)) * sigma2, 0))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot
    print(f"\n== OLS == rank={rank}/{p+2} n={n} R2={r2:.4f}")
    print(f"residual SD={resid_sd:.5f}  vs quantization step 0.001 "
          f"(half-step 0.0005)  ratio={resid_sd/0.001:.2f}")

    H = X @ np.linalg.pinv(X)
    loo = resid / np.maximum(1 - np.diag(H), 1e-9)
    loo_r2 = 1 - float(loo @ loo) / ss_tot
    print(f"leave-one-entry-out R2={loo_r2:.4f}")

    print("\nmember coefficients (support>=2), ranked:")
    out = [(m, support[m], beta[2 + midx[m]], se[2 + midx[m]])
           for m in members if support[m] >= 2]
    for m, s, b, s_ in sorted(out, key=lambda t: -t[2]):
        print(f"  {m:38s} n={s:2d} coef={b:+.5f} se={s_:.5f}")
    print(f"  seat_count coef={beta[1]:+.5f} se={se[1]:.5f}")

    print("\n== ask 4': minimum detectable member effect ==")
    print("quantization floor: a true seat effect <0.001 is rounded away; "
          "<0.0005 lands in the same bin always.")
    print("statistical floor (2*SE, this design), by support level:")
    by_sup = {}
    for m in members:
        by_sup.setdefault(support[m], []).append(se[2 + midx[m]])
    for s in sorted(by_sup):
        ses = by_sup[s]
        print(f"  support n={s:2d} ({len(ses)} members): median SE="
          f"{np.median(ses):.5f} -> min detectable ~{2*np.median(ses):.5f}")


if __name__ == "__main__":
    main()
