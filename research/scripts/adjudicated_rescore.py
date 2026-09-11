"""Rescore an arm's holdout npz on the 469 hand-adjudicated cells, split tear/non-tear/Contusion.

    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe research/scripts/adjudicated_rescore.py <stem> [<stem2> ...]

Truth is the blind verdict, never any label table. Every report prints the CONST-0/CONST-1
trivial baselines first (M362: a ruler whose trivial baseline is unstated is how we got here),
and the ranking column is AUC-vs-verdict — accuracy at 0.5 is reported beside it as calibration,
never alone (M363). Sensitivity carries a Wilson interval because the affirmed n is 32.
"""
import glob
import math
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

COLS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
        "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
TEAR = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"]
ROOT = "D:/data/volumetric/mri/processed/rsna_knee_abnormality/audit400/holdout469"
MODELS = "D:/models/rsna_knee_abnormality"


def verdicts() -> pd.DataFrame:
    key = pd.read_csv(f"{ROOT}/adjudication_key.csv")
    frames = []
    for f in sorted(glob.glob(f"{ROOT}/answers_*of10.csv")):
        a = pd.read_csv(f)
        col = "verdict" if "verdict" in a.columns else "answer"
        frames.append(a[["row_id", col]].rename(columns={col: "verdict"}))
    return key.merge(pd.concat(frames), on="row_id")


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def _cells(stem: str, df: pd.DataFrame) -> np.ndarray:
    d = np.load(f"{MODELS}/{stem}.holdout.npz")
    studies = d["studies"].astype(str)
    sidx = {u: i for i, u in enumerate(studies)}
    cix = {c: j for j, c in enumerate(COLS)}
    return np.array([float(d["predictions"][sidx[r.study_uid], cix[r.column]]) for r in df.itertuples()])


def _row(p: np.ndarray, t: np.ndarray) -> str:
    c = (p >= 0.5).astype(int)
    pos = t == 1
    sens = (c[pos] == 1).mean() if pos.sum() else float("nan")
    spec = (c[~pos] == 0).mean()
    acc = (c == t).mean()
    try:
        auc = roc_auc_score(t, p)
    except ValueError:
        auc = float("nan")
    lo, hi = _wilson(int((c[pos] == 1).sum()), int(pos.sum()))
    return (f"acc {acc:.3f}  AUC {auc:.4f}  sens {sens:.3f} [{lo:.2f},{hi:.2f}]  "
            f"spec {spec:.3f}  bal {(sens + spec) / 2 if pos.sum() else float('nan'):.3f}")


def score(stem: str, df: pd.DataFrame, p: np.ndarray | None = None) -> None:
    p = _cells(stem, df) if p is None else p
    t = df.verdict.to_numpy()
    print(f"== {stem} ==")
    for name, m in (("all-469", np.ones(len(df), bool)),
                    ("tear-277", df.column.isin(TEAR).to_numpy()),
                    ("nontear-192", (~df.column.isin(TEAR)).to_numpy()),
                    ("contusion-120", (df.column == "Contusion").to_numpy())):
        print(f"  {name:14s} n={int(m.sum()):4d} pos={int(t[m].sum()):3d}  {_row(p[m], t[m])}")


def score_all(stems: list[str], df: pd.DataFrame) -> None:
    t = df.verdict.to_numpy()
    for name, m in (("all-469", np.ones(len(df), bool)),
                    ("nontear-192", (~df.column.isin(TEAR)).to_numpy()),
                    ("contusion-120", (df.column == "Contusion").to_numpy())):
        print(f"\n=== {name} (n={int(m.sum())}, pos={int(t[m].sum())}) ===")
        for stem in stems:
            print(f"  {stem:26s} {_row(_cells(stem, df)[m], t[m])}")
        print(f"  {'CONST-0':26s} {_row(np.zeros(int(m.sum())), t[m])}")
        print(f"  {'CONST-1':26s} {_row(np.ones(int(m.sum())), t[m])}")


def mcnemar(a: str, b: str, df: pd.DataFrame, mask_name: str, mask: np.ndarray) -> None:
    t = df.verdict.to_numpy()
    ca = ((_cells(a, df) >= 0.5).astype(int) == t)[mask]
    cb = ((_cells(b, df) >= 0.5).astype(int) == t)[mask]
    print(f"  McNemar {mask_name}: {a}-only {int((ca & ~cb).sum())}, {b}-only {int((~ca & cb).sum())}")


if __name__ == "__main__":
    df = verdicts()
    stems = sys.argv[1:]
    if len(stems) > 2:
        score_all(stems, df)
    else:
        for stem in stems:
            score(stem, df)
    if len(stems) == 2:
        for name, m in (("tear", df.column.isin(TEAR).to_numpy()),
                        ("contusion", (df.column == "Contusion").to_numpy())):
            mcnemar(stems[0], stems[1], df, name, m)
