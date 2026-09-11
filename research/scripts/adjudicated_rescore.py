"""Rescore an arm's holdout npz on the 469 hand-adjudicated cells, split tear/non-tear/Contusion.

    CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe research/scripts/adjudicated_rescore.py <stem> [<stem2> ...]

Truth is the blind verdict, never any label table. The remainder split (non-adjudicated cells,
scored against the npz's own truth) is the control for target-distribution artefacts.
"""
import glob
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


def score(stem: str, df: pd.DataFrame) -> None:
    d = np.load(f"{MODELS}/{stem}.holdout.npz")
    studies = d["studies"].astype(str)
    sidx = {u: i for i, u in enumerate(studies)}
    cix = {c: j for j, c in enumerate(COLS)}
    preds, truth, v2truth = [], [], []
    for r in df.itertuples():
        i, j = sidx[r.study_uid], cix[r.column]
        preds.append(float(d["predictions"][i, j]))
        truth.append(int(r.verdict))
        v2truth.append(float(d["truth"][i, j]))
    df = df.assign(pred=preds, truth=truth, v2truth=v2truth)
    df["pcall"] = (df.pred >= 0.5).astype(int)
    splits = {
        "tear-277": df.column.isin(TEAR),
        "nontear-192": ~df.column.isin(TEAR),
        "contusion-120": df.column == "Contusion",
        "all-469": pd.Series(True, index=df.index),
    }
    print(f"== {stem} ==")
    for name, m in splits.items():
        sub = df[m]
        acc = (sub.pcall == sub.truth).mean()
        auc = roc_auc_score(sub.truth, sub.pred) if sub.truth.nunique() == 2 else float("nan")
        print(f"  {name:14s} n={len(sub):4d}  acc {acc:.3f}  AUC-vs-verdict {auc:.4f}")
    sub = df[df.column.isin(TEAR)]
    print(f"  tear per-col acc:", {c: round(float((sub[sub.column == c].pcall == sub[sub.column == c].truth).mean()), 3) for c in TEAR})


def mcnemar(a: str, b: str, df: pd.DataFrame, mask_name: str, mask) -> None:
    calls = {}
    for stem in (a, b):
        d = np.load(f"{MODELS}/{stem}.holdout.npz")
        studies = d["studies"].astype(str)
        sidx = {u: i for u in enumerate(studies) for u in [u[1]]} if False else {u: i for i, u in enumerate(studies)}
        cix = {c: j for j, c in enumerate(COLS)}
        calls[stem] = [(float(d["predictions"][sidx[r.study_uid], cix[r.column]]) >= 0.5) == bool(r.verdict)
                       for r in df.itertuples()]
    ca = np.array(calls[a])[mask]; cb = np.array(calls[b])[mask]
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    print(f"  McNemar {mask_name}: {a}-only {n01}, {b}-only {n10}")


if __name__ == "__main__":
    df = verdicts()
    for stem in sys.argv[1:]:
        score(stem, df)
    if len(sys.argv) == 3:
        for name, m in (("tear", df.column.isin(TEAR)), ("contusion", df.column == "Contusion")):
            mcnemar(sys.argv[1], sys.argv[2], df, name, m.to_numpy())
