"""TASK 2 (M392 §4): on classb120's agreement-POSITIVE rows, does v2_soft track
the grade stated in the report? Grade extraction is multilingual and thin-n —
reported with the caveat, never a headline.

Run: CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe research/scripts/grade_soft_check.py
"""

import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

A = "D:/data/volumetric/mri/processed/rsna_knee_abnormality/audit400/classb120"

KEYWORDS = {
    "MCL": ["mcl", "medial collateral", "collaterale mediale", "lig. coll."],
    "Lateral OA": ["lateral", "oa", "osteoartr", "arthros", "degenerativ",
                   "cartilage", "hrskav", "zglob"],
    "Fracture": ["fractur", "fraktur", "cortical", "kontuz", "prelom"],
}
GRADE_RES = [
    re.compile(r"\bgrade\s*(?:of\s+)?(IV|I{1,3}|[1-4])\b", re.I),
    re.compile(r"\b(IV|I{1,3}|[1-4])\.?\s*stupnja\b", re.I),
    re.compile(r"\bstupanj\s*(IV|I{1,3}|[1-4])\b", re.I),
    re.compile(r"\b[Gg]rad(?:e?s?)?\s*(IV|I{1,3}|[1-4])\b"),
    re.compile(r"\b(IV|I{1,3}|[1-4])\.\s*stup\b", re.I),
]
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4}


def ordinal(tok: str) -> int:
    tok = tok.upper()
    return ROMAN[tok] if tok in ROMAN else int(tok)


def grade_mentions(text: str) -> list[tuple[int, int]]:
    out = []
    for rx in GRADE_RES:
        for m in rx.finditer(text):
            out.append((m.start(), ordinal(m.group(1))))
    return sorted(set(out))


def keyword_pos(text: str, col: str) -> list[int]:
    low = text.lower()
    return [p for kw in KEYWORDS[col] for p in [m.start() for m in re.finditer(re.escape(kw), low)]]


def main() -> None:
    key = pd.read_csv(f"{A}/adjudication_key.csv")
    pack = pd.read_csv(f"{A}/adjudication_pack.csv")
    df = key.merge(pack[["row_id", "report"]], on="row_id")
    pos = df[df.v2_soft >= 0.5]
    print(f"classb120 agreement-positive rows: {len(pos)} "
          f"(expect 26 = 5 MCL + 18 Lateral OA + 3 Fracture)")

    rows = []
    n_ungradeable = 0
    for r in pos.itertuples():
        text = str(r.report)
        mentions = grade_mentions(text)
        kps = keyword_pos(text, r.column)
        grade = None
        if mentions and kps:
            grade = min(mentions, key=lambda g: min(abs(g[0] - p) for p in kps))[1]
        elif len(mentions) == 1:
            grade = mentions[0][1]
        if grade is None:
            n_ungradeable += 1
        rows.append((r.row_id, r.column, r.v2_soft, grade,
                     len(mentions), len(kps)))
        print(f"  {r.row_id} {r.column:11s} v2_soft {r.v2_soft:.2f} "
              f"grade {grade if grade is not None else '-':>2} "
              f"({len(mentions)} grade mentions, {len(kps)} keyword hits)")

    graded = [(g, s) for _, _, s, g, _, _ in rows if g is not None]
    print(f"\ngradeable: {len(graded)}/{len(pos)} "
          f"(ungradeable {n_ungradeable}: no grade mention or ambiguous/multiple "
          f"with no keyword anchor)")
    if len(graded) >= 5:
        g = np.array([x[0] for x in graded], float)
        s = np.array([x[1] for x in graded], float)
        rho = spearmanr(g, s)
        print(f"Spearman(grade ordinal, v2_soft) = {rho.statistic:+.3f} "
              f"(p={rho.pvalue:.3f}, n={len(graded)})")
        print("CAVEAT: n<30, grade extraction by multilingual regex anchored on "
              "finding keywords — an instrument reading, not a headline. Grade "
              "ordinals are coarse (mostly 2/3), so ties dominate the "
              "correlation; treat the sign as indicative only.")
    by_col = {}
    for _, c, s, g, _, _ in rows:
        if g is not None:
            by_col.setdefault(c, []).append((g, s))
    for c, v in by_col.items():
        if len(v) >= 4:
            gg = np.array([x[0] for x in v], float)
            ss = np.array([x[1] for x in v], float)
            print(f"  {c}: n={len(v)} spearman {spearmanr(gg, ss).statistic:+.3f}")


if __name__ == "__main__":
    main()
