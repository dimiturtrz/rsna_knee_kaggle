"""Draw the SENSITIVITY-side blind pack the label axis has never had (knee-wko44).

Every blind ruler this project owns prices FALSE-POSITIVE SUPPRESSION and nothing else. `blind_effmcl40`
and `blind_tear60` together hold 7 positives in 100 cells, so a v5reader that suppressed real high-grade
findings alongside the rubric's negatives would read exactly the same on them
(`interpretations/kneemri/2026-09-15_labeler_blind_ruler.md`, Limits 2). That is the whole open question
on the axis: the census shows v5reader HALVES v2's positive rate per column - Effusion 0.605 -> 0.259,
MCL 0.174 -> 0.050, Medial OA 0.367 -> 0.179, PF OA 0.452 -> 0.247, ACL 0.278 -> 0.173 - and the blind
packs cannot say whether what it removed was the host rubric's negatives or the grader's findings.

THE DENOMINATOR IS THE DROPPED CELLS, and that is the point. Sampling the corpus would spend almost every
cell on agreement. This pack draws only where v2 >= 0.5 and v5reader < 0.5, so each graded cell is one
finding v5reader took away, and the share the adjudicator grades POSITIVE is the sensitivity cost of the
swap directly. Multiplied by the per-column drop count it is the absolute number of findings lost.

WHY A CONTROL STRATUM EXISTS. The adjudicator knows the selection rule, so on a pack of dropped cells
alone every cell's table verdicts are known before reading a word - v2 yes, v5reader no - and the reading
is no longer blind to the thing under test. Mixing in cells where BOTH tables call positive makes a
dropped cell indistinguishable from a kept one in the shuffled pack, and the control's own positive rate
is the adjudicator's calibration against a stratum whose answer is not in dispute.

    python -m research.scripts.build_blind_sensitivity

Writes the pack, the sealed key and a manifest under the audit400 root. Grade `adjudication_pack.csv`
into `answers_blind.csv` BEFORE opening the key or either label table, then score with `score_blind.py`
from `blind_tear60`.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("D:/data/volumetric/mri/processed/rsna_knee_abnormality")
AUDIT = ROOT / "audit400"
OUT = AUDIT / "blind_sensitivity90"
REPORTS = Path("D:/data/volumetric/mri/raw/rsna_knee_abnormality/train.csv")
GOLD = Path("D:/data/volumetric/mri/raw/knee_label_tables/gxpixel_knee-consensus-labels/gold_58.csv")

V2 = ROOT / "labels/report_labels_v2.csv"
V5 = ROOT / "labels/report_labels_v5reader.csv"

# The five columns the census names, in order of how much of the positive mass v5reader removes. They are
# also the columns where the host's CRITERIA carry a severity or grade threshold, which is where a reader
# that applies one and a lexicon that does not can disagree at scale.
COLUMNS = ["Effusion", "MCL", "Medial OA", "PF OA", "ACL"]
DROPPED_PER_COLUMN = 12
CONTROL_PER_COLUMN = 6
SEED = 20260917
CUT = 0.5


def existing_uids() -> set[str]:
    """Every study any earlier audit400 pack or gold-58 already spent, so no cell is graded twice.

    A study re-drawn here would be one the adjudicator may have read before, and a remembered report is
    not a blind reading of it.
    """
    spent = set(pd.read_csv(GOLD)["StudyInstanceUID"])
    for path in AUDIT.rglob("adjudication_key.csv"):
        key = pd.read_csv(path)
        # `holdout469` spells the same column `study_uid`. Reading only `uid` would silently skip that
        # pack's 469 studies and let this draw re-ask a report the adjudicator has already answered.
        spent |= set(key["uid" if "uid" in key.columns else "study_uid"])
    return spent


def draw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The pack and its key: dropped cells against a control of cells both tables keep."""
    v2 = pd.read_csv(V2).set_index("StudyInstanceUID")[COLUMNS]
    v5 = pd.read_csv(V5).set_index("StudyInstanceUID")[COLUMNS]
    uids = sorted((set(v2.index) & set(v5.index)) - existing_uids())
    v2, v5 = v2.loc[uids], v5.loc[uids]

    rng = np.random.default_rng(SEED)
    rows = []
    for column in COLUMNS:
        kept = v2[column] >= CUT
        dropped = kept & (v5[column] < CUT)
        control = kept & (v5[column] >= CUT)
        for stratum, mask, size in (("dropped", dropped, DROPPED_PER_COLUMN), ("control", control, CONTROL_PER_COLUMN)):
            pool = np.asarray(mask.index[mask])
            taken = rng.choice(pool, size=min(size, len(pool)), replace=False)
            rows += [{"stratum": stratum, "column": column, "uid": uid} for uid in taken]

    key = pd.DataFrame(rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    key["row_id"] = [f"sens{index:03d}" for index in range(1, len(key) + 1)]
    reports = pd.read_csv(REPORTS).set_index("StudyInstanceUID")["Report"]
    pack = pd.DataFrame(
        {"row_id": key["row_id"], "finding": key["column"], "report": reports.loc[key["uid"]].to_numpy()}
    )
    return pack, key


def main() -> None:
    """Write the pack, the sealed key and the manifest that records how they were drawn."""
    pack, key = draw()
    OUT.mkdir(parents=True, exist_ok=True)
    pack.to_csv(OUT / "adjudication_pack.csv", index=False)
    key.to_csv(OUT / "adjudication_key.csv", index=False)
    digest = {name: hashlib.sha256((OUT / f"{name}.csv").read_bytes()).hexdigest()[:16] for name in
              ("adjudication_pack", "adjudication_key")}
    manifest = {
        "bead": "knee-wko44",
        "seed": SEED,
        "question": "of the positives v5reader removes from v2, what share are real findings under the host rubric",
        "strata": key["stratum"].value_counts().to_dict(),
        "per_column": key.groupby(["column", "stratum"]).size().unstack(fill_value=0).to_dict(),
        "rubric": "host topic 733343 (core/findings.py CRITERIA)",
        "tables": {"v2": str(V2), "v5reader": str(V5)},
        "excluded": "gold_58 and every study in an existing audit400 adjudication_key",
        "builder": str(Path(__file__).resolve()),
        **{f"{name}_sha256": value for name, value in digest.items()},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(pack)} cells -> {OUT}")
    print(key.groupby(["column", "stratum"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
