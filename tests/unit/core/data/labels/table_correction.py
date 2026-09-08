"""Corrections applied to a label table: the transform, the backing gate, and the order they compose in."""

from pathlib import Path

import pandas as pd

from core.data.labels.rubric_gap import MENISCUS_RULE
from core.data.labels.table_correction import (
    FRACTURE_CONTUSION,
    RUBRIC_SUBTHRESHOLD,
    ColumnGraft,
    CorrectionSequence,
    LabelSources,
    PartnerLift,
    RubricPull,
)
from core.findings import STUDY, TARGETS

MEDIAL, LATERAL = MENISCUS_RULE.columns

DEGENERATE = (
    "There is intrasubstance degeneration of the medial meniscus without a definite tear. "
    "The lateral meniscus is intact and there is no fracture in the knee."
)
TEAR = (
    "There is an oblique tear of the medial meniscus reaching the inferior articular surface. "
    "The lateral meniscus appears intact and there is no joint effusion in the knee."
)

FLOOR, CEILING = 0.08, 0.94
FRACTURE_TOP = 0.20
"""Narrow enough that a quarter of a ceiling-valued Contusion overtakes it, which is the point of the lift."""


def _table() -> pd.DataFrame:
    """Two studies over the full target set: the first names the subthreshold wording, the second a tear."""
    frame = pd.DataFrame({STUDY: ["s1", "s2"]})
    for column in TARGETS:
        frame[column] = [CEILING, FLOOR]
    frame["Contusion"] = [CEILING, FLOOR]
    frame["Fracture"] = [FLOOR, FRACTURE_TOP]
    return frame


def _sources(labels: Path) -> LabelSources:
    """The table beside its reports, and an annotation that grades the subthreshold meniscus negative."""
    reports = pd.DataFrame({STUDY: ["s1", "s2"], "Report": [DEGENERATE, TEAR]})
    truth = reports.copy()
    for column in TARGETS:
        truth[column] = [0.0, 0.0]
    return LabelSources(_table(), reports, truth, labels)


def _cell(frame: pd.DataFrame, study: str, column: str) -> float:
    """One value as a number — a frame `.loc` scalar is typed as ANY scalar pandas can hold, timestamps too."""
    return float(frame[column].astype(float).to_dict()[study])


def test_partner_lift_rescaled():
    """The mapping lands on the named endpoints and keeps the order it was given."""
    out = PartnerLift("Fracture", "Contusion", 0.25).rescaled(pd.Series([0.0, 1.0, 4.0]), 0.1, 0.9)
    assert (out.min(), out.max()) == (0.1, 0.9)
    assert list(out) == sorted(out)


def test_partner_lift_applied(tmp_path: Path):
    """The lift moves its own column onto the partner's ordering and leaves the range where it was.

    `s1` has the floor Fracture and the ceiling Contusion, so a lift by any positive weight must invert
    the two studies. The endpoints must not move: a rescale that widened the column would change what
    every downstream threshold means while claiming to be order-only.
    """
    sources = _sources(tmp_path)
    out = PartnerLift("Fracture", "Contusion", 0.25).applied(sources)
    assert out["Fracture"][0] > out["Fracture"][1], "the contusion study must overtake the fracture study"
    assert (out["Fracture"].min(), out["Fracture"].max()) == (FLOOR, FRACTURE_TOP)
    assert out["Contusion"].equals(sources.table["Contusion"]), "the partner is read, never written"


def test_rubric_pull_flagged(tmp_path: Path):
    """Only the backed rule's contradicted cells are named, and only where the report is subthreshold."""
    assert RubricPull(0.15).flagged(_sources(tmp_path)) == [("s1", MEDIAL)]


def test_rubric_pull_applied(tmp_path: Path):
    """The flagged cell is pulled toward its column's floor by exactly `1 - residual`, and nothing else moves."""
    sources = _sources(tmp_path)
    out = RubricPull(0.15).applied(sources).set_index(STUDY)
    assert out.loc["s1", MEDIAL] == round(FLOOR + 0.15 * (CEILING - FLOOR), 4)
    assert out.loc["s1", LATERAL] == CEILING, "the intact side is not flagged and must not move"
    assert out.loc["s2", MEDIAL] == FLOOR


def test_correction_sequence_applied(tmp_path: Path):
    """Composition applies the rules in the order given, and each reads what the one before it wrote."""
    sources = _sources(tmp_path)
    out = CorrectionSequence((FRACTURE_CONTUSION, RUBRIC_SUBTHRESHOLD)).applied(sources).set_index(STUDY)
    assert _cell(out, "s1", MEDIAL) < CEILING, "the rubric pull ran"
    assert _cell(out, "s1", "Fracture") > _cell(out, "s2", "Fracture"), "the fracture lift ran"


def test_column_graft_applied(tmp_path: Path):
    """The re-read's verdict lands everywhere; its VALUE lands only where the re-read changed the level.

    `s1` is the cell the donor re-read downward — it takes the mapped level. `s2` is the cell both passes
    read the same way, and it must keep the base table's blended value rather than be flattened onto the
    reader's ladder, which is the whole reason the graft needs an origin at all.
    """
    passes = (("donor.csv", [0.28, 0.94], ["UNK", "YES"]), ("origin.csv", [0.94, 0.94], ["YES", "YES"]))
    for name, level, verdict in passes:
        frame = pd.DataFrame({STUDY: ["s1", "s2"], "Effusion": level})
        frame["Effusion__verdict"] = verdict
        frame["Effusion__conf"] = [0.5, 0.5]
        frame.to_csv(tmp_path / name, index=False)
    graft = ColumnGraft("Effusion", "donor.csv", "origin.csv", {0.28: 0.265, 0.94: 0.98})
    out = graft.applied(_sources(tmp_path)).set_index(STUDY)
    assert out.loc["s1", "Effusion"] == 0.265, "the re-read moved this level, so the mapped value lands"
    assert out.loc["s2", "Effusion"] == FLOOR, "both passes agree here, so the base value is carried"
    assert list(out["Effusion__verdict"]) == ["UNK", "YES"]
    assert out.loc["s1", "Medial OA"] == CEILING, "a graft touches one column"
