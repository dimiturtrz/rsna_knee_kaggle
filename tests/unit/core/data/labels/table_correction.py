"""Corrections applied to a label table: the transform, the backing gate, and the order they compose in."""

import pandas as pd

from core.data.labels.rubric_gap import MENISCUS_RULE
from core.data.labels.table_correction import (
    FRACTURE_CONTUSION,
    RUBRIC_SUBTHRESHOLD,
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


def _sources() -> LabelSources:
    """The table beside its reports, and an annotation that grades the subthreshold meniscus negative."""
    reports = pd.DataFrame({STUDY: ["s1", "s2"], "Report": [DEGENERATE, TEAR]})
    truth = reports.copy()
    for column in TARGETS:
        truth[column] = [0.0, 0.0]
    return LabelSources(_table(), reports, truth)


def test_partner_lift_rescaled():
    """The mapping lands on the named endpoints and keeps the order it was given."""
    out = PartnerLift("Fracture", "Contusion", 0.25).rescaled(pd.Series([0.0, 1.0, 4.0]), 0.1, 0.9)
    assert (out.min(), out.max()) == (0.1, 0.9)
    assert list(out) == sorted(out)


def test_partner_lift_applied():
    """The lift moves its own column onto the partner's ordering and leaves the range where it was.

    `s1` has the floor Fracture and the ceiling Contusion, so a lift by any positive weight must invert
    the two studies. The endpoints must not move: a rescale that widened the column would change what
    every downstream threshold means while claiming to be order-only.
    """
    sources = _sources()
    out = PartnerLift("Fracture", "Contusion", 0.25).applied(sources)
    assert out["Fracture"][0] > out["Fracture"][1], "the contusion study must overtake the fracture study"
    assert (out["Fracture"].min(), out["Fracture"].max()) == (FLOOR, FRACTURE_TOP)
    assert out["Contusion"].equals(sources.table["Contusion"]), "the partner is read, never written"


def test_rubric_pull_flagged():
    """Only the backed rule's contradicted cells are named, and only where the report is subthreshold."""
    assert RubricPull(0.15).flagged(_sources()) == [("s1", MEDIAL)]


def test_rubric_pull_applied():
    """The flagged cell is pulled toward its column's floor by exactly `1 - residual`, and nothing else moves."""
    sources = _sources()
    out = RubricPull(0.15).applied(sources).set_index(STUDY)
    assert out.loc["s1", MEDIAL] == round(FLOOR + 0.15 * (CEILING - FLOOR), 4)
    assert out.loc["s1", LATERAL] == CEILING, "the intact side is not flagged and must not move"
    assert out.loc["s2", MEDIAL] == FLOOR


def test_correction_sequence_applied():
    """Composition applies the rules in the order given, and each reads what the one before it wrote."""
    sources = _sources()
    out = CorrectionSequence((FRACTURE_CONTUSION, RUBRIC_SUBTHRESHOLD)).applied(sources).set_index(STUDY)
    assert out.loc["s1", MEDIAL] < CEILING, "the rubric pull ran"
    assert out.loc["s1", "Fracture"] > out.loc["s2", "Fracture"], "the fracture lift ran"
