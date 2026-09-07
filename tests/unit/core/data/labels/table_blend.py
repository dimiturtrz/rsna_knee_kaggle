from pathlib import Path

import pandas as pd
import pytest

from core.data.labels.reader_agreement import MIDPOINT
from core.data.labels.table_blend import (
    LEAKED,
    MEAN,
    OVERRIDE,
    TIEBREAK,
    TIEBREAK_WEIGHT,
    LabelBlend,
)
from core.findings import STUDY

FINDING, OTHER = "Effusion", "ACL"
CONFIDENCE = f"{OTHER}__conf"


def _base() -> pd.DataFrame:
    """Three studies on our own coarse ladder, two of them TIED — the case the blend exists for."""
    return pd.DataFrame(
        {STUDY: ["a", "b", "c"], FINDING: [0.68, 0.68, 0.08], OTHER: [0.94, 0.08, 0.28], CONFIDENCE: [0.95, 0.95, 0.95]}
    )


def _foreign() -> pd.DataFrame:
    """A near-continuous reader that separates the base's tied pair and misses one finding entirely."""
    return pd.DataFrame({STUDY: ["a", "b", "c"], FINDING: [0.90, 0.30, 0.02]})


def test_findings():
    """Confidence columns are not targets and must not be blended as if they were."""
    blend = LabelBlend(_base(), [_foreign()])

    assert set(blend.findings()) == {FINDING, OTHER}, "TARGETS order leads, not the table's column order"
    assert CONFIDENCE not in blend.findings()


def test_readings():
    """A foreign reader contributes only where it carries the finding, never a column of NaN."""
    blend = LabelBlend(_base(), [_foreign()])

    assert len(blend.readings(FINDING)) == 2
    assert len(blend.readings(OTHER)) == 1, "the foreign table has no ACL column and must not appear"


def test_blended():
    """`MEAN` averages the readers that carry a finding and leaves the others at the base's value."""
    table = LabelBlend(_base(), [_foreign()]).blended(MEAN).set_index(STUDY)

    assert table.loc["a", FINDING] == pytest.approx((0.68 + 0.90) / 2)
    assert table.loc["c", OTHER] == pytest.approx(0.28), "no foreign reader for ACL, so the base stands"
    assert table.loc["a", CONFIDENCE] == pytest.approx(0.95), "confidence is carried, not averaged"


def test_blended_tiebreak_separates_a_tied_pair_without_reordering():
    """The whole mechanism: equal base cells become unequal, and the base's ordering still leads.

    `a` and `b` are both 0.68 in the base and AUC cannot separate them. After the tie-break they differ,
    and both still sit above `c`, whose base level is lower — a nudge large enough to reorder those
    would be measuring the foreign reader rather than breaking our ties.
    """
    table = LabelBlend(_base(), [_foreign()]).blended(TIEBREAK).set_index(STUDY)

    assert table.loc["a", FINDING] > table.loc["b", FINDING], "the tie is broken"
    assert table.loc["b", FINDING] > table.loc["c", FINDING], "and the base's ordering survives it"
    assert table.loc["a", FINDING] == pytest.approx(0.68 + TIEBREAK_WEIGHT * (0.90 - MIDPOINT))


def test_blended_tiebreak_is_a_no_op_without_a_foreign_reader():
    """With nothing to break the ties, the base is returned unchanged rather than nudged toward MIDPOINT."""
    table = LabelBlend(_base(), []).blended(TIEBREAK).set_index(STUDY)

    assert table.loc["a", FINDING] == pytest.approx(0.68)
    assert table.loc["b", FINDING] == pytest.approx(0.68)


def test_findings_narrowed_by_only():
    """`--only` scopes the retarget; a name that is not a target is a typo and must not pass silently."""
    assert LabelBlend(_base(), [_foreign()], [FINDING]).findings() == [FINDING]

    with pytest.raises(ValueError, match="not findings"):
        LabelBlend(_base(), [_foreign()], ["Effusiom"]).findings()


def test_blended_override_replaces_covered_cells_and_keeps_the_base_elsewhere():
    """The mode the two-column retarget rides on: no averaging where covered, base where not."""
    partial = pd.DataFrame({STUDY: ["a", "c"], FINDING: [0.90, 0.02]})

    table = LabelBlend(_base(), [partial], [FINDING]).blended(OVERRIDE).set_index(STUDY)

    assert table.loc["a", FINDING] == pytest.approx(0.90), "the foreign value stands alone, unaveraged"
    assert table.loc["b", FINDING] == pytest.approx(0.68), "uncovered study keeps the base"
    assert table.loc["b", OTHER] == pytest.approx(0.08), "a finding outside --only is never touched"


def test_coverage():
    """A positional join writes NaN everywhere and blends to the base in silence — this is the alarm."""
    stranger = pd.DataFrame({STUDY: ["x", "y", "z"], FINDING: [0.9, 0.3, 0.02]})

    lines = LabelBlend(_base(), [stranger]).coverage()

    assert "0/3 studies aligned" in " ".join(lines)
    assert "3/3 studies aligned" in " ".join(LabelBlend(_base(), [_foreign()]).coverage())


def test_read(tmp_path: Path):
    """A 1.0000 against the ruler is leakage, not quality, and is blocked at the filename."""
    leaky = tmp_path / f"{LEAKED[0]}.csv"
    leaky.write_text("StudyInstanceUID,Effusion\na,1\n", encoding="utf-8")
    honest = tmp_path / "some_reader.csv"
    honest.write_text("StudyInstanceUID,Effusion\na,0.9\n", encoding="utf-8")

    with pytest.raises(ValueError, match="gold 58"):
        LabelBlend.read(leaky)
    assert len(LabelBlend.read(honest)) == 1
