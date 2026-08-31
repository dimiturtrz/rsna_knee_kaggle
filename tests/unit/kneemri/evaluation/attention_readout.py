from typing import cast

import numpy as np

from core.data.corpus.slots import SLOTS
from core.findings import TARGETS
from kneemri.evaluation.attention_readout import AttentionReadout, AttentionReadoutReport, SlotMass

SEED = 2026
STUDIES = 8
CARRIER = 0
"""The slot the fixture weights concentrate on, so `dearest` has a knowable right answer."""


def _weights(peak: int = CARRIER) -> np.ndarray:
    """Attention that is mostly on one slot, so `dearest` and `offtable` have a checkable answer."""
    weights = np.full((STUDIES, len(TARGETS), len(SLOTS)), 0.1, dtype=np.float32)
    weights[:, :, peak] = 1.0 - 0.1 * (len(SLOTS) - 1)
    return weights


def _mask(absent: int | None = None) -> np.ndarray:
    mask = np.ones((STUDIES, len(SLOTS)), dtype=np.float32)
    if absent is not None:
        mask[:, absent] = 0.0
    return mask


def _rows() -> list[SlotMass]:
    return AttentionReadout().rows(_weights(), _mask())


def test_affinity() -> None:
    affinity = AttentionReadout().affinity(TARGETS[0])
    assert set(affinity) == {slot.name for slot in SLOTS}
    assert max(affinity.values()) > 0.0


def test_affinity_of_unknown_finding_is_all_zero() -> None:
    assert set(AttentionReadout().affinity("not a finding").values()) == {0.0}


def test_presence() -> None:
    presence = AttentionReadout().presence(_mask(absent=1))
    assert presence[SLOTS[1].name] == 0.0
    assert presence[SLOTS[0].name] == 1.0


def test_rows() -> None:
    rows = _rows()
    assert len(rows) == len(TARGETS)
    assert [row.finding for row in rows] == list(TARGETS)
    assert all(row.dearest() == SLOTS[CARRIER].name for row in rows)


def test_rows_condition_mass_on_presence() -> None:
    """A slot no study contains must read nan, never a zero that looks like an unattended slot."""
    absent = AttentionReadout().rows(_weights(), _mask(absent=1))[0].mass[SLOTS[1].name]
    assert np.isnan(absent)


def test_record() -> None:
    record = AttentionReadout().record(_weights(), _mask())
    assert record["slots"] == [slot.name for slot in SLOTS]
    assert record["studies"] == STUDIES
    findings = cast(list[dict[str, object]], record["findings"])
    assert len(findings) == len(TARGETS)
    assert findings[0]["dearest"] == SLOTS[CARRIER].name


def test_dearest() -> None:
    assert _rows()[0].dearest() == SLOTS[CARRIER].name


def test_expected() -> None:
    expected = _rows()[0].expected()
    assert expected
    assert set(expected) <= {slot.name for slot in SLOTS}


def test_line() -> None:
    line = _rows()[0].line()
    assert TARGETS[0] in line
    assert "off-table" in line


def test_lines() -> None:
    rows = _rows()
    lines = AttentionReadoutReport(rows, AttentionReadout().presence(_mask()), prior=False).lines()
    assert any("WITHOUT the anatomical prior" in line for line in lines)
    assert any("degenerate-attention" in line for line in lines)


def test_lines_name_the_prior_when_the_member_was_fitted_with_it() -> None:
    """Agreement is not a measurement for a member that was handed the table — the report must say so."""
    lines = AttentionReadoutReport(_rows(), AttentionReadout().presence(_mask()), prior=True).lines()
    assert any("FITTED WITH THE ANATOMICAL PRIOR" in line for line in lines)
