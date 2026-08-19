from pathlib import Path

from kneemri.evaluation.tie_divergence import SlotDivergence, TieDivergence, TieDivergenceReport

from ..conftest import write_series

STUDIES = 200
DIFFERENT = 0.5
"""A distance no two renderings of the same acquisition could produce, used to assert the classes apart."""


def test_of_study(tmp_path: Path):
    """Two tied series drawn from different pixels report a divergence; identical ones report none.

    These are the two equivalence classes the whole hypothesis turns on. A contested slot whose
    candidates are the same picture costs nothing whichever routing rule picks it, and one whose
    candidates differ is a different image under the head — the 0.011 leaderboard gap is only
    explicable by the second.
    """
    study = tmp_path / "study"
    write_series(study / "a", slices=12, seed=1)
    write_series(study / "b", slices=12, seed=2)
    planes = {"a": "Sagittal", "b": "Sagittal"}

    divergences = TieDivergence().of_study(study, planes)

    assert len(divergences) == 1, "one rival against one winner"
    assert divergences[0].slot == "SAG_FLUID_FS"
    assert not divergences[0].identical, "different pixels must not read as the same image"


def test_of_study_is_empty_when_no_slot_is_contested(tmp_path: Path):
    """A slot with a single densest candidate has no rivalry to measure, whatever the list order."""
    study = tmp_path / "study"
    write_series(study / "thin", slices=8)
    write_series(study / "thick", slices=20)

    assert TieDivergence().of_study(study, {"thin": "Sagittal", "thick": "Sagittal"}) == []


def test_identical():
    """The threshold is one quantisation step, because that is the resolution the pixels are stored at."""
    same = SlotDivergence("s", "SAG_FLUID_FS", "a", "b", distance=0.001)
    differing = SlotDivergence("s", "SAG_FLUID_FS", "a", "b", distance=0.02)

    assert same.identical, "below a quantisation step is the same image"
    assert not differing.identical


def test_lines():
    """The report states what each outcome settles, so a number cannot be quoted without its verdict."""
    lines = TieDivergenceReport(
        [SlotDivergence("s", "SAG_FLUID_FS", "a", "b", distance=DIFFERENT)],
        STUDIES,
    ).lines()
    text = "\n".join(lines)

    assert "1 contested-slot rivalries over 200 studies" in text
    assert "SAG_FLUID_FS: 1 rivalries" in text
    assert "NEAR ZERO kills the tie-break hypothesis" in text


def test_lines_say_so_when_nothing_decoded():
    """An empty census is reported as untestable, never as a near-zero result that kills the idea."""
    text = "\n".join(TieDivergenceReport([], STUDIES).lines())

    assert "cannot be tested" in text
