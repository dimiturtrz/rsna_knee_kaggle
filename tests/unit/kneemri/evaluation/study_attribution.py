from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.findings import TARGETS
from kneemri.evaluation.study_attribution import (
    CHANCE,
    AcquisitionCause,
    AttributionCorpus,
    LabelCause,
    SharedTail,
    StudyAttribution,
    SupervisionCause,
)
from kneemri.evaluation.study_errors import StudyDamage
from kneemri.runs.holdout_record import HoldoutRecord


def damaged(study: str, damage: float, over: float = 0.2, under: float = 0.1) -> StudyDamage:
    """One ranking row, only the fields the attribution reads carrying anything."""
    return StudyDamage(
        study=study,
        damage=damage,
        positives=3,
        worst_finding=TARGETS[0],
        worst_damage=damage,
        over=over,
        under=under,
    )


def ranking(order: list[str]) -> list[StudyDamage]:
    """A ranking worst-first, damage descending so the tail is the head of the list."""
    return [damaged(study, 1.0 - index / len(order)) for index, study in enumerate(order)]


def written(directory: Path, name: str, predictions: np.ndarray, truth: np.ndarray, studies: list[str]) -> None:
    """A holdout record beside a checkpoint that never existed, which is all `read` needs."""
    HoldoutRecord(directory / f"{name}.pt").write(studies, predictions, truth.astype(np.float32), epoch=6)


def test_split():
    """The cut is a tenth, worst first, and never empty however short the ranking is."""
    tail, body = StudyAttribution.split(ranking([f"s{index}" for index in range(50)]))
    assert len(tail) == 5
    assert len(body) == 45
    assert tail[0].study == "s0", "the ranking arrives worst-first and the tail is its head"

    thin_tail, thin_body = StudyAttribution.split(ranking(["only", "other"]))
    assert len(thin_tail) == 1, "a tail of zero studies would make every later reading undefined"
    assert len(thin_body) == 1


def test_shared():
    """A tail every member shares reads at one, a tail nobody shares reads at zero and is all private."""
    studies = [f"s{index}" for index in range(30)]
    mine = ranking(studies)
    agreeing = StudyAttribution().shared(mine, {"twin": ranking(studies)})
    assert agreeing.overlap == pytest.approx(1.0)
    assert agreeing.private == ()
    assert "SHARED" in agreeing.lines()[-1]

    disjoint = StudyAttribution().shared(mine, {"other": ranking(list(reversed(studies)))})
    assert disjoint.overlap == pytest.approx(0.0)
    assert len(disjoint.private) == 3, "every tail study is private when no other member ranks it badly"
    assert "PRIVATE" in disjoint.lines()[-1]

    assert StudyAttribution().shared(mine, {}).overlap == 0.0, "no members is no evidence, not full agreement"


def test_shared_tail_lines():
    """An overlap is printed against the chance it is read against, never alone."""
    shared = SharedTail(members=("a", "b"), overlap=0.4, private=("s1",)).lines()
    assert f"{CHANCE:.3f}" in shared[0], "an overlap is meaningless without the chance it is read against"
    assert "SHARED" in shared[-1], "four times chance is a shared tail"


def counted(missed: int, genuine: int, unlocated: int, laterality: int = 0) -> dict[str, int]:
    """One stratum's census, every category present so the two strata always print the same rows."""
    return {
        "missed_negation": missed,
        "laterality": laterality,
        "hedge_dropped": 0,
        "historical": 0,
        "subthreshold": 0,
        "genuine": genuine,
        "unlocated": unlocated,
    }


def test_label_cause_lines():
    """Rates carry their own denominators, and a difference too small or too rare is refused."""
    cause = LabelCause(tail=counted(6, 2, 2), body=counted(2, 6, 2)).lines()
    assert "tail 10" in cause[0]
    assert any("missed_negation" in line and "0.600" in line and "0.200" in line for line in cause)
    assert any("-0.400" in line for line in cause), "GENUINE falling from body to tail is signed"
    assert any("NONE" in line for line in cause), "ten cells against ten separate nothing at two sigma"

    rare = LabelCause(tail=counted(0, 50, 50), body=counted(4, 400, 400)).lines()
    assert any("TOO RARE TO MODEL" in line for line in rare), "a rate of exactly zero has no usable error"


def test_spread():
    """The error of a difference shrinks with both denominators and is zero only where both rates are."""
    wide = LabelCause.spread(0.5, 10, 0.5, 10)
    narrow = LabelCause.spread(0.5, 1000, 0.5, 1000)
    assert wide > narrow, "ninety cells cannot carry the certainty of nine hundred"
    assert LabelCause.spread(0.0, 90, 0.0, 819) == pytest.approx(0.0)


def test_silence():
    """The grouped reading pools the two categories where a zero rests on nothing being said."""
    line = LabelCause(tail=counted(0, 0, 70, 10), body=counted(300, 200, 200, 100)).silence(80, 800)
    assert "1.000" in line, "every tail cell here is unlocated or laterality"
    assert "SEPARATES" in line
    assert "post hoc" in line, "a grouping chosen after seeing the table must say so wherever it prints"

    flat = LabelCause(tail=counted(40, 0, 40, 0), body=counted(400, 0, 400, 0)).silence(80, 800)
    assert "reads nothing" in flat


def test_acquisition_cause_lines():
    """Both strata print on one line, and the difference is signed."""
    acquisition = AcquisitionCause(columns=("planes",), tail={"planes": 2.0}, body={"planes": 3.0}).lines()
    assert "planes" in acquisition[0]
    assert "-1.000" in acquisition[0]


def test_negative_cells():
    """Each study contributes its worst cell the LABEL calls negative, and none where every cell is positive."""
    studies = ["a", "b"]
    truth = np.zeros((2, len(TARGETS)), dtype=np.float32)
    truth[0, 0] = 1.0
    truth[1, :] = 1.0
    table = np.zeros((2, len(TARGETS)))
    table[0, 0] = 0.9
    table[0, 2] = 0.4
    cells = StudyAttribution.negative_cells(ranking(studies), studies, table, truth, TARGETS)
    assert list(cells["study"]) == ["a"], "a study whose every cell is positive has no over-call to explain"
    assert cells.iloc[0]["target"] == TARGETS[2], "the positive cell is not eligible however much damage it holds"
    assert cells.iloc[0]["damage"] == pytest.approx(0.4)


def test_labelled():
    """Every cell gains the language its report is written in, which the detectors need before reading."""
    cells = pd.DataFrame({"study": ["a"], "target": [TARGETS[0]], "damage": [0.5]})
    labelled = StudyAttribution().labelled(cells, {"a": "There is a complete tear of the ACL."})
    assert "language" in labelled.columns
    assert labelled.iloc[0]["language"], "a blank language would silence every cue the audit fires on"


def test_caused():
    """A negated tail against an asserted body reads as the label's defect, not ours."""
    findings = TARGETS[0]
    tail = pd.DataFrame({"study": ["a"], "target": [findings], "damage": [0.9]})
    body = pd.DataFrame({"study": ["b"], "target": [findings], "damage": [0.1]})
    reports = {
        "a": "There is no evidence of a tear of the anterior cruciate ligament.",
        "b": "There is a complete full-thickness tear of the anterior cruciate ligament.",
    }
    cause = StudyAttribution().caused(tail, body, reports)
    assert sum(cause.tail.values()) == 1
    assert sum(cause.body.values()) == 1
    assert cause.body["genuine"] == 1, "a plainly asserted finding is the model's problem, not the label's"
    assert cause.tail["genuine"] == 0, "a negated finding is never GENUINE"


def test_acquired():
    """Acquisition means are per study, so a study with many series does not outvote one with few."""
    series = pd.DataFrame(
        {
            "StudyInstanceUID": ["a", "a", "a", "b"],
            "SeriesInstanceUID": ["1", "2", "3", "4"],
            "Fluid_Sensitive": [1, 1, 0, 0],
            "Fat_Suppression": [1, 0, 0, 0],
            "Anatomical_Plane": ["AX", "COR", "SAG", "AX"],
        }
    )
    acquired = StudyAttribution.acquired({"a"}, {"b"}, series)
    assert acquired.tail["series_count"] == pytest.approx(3.0)
    assert acquired.body["series_count"] == pytest.approx(1.0)
    assert acquired.tail["planes"] == pytest.approx(3.0)
    assert acquired.tail["fluid"] == pytest.approx(2.0 / 3.0)
    assert acquired.body["fat"] == pytest.approx(0.0)


def test_ranked(tmp_path: Path) -> None:
    """A run's record comes back ranked worst-first over the same studies its table was built from."""
    studies = [f"s{index}" for index in range(20)]
    truth = np.tile((np.arange(20) % 2).astype(np.float32)[:, None], (1, len(TARGETS)))
    predictions = 1.0 - truth.astype(np.float64)
    written(tmp_path, "run", predictions, truth, studies)
    rows, table, order, back = AttributionCorpus(models=tmp_path, raw=tmp_path, processed=tmp_path).ranked("run")
    assert order == studies
    assert table.shape == (20, len(TARGETS))
    assert back.shape == truth.shape
    assert rows[0].damage >= rows[-1].damage, "the ranking is worst-first or every tail below is the wrong end"

    with pytest.raises(SystemExit):
        AttributionCorpus(models=tmp_path, raw=tmp_path, processed=tmp_path).ranked("absent")


def test_members(tmp_path: Path) -> None:
    """Only members held out over the SAME studies count, and never the run being attributed itself."""
    studies = [f"s{index}" for index in range(20)]
    truth = np.tile((np.arange(20) % 2).astype(np.float32)[:, None], (1, len(TARGETS)))
    predictions = 1.0 - truth.astype(np.float64)
    written(tmp_path, "run", predictions, truth, studies)
    written(tmp_path, "twin", predictions, truth, studies)
    written(tmp_path, "elsewhere", predictions[:10], truth[:10], studies[:10])
    found = AttributionCorpus(models=tmp_path, raw=tmp_path, processed=tmp_path).members("run", studies, wanted=12)
    assert set(found) == {"twin"}, "a member on another split has a different population, not a differing opinion"

    assert AttributionCorpus(models=tmp_path, raw=tmp_path, processed=tmp_path).members("run", studies, wanted=0) == {}


def test_reports(tmp_path: Path) -> None:
    """Reports come back keyed by study id as text, which is what the audit indexes them by."""
    frame = pd.DataFrame({"StudyInstanceUID": ["a", "b"], "Report": ["intact", "torn"]})
    frame.to_csv(tmp_path / "train.csv", index=False)
    reports = AttributionCorpus(models=tmp_path, raw=tmp_path, processed=tmp_path).reports()
    assert reports == {"a": "intact", "b": "torn"}


def test_series(tmp_path: Path) -> None:
    """The series table is handed over unaggregated, one row per series."""
    frame = pd.DataFrame({"StudyInstanceUID": ["a", "a"], "SeriesInstanceUID": ["1", "2"]})
    frame.to_csv(tmp_path / "train_series.csv", index=False)
    series = AttributionCorpus(models=tmp_path, raw=tmp_path, processed=tmp_path).series()
    assert len(series) == 2, "grouping here would decide the strata before the strata are known"


def test_supervision_cause_lines():
    """An unread rate that separates says under-supervised, and says in the same breath what that is not."""
    apart = SupervisionCause(tail=60, tail_total=90, body=200, body_total=819).lines()
    assert "0.667" in apart[0]
    assert "SEPARATES" in apart[0]
    assert "UNREAD is not MISLABELLED" in apart[1], "a separation here must never be quoted as label error"

    flat = SupervisionCause(tail=20, tail_total=90, body=180, body_total=819).lines()
    assert "reads nothing" in flat[0]
    assert "property of the TEXT" in flat[1]


def test_supervised():
    """The verdict is looked up per cell, so one study can be read on one finding and silent on another."""
    verdicts = pd.DataFrame(
        {
            f"{TARGETS[0]}__verdict": ["UNK", "NO"],
            f"{TARGETS[1]}__verdict": ["NO", "UNK"],
        },
        index=pd.Index(["a", "b"], name="StudyInstanceUID"),
    )
    tail = pd.DataFrame({"study": ["a"], "target": [TARGETS[0]], "damage": [0.9]})
    body = pd.DataFrame({"study": ["a", "b"], "target": [TARGETS[1], TARGETS[0]], "damage": [0.1, 0.1]})
    cause = StudyAttribution.supervised(tail, body, verdicts)
    assert cause.tail == 1, "the same study is UNK on one finding and read on the other"
    assert cause.body == 0
    assert cause.body_total == 2

    absent = pd.DataFrame({"study": ["missing"], "target": [TARGETS[0]], "damage": [0.5]})
    assert StudyAttribution.supervised(absent, body, verdicts).tail == 0, "a study off the table is not UNK"


def test_verdicts(tmp_path: Path) -> None:
    """Verdicts come back indexed by study id as text, which is how the cells are keyed."""
    labels = tmp_path / "labels"
    labels.mkdir()
    frame = pd.DataFrame({"StudyInstanceUID": ["a"], f"{TARGETS[0]}__verdict": ["UNK"]})
    frame.to_csv(labels / "report_labels_v2.csv", index=False)
    read = AttributionCorpus(models=tmp_path, raw=tmp_path, processed=tmp_path).verdicts()
    assert read.at["a", f"{TARGETS[0]}__verdict"] == "UNK"
