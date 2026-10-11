"""A weighted vote rule must be scored on studies its weights never saw, or it grades its own homework."""

import sys
from pathlib import Path

import numpy as np
import pytest

from core.paths import ENV_OVERRIDE, DataRoot
from kneemri.evaluation import vote_ceiling
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.scores import LabelScores
from kneemri.evaluation.vote_ceiling import RuleReading, VoteCeiling
from kneemri.inference.vote import RankAverage

STUDIES = 60
LABELS = 12
SEED = 2026


def _ceiling(strengths: list[float]) -> VoteCeiling:
    """Members that separate the truth as well as their strength says, so a weighting has something to find."""
    generator = np.random.default_rng(SEED)
    truth = generator.integers(0, 2, size=(STUDIES, LABELS)).astype(np.float32)
    predictions = [
        truth.astype(np.float64) * strength + generator.normal(0.0, 1.0, size=(STUDIES, LABELS))
        for strength in strengths
    ]
    names = [f"m{index}" for index in range(len(strengths))]
    return VoteCeiling(Pool(names, predictions, truth, RankAverage()))


def test_over(tmp_path: Path):
    """A pool where nothing carries a holdout record is refused, not read over whatever resolved.

    The failure this prevents is the wrong-denominator one: a rule compared over a pool silently
    shorter than the caller named would land in the ledger as a statement about the RULE. `RecordMix`
    refuses the missing file before the `Denominator` is reached, which is the earlier of the two
    guards and the one that names the path.
    """
    with pytest.raises(FileNotFoundError):
        VoteCeiling.over([tmp_path / "absent.pt"])


def test_halves():
    """The halves partition the studies exactly once each, which is what makes the two reads paired."""
    halves = _ceiling([3.0, 2.0]).halves()

    assert sum(half.sum() for half in halves) == STUDIES, "every study is scored in exactly one half"
    assert not (halves[0] & halves[1]).any(), "no study is in both halves"


def test_quality():
    """The fitted table is one AUC per member per label, over the named studies only."""
    ceiling = _ceiling([3.0, 0.5])

    table = ceiling.quality(LabelScores(), ceiling.halves()[0])

    assert table.shape == (2, LABELS)
    assert table[0].mean() > table[1].mean(), "the stronger member scores higher on the half it is fitted on"


def test_rules():
    """Every rule the project owns is read, so a comparison cannot quietly omit the one that wins."""
    ceiling = _ceiling([3.0, 2.0])

    rules = ceiling.rules(ceiling.quality(LabelScores(), ceiling.halves()[0]))

    assert [rule.named() for rule in rules] == [
        "RankAverage",
        "QualityWeightedRank",
        "LabelQualityRank",
        "LogitAverage",
        "MedianLogit",
        "SharpenedRank(2)",
        "SharpenedRank(4)",
        "SharpenedRank(8)",
        "SharpenedRank(16)",
    ]


def test_scored():
    """A rule is scored on the half it is given and nowhere else — the whole point of the split."""
    ceiling = _ceiling([3.0, 2.0])
    first, second = ceiling.halves()

    assert ceiling.scored(LabelScores(), RankAverage(), first) != ceiling.scored(LabelScores(), RankAverage(), second)


def test_read():
    """The flat rule is the baseline every other row is a delta against, so its own delta is zero."""
    readings = _ceiling([3.0, 2.0, 1.0]).read(LabelScores())

    assert readings[0].rule == "RankAverage"
    assert readings[0].against_flat == pytest.approx(0.0), "flat is measured against itself"
    assert all(reading.fitted_on_self >= reading.honest - 0.05 for reading in readings), "self-fit does not lose badly"


def test_lines():
    """The report ends on the rule a submission would serve, named with its honest margin."""
    rows = _ceiling([3.0, 2.0]).lines(LabelScores())

    assert rows[0].startswith("vote rules over 2 members")
    assert "BEST honest rule" in rows[-1]


def test_line():
    """A row carries the honest number, its margin and the overfit gap that qualifies both."""
    row = RuleReading("LabelQualityRank", 0.8739, 0.8745, 0.0001).line()

    assert "honest 0.8739" in row
    assert "+0.0001" in row and "fit-on-self 0.8745" in row
    assert "overfit gap +0.0006" in row, "the gap is derived, not passed, so it cannot disagree with its parts"


class _ResolvedError(Exception):
    """Raised with the checkpoints the command resolved, before any of them is read."""


def _resolved(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> list[Path]:
    """The checkpoint list `main` hands to the ceiling for `argv`."""

    def _stop(cls: type[VoteCeiling], paths: list[Path]) -> VoteCeiling:
        raise _ResolvedError(paths)

    monkeypatch.setattr(sys, "argv", ["vote_ceiling", *argv])
    monkeypatch.setattr(VoteCeiling, "over", classmethod(_stop))
    with pytest.raises(_ResolvedError) as caught:
        vote_ceiling.main()
    return caught.value.args[0]


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A roster and a candidate list name the same checkpoints the same way, and naming none is refused."""
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    models = DataRoot(tmp_path).models()
    roster = tmp_path / "members.roster"
    roster.write_text("# header\n\nalpha.pt   # why\n  beta  \nelsewhere/gamma.pt\n", encoding="utf-8")

    assert _resolved(monkeypatch, ["--roster", str(roster)]) == [
        models / "alpha.pt",
        models / "beta.pt",
        Path("elsewhere/gamma.pt"),
    ]
    assert _resolved(monkeypatch, ["--candidates", "alpha", "beta.pt"]) == [models / "alpha.pt", models / "beta.pt"]

    monkeypatch.setattr(sys, "argv", ["vote_ceiling"])
    with pytest.raises(SystemExit):
        vote_ceiling.main()
