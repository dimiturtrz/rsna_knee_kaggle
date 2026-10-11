"""A roster's score is bought with decodes, and the search must refuse to overspend them."""

import sys
from pathlib import Path

import numpy as np
import pytest

from core.paths import ENV_OVERRIDE, DataRoot
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.scores import LabelScores
from kneemri.inference.vote import RankAverage
from kneemri.kaggle import decode_entry
from kneemri.kaggle.decode_entry import BudgetedRoster, DecodeEntry

STUDIES = 40
LABELS = 12
SEED = 2026


def _pool(strengths: dict[str, float]) -> Pool:
    """Members that separate the truth as well as their strength says, and no better.

    A member is truth plus noise scaled by how weak it is, so the ranking `LabelScores` reads is real
    rather than random and a stronger member genuinely scores higher — which is what makes the greedy
    walk's ORDER a thing worth asserting.
    """
    generator = np.random.default_rng(SEED)
    truth = generator.integers(0, 2, size=(STUDIES, LABELS)).astype(np.float32)
    predictions = [
        truth.astype(np.float64) * strength + generator.normal(0.0, 1.0, size=(STUDIES, LABELS))
        for strength in strengths.values()
    ]
    return Pool(list(strengths), predictions, truth, RankAverage())


def _entry(strengths: dict[str, float], keys: dict[str, str]) -> DecodeEntry:
    """A search over a synthetic pool, with each member assigned to the read it rides."""
    return DecodeEntry(_pool(strengths), keys)


def test_over(tmp_path: Path):
    """A pool where nothing describes a decode is refused, not searched over silently.

    The failure this prevents is the wrong-denominator one: a name list with a typo in it would
    otherwise report the best roster over the members that happened to resolve.
    """
    with pytest.raises(SystemExit):
        DecodeEntry.over([tmp_path / "absent.pt"])


def test_cost():
    """Cost is DISTINCT reads, so a second member on a seated read is free and a new read is not."""
    entry = _entry({"a": 3.0, "b": 2.0, "c": 1.0}, {"a": "read A", "b": "read A", "c": "read B"})

    assert entry.cost([0, 1]) == 1, "two members on one read spend one decode"
    assert entry.cost([0, 2]) == 2, "a member on another read spends a second"


def test_walk():
    """The ceiling binds per candidate: at one decode the search may never leave the read it opened."""
    keys = {"strong": "read A", "shared": "read A", "distant": "read B"}
    entry = _entry({"strong": 3.0, "shared": 2.0, "distant": 2.5}, keys)

    within_one = entry.walk(LabelScores(), max_decodes=1)
    within_two = entry.walk(LabelScores(), max_decodes=2)

    assert all(step.decodes == 1 for step in within_one), "a one-decode budget can never serve two reads"
    assert "distant" not in within_one[-1].names, "the second read is unaffordable however good it is"
    assert max(step.decodes for step in within_two) <= 2, "the ceiling holds at every step, not only the last"


def test_walk_stops_on_no_gain():
    """A step that does not improve the score ends the walk, so the roster reported is the smallest one.

    TWINS, not merely weak members, are what makes this bite: independent members with independent noise
    all help a little, so the walk is right to seat them. A perfect copy adds nothing to a rank average,
    and a walk that accepts a tie would seat it anyway — and the efficiency track pays for seats.
    """
    twin = _pool({"a": 3.0})
    pool = Pool([*twin.names, "copy"], [*twin.predictions, twin.predictions[0]], twin.truth, RankAverage())
    entry = DecodeEntry(pool, {"a": "read A", "copy": "read A"})

    steps = entry.walk(LabelScores(), max_decodes=1)

    assert len(steps) == 1, "one of the twins is seated and the other buys no score, so it takes no seat"


def test_lines():
    """The report ends on the roster a slot would be spent on, named in full."""
    entry = _entry({"a": 3.0, "b": 2.0}, {"a": "read A", "b": "read A"})

    rows = entry.lines(LabelScores(), max_decodes=1)

    assert rows[0].startswith("greedy walk under a ceiling of 1 decodes")
    assert "BEST at 1 decodes" in rows[-2], "the score and its seat count read together"
    assert "a" in rows[-1], "the last line names the members, because that is what gets served"


def test_per_decode():
    """The trade: three seats on one decode is three times the value of three seats on three."""
    assert BudgetedRoster(["a", "b", "c"], 0.9, 1).per_decode == pytest.approx(3.0)
    assert BudgetedRoster(["a", "b", "c"], 0.9, 3).per_decode == pytest.approx(1.0)


def test_line():
    """A step reads as a decision, so the member it just seated is what ends the row."""
    row = BudgetedRoster(["seated", "newest"], 0.9312, 1).line()

    assert row.endswith("+newest"), "the newest seat is the decision the row records"
    assert "0.9312" in row and "2 seats / 1 decodes" in row


class _ResolvedError(Exception):
    """Raised with the checkpoints the command resolved, before any of them is read."""


def _resolved(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> list[Path]:
    """The checkpoint list `main` hands to the search for `argv`."""

    def _stop(cls: type[DecodeEntry], paths: list[Path]) -> DecodeEntry:
        raise _ResolvedError(paths)

    monkeypatch.setattr(sys, "argv", ["decode_entry", *argv])
    monkeypatch.setattr(DecodeEntry, "over", classmethod(_stop))
    with pytest.raises(_ResolvedError) as caught:
        decode_entry.main()
    return caught.value.args[0]


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A roster and a candidate list name the same checkpoints the same way, and naming none is refused."""
    monkeypatch.setenv(ENV_OVERRIDE, str(tmp_path))
    models = DataRoot(tmp_path).models()
    roster = tmp_path / "members.roster"
    roster.write_text("# header\n\nalpha.pt   # why\n  beta  \nelsewhere/gamma.pt\n", encoding="utf-8")

    assert _resolved(monkeypatch, ["--roster", str(roster), "--max-decodes", "2"]) == [
        models / "alpha.pt",
        models / "beta.pt",
        Path("elsewhere/gamma.pt"),
    ]
    assert _resolved(monkeypatch, ["--candidates", "alpha", "beta.pt"]) == [models / "alpha.pt", models / "beta.pt"]

    monkeypatch.setattr(sys, "argv", ["decode_entry"])
    with pytest.raises(SystemExit):
        decode_entry.main()
