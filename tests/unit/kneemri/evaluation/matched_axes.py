import json
from pathlib import Path

import pytest

from kneemri.evaluation.ledger75 import Ledger75Row
from kneemri.evaluation.matched_axes import (
    DEAD_RUN_FLOOR,
    METRICS_SUFFIX,
    MIN_GROUPS,
    AxisContrast,
    MatchedAxes,
    main,
)

POPULATION = "eb3ba199"
"""One digest for every member here: a group read on two study sets is two questions, not a contrast."""

RECIPES = {
    # Two matched groups on `epochs`, agreeing, plus a third value only one group ever ran.
    "short_a": {"epochs": "12", "variant": "small", "out": "wherever"},
    "long_a": {"epochs": "24", "variant": "small", "out": "elsewhere"},
    "longer_a": {"epochs": "48", "variant": "small"},
    "short_b": {"epochs": "12", "variant": "base"},
    "long_b": {"epochs": "24", "variant": "base"},
    # A seed twin: two members at one value of the held-out key, which must average, never count twice.
    "long_b_twin": {"epochs": "24", "variant": "base", "seed": "2027"},
    # Differs on two keys at once, so it is nobody's matched group.
    "unmatched": {"epochs": "12", "variant": "large"},
}

READINGS = {
    "short_a": 0.8800,
    "long_a": 0.8850,
    "longer_a": 0.8700,
    "short_b": 0.8900,
    "long_b": 0.8930,
    "long_b_twin": 0.8910,
    "unmatched": 0.8700,
}


def _row(name: str, ledger75: float, *, population: str = POPULATION, void: bool = False) -> Ledger75Row:
    return Ledger75Row(
        name=name,
        ledger=ledger75,
        ledger75=ledger75,
        epoch=18,
        studies=909,
        population=population,
        void=void,
    )


def _axes() -> MatchedAxes:
    return MatchedAxes(
        recipes=dict(RECIPES),
        rows={name: _row(name, reading) for name, reading in READINGS.items()},
    )


def test_mean():
    contrast = AxisContrast(key="epochs", favoured="24", against="12", deltas=(0.004, 0.006))
    assert contrast.mean == pytest.approx(0.005)


def test_spread():
    """One group has no spread to report — the floor problem, not a number to divide by."""
    assert AxisContrast(key="k", favoured="a", against="b", deltas=(0.004,)).spread == 0.0
    assert AxisContrast(key="k", favoured="a", against="b", deltas=(0.002, 0.006)).spread == pytest.approx(0.002)


def test_agreement():
    contrast = AxisContrast(key="k", favoured="a", against="b", deltas=(0.004, -0.001, 0.006))
    assert contrast.agreement == 2


def test_line():
    line = AxisContrast(key="epochs", favoured="24", against="12", deltas=(0.004, 0.006)).line()
    assert line.startswith("epochs")
    assert "24 > 12" in line
    assert "+0.0050" in line
    assert "2/2" in line


def test_keys():
    """`out` and `seed` name where a run wrote and which draw it took — never what it tried."""
    assert _axes().keys() == ["epochs", "variant"]


def test_groups():
    groups = _axes().groups("epochs")

    assert sorted(sorted(members) for members in groups) == [
        ["long_a", "longer_a", "short_a"],
        ["long_b", "long_b_twin", "short_b"],
    ]


def test_context():
    """The held-out key and the noise keys drop out; what is left is sorted, so dict order cannot split a group."""
    signature = MatchedAxes.context({"variant": "small", "epochs": "12", "out": "x", "aug": "on"}, "epochs")

    assert signature == (("aug", "on"), ("variant", "small"))


def test_varies():
    """A group is a contrast only with two values of the key, all read on one population."""
    axes = _axes()
    recut = MatchedAxes(recipes=axes.recipes, rows={**axes.rows, "long_a": _row("long_a", 0.885, population="other")})

    assert axes.varies(["short_a", "long_a"], "epochs")
    assert not axes.varies(["short_a", "unmatched"], "epochs"), "one value of the key is no contrast"
    assert not recut.varies(["short_a", "long_a"], "epochs"), "two populations are two questions"


def test_groups_splits_populations():
    """Two rows read on different study sets are two questions, so they are not one another's control."""
    axes = _axes()
    recut = MatchedAxes(
        recipes=axes.recipes,
        rows={**axes.rows, "long_b": _row("long_b", 0.8930, population="other")},
    )

    assert all("long_b" not in members for members in recut.groups("epochs"))


def test_readings():
    """The seed twin averages into its value rather than entering the pool twice."""
    axes = _axes()

    read = axes.readings(["short_b", "long_b", "long_b_twin"], "epochs")

    assert read == {"12": pytest.approx(0.8900), "24": pytest.approx(0.8920)}


def test_contrasts():
    contrasts = {(c.key, c.favoured, c.against): c for c in _axes().contrasts()}

    assert ("epochs", "48", "24") not in contrasts, "48 ran in one group only, under MIN_GROUPS"
    pooled = contrasts["epochs", "24", "12"]
    assert len(pooled.deltas) == MIN_GROUPS
    assert pooled.mean == pytest.approx(0.0035)
    assert pooled.agreement == 2


def test_contrasts_orients_to_the_favoured_value():
    """A pooled mean is printed positive, whichever way the recipe keys happened to sort."""
    axes = MatchedAxes(
        recipes=dict(RECIPES),
        rows={name: _row(name, reading) for name, reading in {**READINGS, "long_a": 0.8700, "long_b": 0.8800}.items()},
    )

    (pooled,) = [c for c in axes.contrasts() if c.key == "epochs"]

    assert (pooled.favoured, pooled.against) == ("12", "24")
    assert pooled.mean > 0


def test_recipe_value():
    """A stamp is text, so `24` and `'24'` are one value and not two arms of an A/B."""
    assert MatchedAxes.recipe_value({"epochs": 24}, "epochs") == "24"


def test_recipe_at(tmp_path: Path):
    """A run with no stamp reads as None, never as a member whose recipe happened to be empty."""
    stamped = tmp_path / f"stamped{METRICS_SUFFIX}"
    stamped.write_text(json.dumps({"recipe": {"epochs": 24}, "auc": 0.9}), encoding="utf-8")
    bare = tmp_path / f"bare{METRICS_SUFFIX}"
    bare.write_text(json.dumps({"auc": 0.9}), encoding="utf-8")

    assert MatchedAxes.recipe_at(stamped) == {"epochs": "24"}
    assert MatchedAxes.recipe_at(bare) is None
    assert MatchedAxes.recipe_at(tmp_path / "absent.jsonl") is None


def test_oriented():
    """A contrast whose deltas run negative is flipped, not printed with a minus sign."""
    flipped = AxisContrast.oriented("epochs", "12", "24", [-0.004, -0.006])

    assert (flipped.favoured, flipped.against) == ("24", "12")
    assert flipped.mean == pytest.approx(0.005)
    assert AxisContrast.oriented("epochs", "12", "24", [0.004]).favoured == "12"


def test_at(tmp_path: Path):
    """A pool with no records reads as no contrasts, rather than as an axis nobody ran."""
    declines = tmp_path / "members.declines"
    declines.write_text("", encoding="utf-8")

    axes = MatchedAxes.at(tmp_path, declines)

    assert axes.recipes == {}
    assert axes.contrasts() == []


def test_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    declines = tmp_path / "members.declines"
    declines.write_text("", encoding="utf-8")
    (tmp_path / f"absent{METRICS_SUFFIX}").write_text(json.dumps({"recipe": {"epochs": "24"}}) + "\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["matched_axes", "--holdouts", str(tmp_path), "--declines", str(declines)])

    with caplog.at_level("INFO"):
        main()

    assert "0 members carry a recipe" in caplog.text


def test_dead_run_floor_is_legislated():
    """Below the floor a member is a crashed arm, and pairing one against a twin reads as a huge effect."""
    assert min(READINGS.values()) > DEAD_RUN_FLOOR
