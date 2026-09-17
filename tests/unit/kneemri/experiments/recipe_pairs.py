"""The four guards, each against the wrong number its absence produced."""

import json
from pathlib import Path

from kneemri.evaluation.ledger75 import Ledger75Row
from kneemri.experiments.recipe_pairs import PAIRED_FLOOR, KnobPair, RecipePairs

MAIN = "66c6f231"
"""The 909-study population every shipping number in this repo is quoted on."""

OTHER = "1c0ffee1"
"""A second holdout. A row read here answers a different question, whatever its recipe says."""

CORNER = {"epochs": "24", "slice_pooling": "labelwise", "slot_dropout": "0.4", "labels": "report_labels_v2.csv"}
"""The shipping corner's recipe, cut to the leaves these cases need."""


def write_run(directory: Path, name: str, recipe: dict[str, str], *, epochs: int | None = None) -> None:
    """A metrics file stamping `recipe` and running `epochs` of the budget it asked for.

    Defaults to running the full budget, because a truncation is the exception every case here has to
    name deliberately.
    """
    asked = int(recipe["epochs"])
    ran = asked if epochs is None else epochs
    lines = [json.dumps({"recipe": {**recipe, "fold": "0", "patience": "0", "out": f"{name}.pt"}})]
    lines += [
        json.dumps({"epoch": epoch, "loss": 1.0, "auc": 0.80 + (0.01 if epoch == 1 else 0.0), "per_label": {}})
        for epoch in range(ran)
    ]
    (directory / f"{name}.metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def row(name: str, score: float, *, population: str = MAIN, void: bool = False) -> Ledger75Row:
    """One ledger row, carrying the two axes a recipe cannot state: which studies, and whether in-sample."""
    return Ledger75Row(name=name, ledger=score, ledger75=score, epoch=1, studies=909, population=population, void=void)


def test_delta() -> None:
    """The step from the first value to the second, signed."""
    pair = KnobPair("epochs", "a", "b", "12", "24", 0.8900, 0.9053)
    assert pair.delta() == 0.9053 - 0.8900


def test_resolved() -> None:
    """A single pair may claim only that it clears the holdout's own floor."""
    assert KnobPair("epochs", "a", "b", "12", "24", 0.8900, 0.9053).resolved()
    assert not KnobPair("labels", "a", "b", "v2", "negpull", 0.9053, 0.9044).resolved(), "inside the floor"
    assert not KnobPair("k", "a", "b", "0", "1", 0.9000, 0.9000 + PAIRED_FLOOR / 2).resolved()


def test_line() -> None:
    """The line carries both run names, so the reading can be re-taken by hand."""
    line = KnobPair("epochs", "span_s2026", "corner_s2026", "12", "24", 0.8900, 0.9053).line()
    assert "+0.0153 CLEARS" in line
    assert "span_s2026 / corner_s2026" in line
    assert "CLEARS" not in KnobPair("labels", "a", "b", "v2", "negpull", 0.9053, 0.9044).line()


def test_recipes(tmp_path: Path) -> None:
    """Every run that stamped a resolved recipe, keyed by stem, with the null leaves dropped."""
    write_run(tmp_path, "corner_s2026", CORNER)
    (tmp_path / "unstamped.metrics.jsonl").write_text(
        json.dumps({"epoch": 0, "loss": 1.0, "auc": 0.8, "per_label": {}}) + "\n", encoding="utf-8"
    )

    found = RecipePairs(tmp_path).recipes()

    assert set(found) == {"corner_s2026"}, "a run with no recipe cannot be paired against anything"
    assert found["corner_s2026"]["slice_pooling"] == "labelwise"


def test_quotable(tmp_path: Path) -> None:
    """One population, not void, and finished — the three refusals, each on its own case."""
    write_run(tmp_path, "corner_s2026", CORNER)
    write_run(tmp_path, "elsewhere_s2026", CORNER)
    write_run(tmp_path, "insample_s2026", CORNER)
    write_run(tmp_path, "dead_s2026", CORNER, epochs=3)

    kept = RecipePairs(tmp_path).quotable(
        [
            row("corner_s2026", 0.9053),
            row("elsewhere_s2026", 0.9100, population=OTHER),
            row("insample_s2026", 0.9400, void=True),
            row("dead_s2026", 0.8054),
        ],
        MAIN,
    )

    assert set(kept) == {"corner_s2026"}
    assert kept["corner_s2026"] == 0.9053


def test_isolating(tmp_path: Path) -> None:
    """One differing leaf makes a pair; none, several, or a shape mismatch make nothing."""
    pairs = RecipePairs(tmp_path)

    assert pairs.isolating(CORNER, {**CORNER, "epochs": "12"}) == "epochs"
    assert pairs.isolating(CORNER, CORNER) is None
    assert pairs.isolating(CORNER, {**CORNER, "epochs": "12", "slot_dropout": "0.0"}) is None

    # Schema drift BREAKS the pair. Tolerating it admitted recipes thirty-six leaves apart as isolating.
    assert pairs.isolating(CORNER, {**CORNER, "layer_decay": "0.75"}) is None

    # A run is NAMED differently from every other run and never differs BECAUSE of it.
    assert pairs.isolating({**CORNER, "out": "a.pt"}, {**CORNER, "out": "b.pt", "epochs": "12"}) == "epochs"


def test_table(tmp_path: Path) -> None:
    """Every single-key pair among the quotable runs, grouped by the leaf it prices."""
    recipes = {
        "corner_s2026": CORNER,
        "short_s2026": {**CORNER, "epochs": "12"},
        "negpull_s2026": {**CORNER, "labels": "report_labels_negpull.csv"},
    }
    scores = {"corner_s2026": 0.9053, "short_s2026": 0.8900, "negpull_s2026": 0.9044}

    table = RecipePairs(tmp_path).table(scores, recipes)

    assert set(table) == {"epochs", "labels"}, "corner-to-short and corner-to-negpull; short-to-negpull differs twice"
    assert len(table["epochs"]) == 1
    assert round(table["labels"][0].delta(), 4) == -0.0009


def test_lines(tmp_path: Path) -> None:
    """Keys with the most pairs first, each key's largest |delta| at its top, floor named beside it."""
    table = {
        "labels": [
            KnobPair("labels", "a", "b", "v2", "negpull", 0.9053, 0.9044),
            KnobPair("labels", "c", "d", "confident", "v6tears", 0.8785, 0.8580),
        ],
        "epochs": [KnobPair("epochs", "e", "f", "12", "24", 0.8900, 0.9053)],
    }

    rendered = RecipePairs(tmp_path).lines(table)

    assert rendered[0] == f"== labels  (2 pairs, 1 clearing {PAIRED_FLOOR})"
    assert "-0.0205" in rendered[1], "the largest |delta| leads its own key"
    assert rendered[3].startswith("== epochs")
