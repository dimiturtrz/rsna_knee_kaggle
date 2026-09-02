import json
import sys
from pathlib import Path

import pytest

from core.paths import COMPETITION
from kneemri.experiments.budget_residuals import ALLOCATED, RESERVED, PricedRun, main
from kneemri.runs.run_metrics import METRICS_RECORD, RunMetrics
from kneemri.training.options import TrainingOptions

SETTLED = TrainingOptions(slices=12, batch=12, variant="small", unfreeze_last=6)
"""The recipe `GB_PER_SLICE` was fitted on, whose recorded arms all measured 16.24 GB reserved."""

FITTED_PEAK = 16.24
"""What every arm at that recipe recorded — the point the model reproduces by construction."""


def _recorded(root: Path, name: str, options: TrainingOptions, peaks: list[dict[str, float]]) -> RunMetrics:
    """A metrics file as a finished run leaves one: a recipe line, then a row an epoch."""
    path = root / "models" / COMPETITION / f"{name}{METRICS_RECORD}"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"recipe": options.recipe})]
    lines += [json.dumps({"epoch": index, "auc": 0.9, "per_label": {}, **peak}) for index, peak in enumerate(peaks)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return RunMetrics(path)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A data root a command can be pointed at, holding nothing but what a test writes into it."""
    (tmp_path / "models" / COMPETITION).mkdir(parents=True)
    (tmp_path / "paths.yaml").write_text(f"data: {tmp_path.as_posix()}\n", encoding="utf-8")
    return tmp_path


def test_of(root: Path) -> None:
    """A run is priced from its own recipe, and only against the CORRECTED peak.

    The denominator is the whole difficulty here: rows written before 2026-08-22 recorded ALLOCATED
    memory under the name `vram_peak_gb`, which is the smaller quantity, so grading the model against
    them would credit it for the fragmentation it does not model. Such a row is skipped, and a file
    holding only such rows is not priced at all rather than priced against the wrong number.
    """
    corrected = _recorded(root, "corrected", SETTLED, [{RESERVED: FITTED_PEAK, ALLOCATED: 15.11}])
    priced = PricedRun.of(corrected)
    assert priced is not None
    assert priced.measured == FITTED_PEAK
    assert priced.budget.slices == SETTLED.slices, "the recipe's own flags, not the model's defaults"

    old = _recorded(root, "old", SETTLED, [{RESERVED: 15.11}])
    assert PricedRun.of(old) is None, "a row from the era when that name meant ALLOCATED cannot grade it"

    unrecorded = RunMetrics(root / "models" / COMPETITION / f"nothing{METRICS_RECORD}")
    assert PricedRun.of(unrecorded) is None, "and a run that left no account is silent, not an error"


def test_derived(root: Path) -> None:
    """The model's claim about the world is `needed` MINUS its own slack, and it reproduces the fit."""
    priced = PricedRun.of(_recorded(root, "settled", SETTLED, [{RESERVED: FITTED_PEAK, ALLOCATED: 15.11}]))
    assert priced is not None
    assert priced.derived == pytest.approx(FITTED_PEAK, abs=0.01), (
        "the fitted point must price to what it measured, or the fit has been broken"
    )
    assert priced.derived < priced.budget.needed(), "what a run reserves is under what it is allowed"


def test_slack(root: Path) -> None:
    """The margin is ONE SLICE of this recipe — asked of the model, never respelt from its constants."""
    priced = PricedRun.of(_recorded(root, "settled", SETTLED, [{RESERVED: FITTED_PEAK, ALLOCATED: 15.11}]))
    wider = PricedRun.of(
        _recorded(root, "wider", TrainingOptions(slices=20, batch=12), [{RESERVED: 26.92, ALLOCATED: 25.01}])
    )
    assert priced is not None
    assert wider is not None
    assert priced.slack == pytest.approx(wider.slack, abs=0.01), "a slice costs the same at either depth"
    assert priced.derived + priced.slack == pytest.approx(priced.budget.needed())


def test_residual(root: Path) -> None:
    """Measured minus derived, signed so that POSITIVE is the direction that kills an arm."""
    over = PricedRun.of(_recorded(root, "over", SETTLED, [{RESERVED: FITTED_PEAK + 2.0, ALLOCATED: 15.11}]))
    under = PricedRun.of(_recorded(root, "under", SETTLED, [{RESERVED: FITTED_PEAK - 2.0, ALLOCATED: 15.11}]))
    assert over is not None
    assert under is not None
    assert over.residual == pytest.approx(2.0, abs=0.01)
    assert under.residual == pytest.approx(-2.0, abs=0.01)


def test_refused(root: Path) -> None:
    """A cap under a measured demand is the one failure this model calls fatal, and the slack is where.

    Below the slack the arm still starts; above it the allocator is refused memory the run went on to
    hold, which on this card is an immediate OutOfMemoryError rather than the crawl a cap prevents.
    """
    priced = PricedRun.of(_recorded(root, "settled", SETTLED, [{RESERVED: FITTED_PEAK, ALLOCATED: 15.11}]))
    assert priced is not None
    inside = PricedRun.of(
        _recorded(root, "inside", SETTLED, [{RESERVED: FITTED_PEAK + priced.slack / 2, ALLOCATED: 15.11}])
    )
    beyond = PricedRun.of(
        _recorded(root, "beyond", SETTLED, [{RESERVED: FITTED_PEAK + priced.slack * 2, ALLOCATED: 15.11}])
    )
    assert inside is not None
    assert beyond is not None
    assert not inside.refused, "a residual the slack covers is a run that started"
    assert beyond.refused


def test_line(root: Path) -> None:
    """One run on one line: what it measured, what the model said, and the gap named as a gap."""
    priced = PricedRun.of(_recorded(root, "focal_dense", SETTLED, [{RESERVED: FITTED_PEAK + 4.0, ALLOCATED: 15.11}]))
    assert priced is not None
    line = priced.line()
    assert "focal_dense" in line
    assert f"{FITTED_PEAK + 4.0:.2f}" in line
    assert "CAP UNDER DEMAND" in line, "the fatal direction has to be readable without arithmetic"


def test_main(root: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """The report prices every recorded run, and `--assert` fails only on the fatal direction."""
    _recorded(root, "fits", SETTLED, [{RESERVED: FITTED_PEAK, ALLOCATED: 15.11}])
    argv = ["budget_residuals", "--paths", str(root / "paths.yaml")]

    monkeypatch.setattr(sys, "argv", [*argv, "--assert"])
    with caplog.at_level("INFO"):
        main()
    assert "1 recorded runs, 1 priceable" in caplog.text
    assert "no recorded run measured more than the model would have allowed it" in caplog.text

    _recorded(root, "exceeds", SETTLED, [{RESERVED: FITTED_PEAK * 2, ALLOCATED: 15.11}])
    monkeypatch.setattr(sys, "argv", [*argv, "--assert"])
    with pytest.raises(SystemExit) as refusal:
        main()
    assert refusal.value.code == 1, "a run that outgrew its own cap is a finding, not a warning"

    monkeypatch.setattr(sys, "argv", argv)
    main()  # and without --assert it REPORTS, so a person can read the residuals of a model in progress
