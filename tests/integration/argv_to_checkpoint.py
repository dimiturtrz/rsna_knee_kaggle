"""The command line to a checkpoint: `train.main` itself, on a synthetic corpus, on the CPU.

Every piece of the run's setup is unit-tested and `corpus_to_checkpoint` fits a member through the real
setup methods — but `main` is the function that WIRES them, and it was executed by nothing. That is the
half of this repo's failure class no unit test can see: nine times a value was recorded in a config
while its consumer read its own default, and the way it happens is a flag that reaches the options
object and then never reaches the thing it configures. `fitted_trainer` attaching a policy proves
nothing if `main` stops calling it.

So this drives the real entry point from argv: cache load, label pairing, fold assignment, model build,
trainer build, the fitting loop and the final save. The assertion is that the checkpoint records the
reading the COMMAND LINE asked for — the flags that select pixels, because a member served under a
reading it was not fitted under is the defect that reached the leaderboard.
"""

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import torch

from core.paths import COMPETITION
from kneemri import train as train_module
from kneemri.backbone import ENCODERS, SMALL
from kneemri.folds import ReportFolds
from kneemri.model import TARGETS
from kneemri.slots import LAST_MAXIMAL, SCHEMES, Routing
from kneemri.study_cache import StudyCache

from ..unit.kneemri.model import DIM, _FakeBackbone

STUDIES = 8
SLICES = 4
SIZE = 28
"""A multiple of the patch grid, because `main` refuses a read size the encoder cannot tile."""
FOLDS = 2
SPLIT_SEED = 2026
"""Passed on the argv too, so the labels are balanced against the split the run actually uses."""


def _corpus(root: Path, routing: Routing | None = None) -> list[str]:
    """Write the three files `StudyCache.load` reads, under the key the argv below selects.

    The routing is a parameter because it is part of that key: an arm asking for the other tie rule
    reads a DIFFERENT cache, which is the property that keeps two routings from overwriting each other.
    """
    identifiers = [f"s{index}" for index in range(STUDIES)]
    cache = StudyCache(root / "processed" / COMPETITION, slices=SLICES, image_size=SIZE, routing=routing)
    cache.directory.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(2026)
    np.save(cache.images_path, generator.integers(0, 256, cache.shape(STUDIES), dtype=np.uint8))
    np.save(cache.mask_path, np.ones((STUDIES, cache.slots), dtype=np.float32))
    cache.index_path.write_text(json.dumps(identifiers), encoding="utf-8")
    return identifiers


def _tables(root: Path, identifiers: list[str]) -> Path:
    """The report-derived labels and the shipped `train.csv` the folds are hashed from.

    The findings are laid out AGAINST the fold assignment rather than against the study order, so each
    fold holds both classes in every column. A holdout of one class scores NaN in every label, which
    ranks no epoch, which raises before the run reaches the save this test exists to check.
    """
    raw = root / "raw" / COMPETITION
    raw.mkdir(parents=True, exist_ok=True)
    # Distinct report text per study: two studies sharing a report share a fold by construction.
    table = raw / "train.csv"
    reports = pd.DataFrame({"StudyInstanceUID": identifiers, "Report": [f"report {name}" for name in identifiers]})
    reports.to_csv(table, index=False)

    assignments = ReportFolds(FOLDS, SPLIT_SEED).assign(str(table), identifiers)
    positive = np.zeros(len(identifiers), dtype=np.float32)
    for fold in range(FOLDS):
        members = np.flatnonzero(assignments == fold)
        positive[members[: len(members) // 2]] = 1.0
    labels = raw / "labels.csv"
    scores = dict.fromkeys(TARGETS, positive)
    pd.DataFrame({"StudyInstanceUID": identifiers, **scores}).to_csv(labels, index=False)
    return labels


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A data root `main` can run against, with the weights and the card taken out of its way."""
    _tables(tmp_path, _corpus(tmp_path))
    paths = tmp_path / "paths.yaml"
    paths.write_text(f"data: {tmp_path.as_posix()}\n", encoding="utf-8")

    monkeypatch.setattr(train_module.AutoModel, "from_pretrained", lambda repository: _FakeBackbone())
    # The stand-in backbone is DIM wide, and the head is built at the ENCODER's width: a mismatch here
    # is a shape error rather than a test of the wiring.
    monkeypatch.setattr(train_module, "ENCODERS", {SMALL: replace(ENCODERS[SMALL], dim=DIM)})
    # This suite never trains on the card, and `DeviceBudget` takes it whenever it is there.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return tmp_path


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *flags: str) -> dict[str, object]:
    """Run `main` from an argv and read back the checkpoint it wrote."""
    destination = tmp_path / "member.pt"
    argv = [
        "train",
        "--paths",
        str(tmp_path / "paths.yaml"),
        "--labels",
        str(tmp_path / "raw" / COMPETITION / "labels.csv"),
        "--out",
        str(destination),
        "--epochs",
        "1",
        "--batch",
        "2",
        "--folds",
        str(FOLDS),
        "--split-seed",
        str(SPLIT_SEED),
        "--slices",
        str(SLICES),
        "--image-size",
        str(SIZE),
        *flags,
    ]
    monkeypatch.setattr("sys.argv", argv)
    train_module.main()
    return cast(dict[str, object], torch.load(destination, map_location="cpu", weights_only=False))


def _nested(payload: dict[str, object], key: str) -> dict[str, object]:
    """One of the checkpoint's nested tables, as the loaders read it."""
    return cast(dict[str, object], payload[key])


def test_the_command_line_reaches_the_checkpoint(corpus: Path, monkeypatch: pytest.MonkeyPatch):
    """The flags that decide WHICH PIXELS a member reads must arrive on the member.

    Each of these travels through a different part of `main`: `--whole-stack` through `fitted_trainer`,
    `--slice-pooling` through the `StudyDesign` built here, `--crop-mm`/`--tie` through the cache key
    and the record. A flag parsed and then dropped is this project's most expensive failure — it does
    not crash, it trains a recipe nobody asked for and reports a plausible number for it.
    """
    _corpus(corpus, Routing(tie=LAST_MAXIMAL))
    payload = _run(corpus, monkeypatch, "--whole-stack", "--slice-pooling", "max", "--tie", LAST_MAXIMAL)

    config = _nested(payload, "config")
    assert config["whole_stack"] is True, "a whole-stack member served in windows predicts from other pixels"
    assert config["slice_pooling"] == "max"
    assert config["tie"] == "last", "the router is what fills each slot, so serving cannot fall back to ours"
    assert config["slices"] == SLICES
    assert config["train_size"] == SIZE, "0 means the cache's own size, and the member must state which that was"
    assert payload["fingerprint"], "a member without one cannot be verified, so it cannot be submitted"
    assert _nested(payload, "fingerprint_config")["n_slot"] == len(SCHEMES[StudyCache.CANONICAL])


def test_the_run_keeps_an_epoch_and_saves_it(corpus: Path, monkeypatch: pytest.MonkeyPatch):
    """The final save is the KEPT epoch's, with the score and the witness that ranked it.

    A run once left its best epoch as weights with no fingerprint and no recorded reading, and the
    witness is what `witness_check` re-derives before a submission — an empty one is a member that
    cannot be checked against the cache a kernel builds.
    """
    payload = _run(corpus, monkeypatch, "--whole-stack", "--slice-pooling", "max")

    assert payload["epoch"] == 0
    assert isinstance(payload["holdout"], float)
    witness = _nested(_nested(payload, "config"), "witness")
    studies = cast(list[str], witness["studies"])
    assert studies, "the witness names studies, so it travels to another process"
    assert len(cast(list[object], witness["predictions"])) == len(studies)
    assert payload["model"], "the weights the kept epoch produced"


def test_a_full_fit_holds_nothing_back(corpus: Path, monkeypatch: pytest.MonkeyPatch):
    """`--fold -1` is the final-fit mode: no fold is ever assigned -1, so the holdout is empty.

    It cannot report a score by construction, which is the one path through `Fitting` where the kept
    epoch is the LAST rather than the best — and a run that raised here would be discovered after the
    card had already been held for an hour.
    """
    payload = _run(corpus, monkeypatch, "--fold", "-1")

    assert _nested(payload, "config")["witness"] == {}, "a run with no holdout has nothing to witness"
