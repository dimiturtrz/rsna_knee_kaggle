"""Re-deriving a finished run's holdout: the split it must reproduce, and the runs it refuses."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core.data.corpus.study_cache import CachedCorpus
from core.findings import TARGETS
from core.paths import DataRoot
from kneemri.inference.holdout_backfill import Backfill, HoldoutBackfill, Reading, ReadSize, UnbackfillableError
from kneemri.inference.members import RESIZED_FROM, Member
from kneemri.inference.prediction_cache import PredictionCache
from kneemri.inference.test_time import SpatialViews
from kneemri.runs.holdout_record import DERIVED, UNKNOWN_EPOCH, HoldoutRecord

STUDIES = ["s0", "s1", "s2", "s3", "s4", "s5"]

RECIPE = {"folds": "3", "split_seed": "2026", "fold": "1", "labels": "labels.csv", "eval_labels": "None"}


def _metrics(checkpoint: Path, recipe: dict[str, str] | None, epochs: list[dict[str, object]]) -> Path:
    lines = ([json.dumps({"recipe": recipe})] if recipe else []) + [json.dumps(row) for row in epochs]
    checkpoint.with_suffix(".metrics.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checkpoint


def _epoch(epoch: int, auc: float) -> dict[str, object]:
    return {"epoch": epoch, "loss": 1.0, "auc": auc, "seconds": 10, "per_label": {"MCL": auc}}


def _tables(root: Path) -> Path:
    """A raw directory holding the two tables the split and the ruler are read from."""
    raw = DataRoot(root).raw()
    raw.mkdir(parents=True, exist_ok=True)
    reports = "\n".join(f"{name},report {index}" for index, name in enumerate(STUDIES))
    (raw / "train.csv").write_text("StudyInstanceUID,Report\n" + reports + "\n", encoding="utf-8")
    return raw


def test_of(tmp_path: Path):
    """The epoch re-derived is the BEST one, because that is the checkpoint sitting beside the file."""
    checkpoint = _metrics(tmp_path / "arm.pt", RECIPE, [_epoch(0, 0.80), _epoch(1, 0.86), _epoch(2, 0.84)])

    plan = Backfill.of(checkpoint)

    assert plan.epoch == 1, "under patience the last epoch's weights were thrown away"
    assert plan.recipe["fold"] == "1"


def test_of_refuses_a_run_that_recorded_no_recipe(tmp_path: Path):
    """Most of the ledger predates the recipe, and a guessed split is not an error anywhere downstream."""
    checkpoint = _metrics(tmp_path / "old.pt", None, [_epoch(0, 0.80)])

    with pytest.raises(UnbackfillableError, match="no recipe"):
        Backfill.of(checkpoint)


def test_of_refuses_a_full_data_fit(tmp_path: Path):
    """A run with `--fold -1` held nothing out, so there is no per-label claim to re-read."""
    checkpoint = _metrics(tmp_path / "final.pt", {**RECIPE, "fold": "-1"}, [_epoch(0, float("nan"))])

    with pytest.raises(UnbackfillableError, match="held nothing out"):
        Backfill.of(checkpoint)


def test_split(tmp_path: Path):
    """The run's OWN fold, from its recorded recipe — this command's flags never enter it."""
    raw = _tables(tmp_path)
    plan = Backfill(checkpoint=tmp_path / "arm.pt", recipe=RECIPE, epoch=1)

    holdout = plan.split(STUDIES, raw / "train.csv")
    other = Backfill(checkpoint=tmp_path / "arm.pt", recipe={**RECIPE, "fold": "2"}, epoch=1).split(
        STUDIES, raw / "train.csv"
    )

    assert set(holdout) <= set(range(len(STUDIES)))
    assert not set(holdout) & set(other), "two folds of one split share no study"


def test_ruler(tmp_path: Path):
    """An arm scored against a held-fixed table is paired by THAT table, not by the one it trained on."""
    raw = _tables(tmp_path)
    (tmp_path / "trained.csv").write_text("StudyInstanceUID\n" + "\n".join(STUDIES) + "\n", encoding="utf-8")
    (tmp_path / "ruler.csv").write_text(
        f"StudyInstanceUID,{TARGETS[0]}\n" + "\n".join(f"{name},1.0" for name in STUDIES) + "\n", encoding="utf-8"
    )
    recipe = {**RECIPE, "labels": str(tmp_path / "trained.csv"), "eval_labels": str(tmp_path / "ruler.csv")}

    scored = Backfill(checkpoint=tmp_path / "arm.pt", recipe=recipe, epoch=1).ruler(STUDIES, raw / "train.csv")

    assert scored.scores[:, 0] == pytest.approx(1.0), "the table it was SCORED against, not the one it trained on"


def _member(checkpoint: Path) -> Member:
    # A REAL torch file, not placeholder bytes: the cache key only stats it, but `Backfill.of` reads
    # the stamp inside it, and a checkpoint this suite writes must be one our own loaders can open.
    torch.save({"model": {}}, checkpoint)
    return Member(
        identifier=checkpoint.stem,
        file=checkpoint,
        fold=1,
        config={
            "img": 336,
            "slices": 4,
            "group": 2,
            "crop_mm": 130.0,
            "band": [0.2, 0.8],
            "slots": ["a", "b"],
            "variant": "small",
            "pool": "cls_mean",
        },
        fingerprint=np.zeros((2, 12), np.float32),
        holdout=0.8,
        annot=0.8,
    )


def test_derive(tmp_path: Path):
    """A back-filled record says DERIVED, over the run's own studies, without touching an encoder.

    The predictions come from the cache, so nothing here decodes pixels or loads weights — which is
    also what makes the back-fill runnable while the card belongs to something else.
    """
    raw = _tables(tmp_path)
    (tmp_path / "labels.csv").write_text("StudyInstanceUID\n" + "\n".join(STUDIES) + "\n", encoding="utf-8")
    checkpoint = _metrics(tmp_path / "arm.pt", RECIPE, [_epoch(0, 0.80), _epoch(1, 0.86)])
    member = _member(checkpoint)
    plan = Backfill.of(checkpoint)
    plan = Backfill(checkpoint=checkpoint, recipe={**RECIPE, "labels": str(tmp_path / "labels.csv")}, epoch=plan.epoch)

    corpus = CachedCorpus(np.zeros((len(STUDIES), 2, 4, 1, 1), np.uint8), np.ones((len(STUDIES), 2), np.float32))
    held = plan.split(STUDIES, raw / "train.csv")
    views = SpatialViews(degrees=(0.0,), shifts=((0.0, 0.0),))
    root = DataRoot(tmp_path)
    store = PredictionCache(root.processed() / "holdout_predictions", HoldoutRecord.named(STUDIES, held), views)
    backfill = HoldoutBackfill(root, torch.device("cpu"))
    window = member.window(corpus.slices, overlap=False)
    store.write(store.key(member.file, window, member.precision), np.full((len(held), len(TARGETS)), 0.25))

    written = backfill.derive(member, plan, corpus, STUDIES)

    stored = HoldoutRecord(checkpoint).read()
    assert written == len(held)
    assert stored is not None
    assert stored.source == DERIVED, "a re-derivation is accurate to a tolerance wider than the interval"
    assert stored.studies == HoldoutRecord.named(STUDIES, held)
    assert stored.epoch == 1


def test_of_reads_the_split_from_a_checkpoint_that_kept_no_metrics_file(tmp_path: Path):
    """50 of the 69 members on disk wrote no metrics file, and 36 of those stamp the split anyway.

    Refusing them read as "no recipe recorded" when the recipe was sitting inside the checkpoint the
    command had already been handed — a whole ledger held unreadable for want of a file rather than a
    fact.
    """
    checkpoint = tmp_path / "stamped.pt"
    torch.save({"model": {}, "options": {**RECIPE, "epochs": "8"}, "epochs_done": 8}, checkpoint)

    plan = Backfill.of(checkpoint)

    assert plan.recipe["fold"] == "1", "its own split, taken from its own stamp"
    # `epochs_done` is NOT the kept epoch plus one on this half of the ledger — all three pre-`epoch`
    # checkpoints checked write the run's configured length there, and reading it as a save stamped
    # `wsfold0_e6`'s predictions epoch 11 when its weights are the epoch-6 best. The arrays are still
    # re-derivable, so the run is not refused; only the stamp says it does not know.
    assert plan.epoch == UNKNOWN_EPOCH, "a length is not a save, and a plausible guess is the worse answer"


def test_of_prefers_the_metrics_file_when_a_run_kept_both(tmp_path: Path):
    """The metrics file names the epoch that SCORED best; the stamp can only name how many ran."""
    checkpoint = _metrics(tmp_path / "both.pt", RECIPE, [_epoch(0, 0.80), _epoch(1, 0.86), _epoch(2, 0.84)])
    torch.save({"model": {}, "options": RECIPE, "epochs_done": 3}, checkpoint)

    assert Backfill.of(checkpoint).epoch == 1, "under patience the last epoch's weights were thrown away"


def test_read(tmp_path: Path):
    """Pricing a size overrides `train_size`, the knob the encoder actually runs at.

    Moving `img` instead names a different decode cache and leaves the encoder where it was, which
    times and scores as though nothing changed (knee-d2fc). The member also comes back marked, so its
    own trainer record can no longer stand in for the pass.
    """
    priced = ReadSize(224).read(_member(tmp_path / "fitted.pt"))

    assert priced.read_size == 224, "the encoder runs at the priced size"
    assert priced.image_size == 336, "and decodes from the cache it was fitted on"
    assert RESIZED_FROM in priced.config, "so the trainer's own array is refused for it"


def test_read_size_destination(tmp_path: Path):
    """A priced record is named for its size, because the unsized name is the one everything reads.

    `HoldoutRecord` derives its path from whatever it is handed, so writing a 224-px pass against the
    checkpoint would overwrite the member's native holdout — the array every ensemble, seat search and
    curve in this project scores from.
    """
    checkpoint = tmp_path / "member.pt"

    destination = ReadSize(224).destination(checkpoint)

    assert destination == tmp_path / "member_r224.pt"
    assert HoldoutRecord(destination).path != HoldoutRecord(checkpoint).path, "the native record survives a price"


def test_reading_destination(tmp_path: Path):
    """A sized reading writes beside the member; a native one writes the member's own record."""
    checkpoint = tmp_path / "member.pt"
    assert Reading(sizing=ReadSize(224)).destination(checkpoint) == tmp_path / "member_r224.pt"
    assert Reading().destination(checkpoint) == checkpoint


def test_stamp(tmp_path: Path):
    """A checkpoint that already witnesses its own weights is left alone.

    The half that rebuilds the encoder is not exercised here, for the reason `HoldoutPredictions.load`
    states about its own backbone: it needs a network and a 90 MB download, so a unit test of it would
    not be a unit test. What IS testable is the refusal, and it is the one that decides whether a member
    silently gets a fresh witness for weights nobody re-read.
    """
    backfill = HoldoutBackfill(DataRoot(tmp_path), torch.device("cpu"))

    witnessed = tmp_path / "witnessed.pt"
    torch.save({"model": {}, "fingerprint": [0.1, 0.2], "holdout": 0.87}, witnessed)
    assert not backfill.stamp(_member(tmp_path / "ignored.pt"), witnessed), "a witness is never recomputed silently"


def _fitted(checkpoint: Path) -> Path:
    torch.save({"model": {}, "config": _member(checkpoint).config, "fingerprint": [0.1], "holdout": 0.8}, checkpoint)
    return checkpoint


def test_native():
    """Only the reading a member was fitted at may stamp its checkpoint's price."""
    assert Reading().native()
    assert not Reading(sizing=ReadSize(224)).native(), "a sized pass prices something else"


def test_load(tmp_path: Path):
    """The member comes back at the priced size, over the cache it was FITTED on."""
    checkpoint = _fitted(tmp_path / "fitted.pt")
    cache = Member.from_checkpoint(checkpoint).cache(tmp_path)
    cache.prepare()
    np.save(cache.images_path, np.zeros((2, 2, 4, 1, 1), np.uint8))
    np.save(cache.mask_path, np.ones((2, 2), np.float32))
    cache.record(["s0", "s1"])

    member, corpus, identifiers = Reading(sizing=ReadSize(224)).load(checkpoint, tmp_path)

    assert member.read_size == 224
    assert identifiers == ["s0", "s1"]
    assert corpus.slices == 4


def test_visit(tmp_path: Path):
    """A walk steps over a checkpoint it cannot re-derive rather than dying on it."""
    checkpoint = _metrics(_fitted(tmp_path / "old.pt"), None, [_epoch(0, 0.80)])
    backfill = HoldoutBackfill(DataRoot(tmp_path), torch.device("cpu"))

    assert not backfill.visit(checkpoint, Reading(sizing=ReadSize(224))), "no recipe, so nothing re-derived"
