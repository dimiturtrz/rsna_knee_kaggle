"""The epoch loop, on a corpus small enough that every epoch's decision is checkable by hand.

These are the behaviours that decide what a run PRODUCES, and none of them was reachable while the
loop lived inside `train.main`: it needs a parsed command line, a mapped cache, a downloaded encoder
and a data root before it reaches the first epoch.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from kneemri.corpus.study_cache import CachedCorpus
from kneemri.findings import TARGETS
from kneemri.labels.targets import Targets
from kneemri.models.backbone import Encoder
from kneemri.models.model import StudyModel
from kneemri.runs.member_record import MemberRecord
from kneemri.training.fitting import Fitted, Fitting, Fold, RunPlan
from kneemri.training.trainer import Trainer

from ..models.model import DIM, SLOTS, _FakeBackbone
from ..runs.member_record import OPTIONS

SEED = 2026
STUDIES = 8
BATCH = 4


def _fold(*, holdout: bool = True) -> Fold:
    """A corpus whose holdout is scoreable — the labels must differ, or every AUC comes back NaN."""
    generator = np.random.default_rng(SEED)
    corpus = CachedCorpus(
        generator.integers(0, 256, (STUDIES, SLOTS, 6, 14, 14), dtype=np.uint8),
        np.ones((STUDIES, SLOTS), dtype=np.float32),
    )
    scores = np.zeros((STUDIES, len(TARGETS)), dtype=np.float32)
    scores[STUDIES // 2 :, :] = 1.0
    targets = Targets(scores, np.ones_like(scores), [f"s{index}" for index in range(STUDIES)])
    positions = np.arange(STUDIES, dtype=np.int64)
    return Fold(
        corpus=corpus,
        targets=targets,
        evaluation=targets,
        training=positions,
        holdout=positions if holdout else np.array([], dtype=np.int64),
    )


def _fitting(destination: Path, epochs: int = 1, patience: int = 0) -> Fitting:
    torch.manual_seed(SEED)
    model = StudyModel(_FakeBackbone(), DIM, SLOTS)
    trainer = Trainer(model, torch.device("cpu"))
    encoder = Encoder(variant="small", repository="facebook/dinov2-small", dim=DIM)
    record = MemberRecord(OPTIONS, encoder, model, torch.device("cpu"), [f"S{i}" for i in range(SLOTS)])
    plan = RunPlan(destination=destination, epochs=epochs, batch=BATCH, patience=patience, seed=SEED)
    return Fitting(trainer, record, plan)


def test_a_run_states_its_recipe_before_it_has_a_single_score(tmp_path: Path):
    """Built, not fitted: a run killed in its first epoch still leaves a file saying what it tried.

    Written where the metrics file is created rather than at the first scored epoch, because a metrics
    file with no recipe is exactly the state that lets an unmatched arm read as a matched control.
    """
    fitting = _fitting(tmp_path / "member.pt")

    assert fitting.metrics.recipe() == {name: str(value) for name, value in vars(OPTIONS).items()}
    assert not fitting.metrics.rows(), "and it is not an epoch"


def test_scored():
    """A fold with nothing held back cannot rank its epochs by a score, and must say so."""
    assert _fold().scored
    assert not _fold(holdout=False).scored


def test_ranked(tmp_path: Path):
    """The number an epoch is ranked by: its holdout mean, or the epoch index when there is none."""
    fitting = _fitting(tmp_path / "member.pt")

    scored, per_label, predicted = fitting.ranked(_fold(), epoch=0)
    assert 0.0 <= scored <= 1.0
    assert set(per_label) == set(TARGETS)
    assert predicted is not None

    unscored, empty, nothing = fitting.ranked(_fold(holdout=False), epoch=4)
    assert unscored == 4.0, "the epoch index stands in, so keep-the-best keeps the last state"
    assert not empty
    assert nothing is None


def test_keep(tmp_path: Path):
    """Keeping an epoch clones its weights and puts them on disk before the next one runs."""
    destination = tmp_path / "member.pt"
    fitting = _fitting(destination)
    fold = _fold()
    predicted = np.zeros((STUDIES, len(TARGETS)))

    kept = fitting.keep(fold, epoch=2, ranked=0.84, predicted=predicted)

    assert kept.epoch == 2
    assert kept.holdout == pytest.approx(0.84)
    assert kept.state["head.out.weight"].device.type == "cpu", "a kept state must outlive the card"
    assert destination.with_suffix(".partial").is_file()


def test_run(tmp_path: Path):
    """The kept epoch is the highest-scoring one, and its own score travels with it."""
    fitting = _fitting(tmp_path / "member.pt", epochs=3)
    ranked = iter([0.70, 0.90, 0.60])
    fitting.ranked = lambda fold, epoch: (next(ranked), {TARGETS[0]: 0.9}, None)  # type: ignore[method-assign]

    kept = fitting.run(_fold())

    assert kept.epoch == 1, "the last epoch is not the best one, and the best one is what is kept"
    assert kept.holdout == pytest.approx(0.90)


def test_epochs(tmp_path: Path):
    """The loop itself, without the wakelock `run` wraps it in — same answer, so the hold is a property
    of the RUN and not of what the epochs compute."""
    fitting = _fitting(tmp_path / "member.pt", epochs=2)
    ranked = iter([0.80, 0.60])
    fitting.ranked = lambda fold, epoch: (next(ranked), {TARGETS[0]: 0.8}, None)  # type: ignore[method-assign]

    kept = fitting.epochs(_fold())

    assert kept.epoch == 0
    assert kept.holdout == pytest.approx(0.80)


def test_the_epoch_line_states_its_denominator_when_a_label_is_undefined(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """A mean over eleven labels and a mean over twelve are not comparable, and read identically.

    `nanmean` drops a finding whose holdout holds only one class — which is normal on a smoke, on a
    confidence-filtered ruler, and on any small split — so the headline silently changes denominator
    while keeping its wording. Two runs' numbers then get subtracted from each other, which is the
    denominator error this project has already paid for three times in one session.

    Said ONLY when it differs from the full set: an every-epoch caveat is one nobody reads by epoch 3.
    """
    fitting = _fitting(tmp_path / "member.pt")
    undefined = dict.fromkeys(TARGETS, 0.9)
    undefined[TARGETS[0]] = float("nan")
    fitting.ranked = lambda fold, epoch: (0.9, undefined, None)  # type: ignore[method-assign]

    with caplog.at_level("INFO"):
        fitting.run(_fold())
    assert f"over {len(TARGETS) - 1} of {len(TARGETS)} labels" in caplog.text

    caplog.clear()
    fitting = _fitting(tmp_path / "full.pt")
    fitting.ranked = lambda fold, epoch: (0.9, dict.fromkeys(TARGETS, 0.9), None)  # type: ignore[method-assign]
    with caplog.at_level("INFO"):
        fitting.run(_fold())
    assert "labels" not in caplog.text, "the competition's own denominator needs no caveat"


def test_the_improved_epoch_reaches_the_disk_before_the_next_one_runs(tmp_path: Path):
    """A killed run must leave a member that is submittable, not merely inspectable."""
    destination = tmp_path / "member.pt"

    _fitting(destination, epochs=1).run(_fold())

    saved = torch.load(destination.with_suffix(".partial"), map_location="cpu", weights_only=False)
    assert saved["fingerprint"], "a partial with no fingerprint is refused by MemberPackage"
    assert saved["config"]["crop_mm"], "and a partial with no reading cannot be served as it was fitted"
    assert saved["config"]["witness"]["studies"], "the witness travels with the weights it came from"


def test_an_unscored_run_reports_no_number_and_keeps_the_last_epoch(tmp_path: Path):
    """`--fold -1` holds nothing back, so it ranks by epoch INDEX and cannot report a score.

    Recording the standing-in index would have a final-fit member advertise `holdout 6.0`, beating
    every real member in `strongest()` and reading as a score in both the manifest and the ledger.
    """
    fitting = _fitting(tmp_path / "final.pt", epochs=3)
    kept = fitting.run(_fold(holdout=False))

    assert np.isnan(kept.holdout), "a run with no holdout cannot report one by construction"
    assert kept.epoch == 2, "ranking by epoch number makes keep-the-best keep the LAST state"
    assert kept.witness is None, "there is no held-out study to witness"
    # The LEDGER too, not only the checkpoint: the epoch index used to land in the `auc` column, where
    # `ledger_check.best` reads the maximum and would quote a twelve-epoch final fit as scoring 11.0.
    assert all(np.isnan(row["auc"]) for row in fitting.metrics.rows()), "three epochs, and not one score"


def test_patience_ends_the_loop_after_stale_epochs(tmp_path: Path):
    """Two epochs without improvement ends the run, and the best epoch is untouched by stopping."""
    fitting = _fitting(tmp_path / "member.pt", epochs=6, patience=2)
    ranked = iter([0.90, 0.10, 0.10, 0.10, 0.10, 0.10])
    fitting.ranked = lambda fold, epoch: (next(ranked), {}, None)  # type: ignore[method-assign]

    kept = fitting.run(_fold())

    assert kept.epoch == 0
    assert len(fitting.metrics.rows()) == 3, "epoch 0, then two stale ones, then stop"


def test_the_full_schedule_runs_when_patience_is_off(tmp_path: Path):
    """The default must not stop early, or every arm already in the ledger changes meaning."""
    fitting = _fitting(tmp_path / "member.pt", epochs=3)
    ranked = iter([0.90, 0.10, 0.10])
    fitting.ranked = lambda fold, epoch: (next(ranked), {}, None)  # type: ignore[method-assign]

    fitting.run(_fold())

    assert len(fitting.metrics.rows()) == 3


def test_the_epoch_is_ranked_under_the_weights_it_would_save(tmp_path: Path):
    """With an EMA attached, scoring the LIVE weights ranks epochs for a model never written."""
    fitting = _fitting(tmp_path / "member.pt", epochs=1)
    fitting.trainer.exponential_average()
    seen: list[torch.Tensor] = []

    def ranked(fold: Fold, epoch: int) -> tuple[float, dict[str, float], None]:
        seen.append(fitting.trainer.model.head.out.weight.detach().clone())
        return 0.9, {}, None

    fitting.ranked = ranked  # type: ignore[method-assign]
    kept = fitting.run(_fold())

    assert not torch.allclose(seen[0], fitting.trainer.model.head.out.weight), (
        "the score was measured on the live weights rather than the averaged ones the run saves"
    )
    assert torch.allclose(kept.state["head.out.weight"], seen[0]), "and the kept state is the scored one"


def test_a_run_that_keeps_nothing_raises_rather_than_returning_empty_weights(tmp_path: Path):
    """Zero epochs produces no checkpoint, and must say so instead of writing an empty one."""
    fitting = _fitting(tmp_path / "member.pt", epochs=0)

    with pytest.raises(ValueError, match="no epoch was kept"):
        fitting.run(_fold())


def test_a_scored_run_witnesses_what_the_training_path_predicted(tmp_path: Path):
    """The witness is the only record of how INFERENCE must read the member it was saved with."""
    kept = _fitting(tmp_path / "member.pt", epochs=1).run(_fold())

    assert kept.witness is not None
    assert kept.witness["studies"], "named studies, so another process can re-derive them"
    assert not np.isnan(kept.holdout)


def test_fitted_carries_the_four_parts_the_final_save_writes():
    """One object, because a checkpoint's weights, score, epoch and witness are written together."""
    kept = Fitted(state={"w": torch.zeros(1)}, holdout=0.84, epoch=6, witness={"studies": ["s0"]})

    assert kept.epoch == 6
    assert kept.witness == {"studies": ["s0"]}


def test_a_scored_run_keeps_every_holdout_prediction_not_only_the_witnessed_eight(tmp_path: Path):
    """The array the ledger's per-label numbers were read off, kept whole beside the checkpoint.

    Without it an arm can state a per-label delta and not its own interval, which is how knee-3vh ended
    up borrowing a floor measured on someone else's seed twins.
    """
    fitting = _fitting(tmp_path / "member.pt", epochs=1)
    kept = fitting.run(_fold())

    reading = fitting.predictions.read()
    assert reading is not None
    assert reading.studies == [f"s{index}" for index in range(STUDIES)]
    assert reading.predictions.shape == (STUDIES, len(TARGETS))
    assert reading.epoch == kept.epoch, "the kept epoch's, not the last one's"


def test_an_unscored_run_records_no_predictions_because_it_made_none(tmp_path: Path):
    """A final fit holds nothing back, so there is no per-study record and no pairing to be had."""
    fitting = _fitting(tmp_path / "final.pt", epochs=1)
    fitting.run(_fold(holdout=False))

    assert not fitting.predictions.exists()
