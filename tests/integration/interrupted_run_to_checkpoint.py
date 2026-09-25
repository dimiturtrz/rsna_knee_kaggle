"""A killed run resumed lands exactly where the uninterrupted run would have.

This is the constraint knee-c317d names as the whole feature. `ResumeState` restores an optimiser, a
schedule, a gradient scaler and four random streams, and its unit tests prove each one survives the
write. None of them proves the thing the module exists for: that putting the trajectory back CONTINUES
the schedule rather than starting a new one. A resume that quietly drops any single stream still loads,
still logs "continuing the same schedule", and still produces a checkpoint — it is `WarmStart` under a
new name, and the arm it returns is no longer comparable to the control it was priced against.

The claim is made on a synthetic corpus on the CPU in seconds, because it asks nothing about the data:
an interrupted run either takes the same next step as an uninterrupted one or it does not, and a
fake backbone exposes that exactly as well as a real trunk would.
"""

from pathlib import Path

import numpy as np
import torch

from core.data.corpus.study_cache import CachedCorpus
from core.data.labels.targets import Targets
from core.findings import TARGETS
from kneemri.models.encoder import Encoder
from kneemri.models.model import StudyDesign, StudyModel
from kneemri.runs.member_record import MemberRecord
from kneemri.training.config_tree import TrainConfig
from kneemri.training.fitting import Fitted, Fitting, Fold, RunPlan
from kneemri.training.run_setup import RunSetup

from ..unit.kneemri.models.model import DIM, SLOTS, _FakeBackbone

SEED = 2026
STUDIES = 8
SLICES = 6
SIZE = 14
DEVICE = torch.device("cpu")
EPOCHS = 4
KILLED_AT = 2
"""Where the machine dies. Two epochs run, two are owed, so the resumed half is neither the whole run
nor a single step — a resume that restarted the schedule would still get the epoch COUNT right."""

WEIGHT_TOLERANCE = 1e-6
"""How far two float32 trajectories may differ before this is a dropped stream rather than rounding."""


def _fold() -> Fold:
    """A corpus whose two halves are separable, so the score moves and epochs are actually kept."""
    generator = np.random.default_rng(SEED)
    corpus = CachedCorpus(
        generator.integers(0, 256, (STUDIES, SLOTS, SLICES, SIZE, SIZE), dtype=np.uint8),
        np.ones((STUDIES, SLOTS), dtype=np.float32),
    )
    scores = np.zeros((STUDIES, len(TARGETS)), dtype=np.float32)
    scores[STUDIES // 2 :, :] = 1.0
    targets = Targets(scores, np.ones_like(scores), [f"s{index}" for index in range(STUDIES)])
    positions = np.arange(STUDIES, dtype=np.int64)
    return Fold(corpus=corpus, targets=targets, evaluation=targets, training=positions, holdout=positions)


def _fitted(destination: Path, epochs: int, *, resume: bool) -> Fitted:
    """Fit through the same setup `train.main` uses, so a policy it attaches is attached here too."""
    torch.manual_seed(SEED)
    design = StudyDesign(slice_pooling="max")
    model = StudyModel(_FakeBackbone(), DIM, SLOTS, design)
    options = TrainConfig().with_leaves(image_size=SIZE, slices=SLICES, whole_stack=True, slice_pooling="max")
    trainer = RunSetup(options).trainer(model, DEVICE, [], np.arange(STUDIES))
    record = MemberRecord(
        options.stamp,
        Encoder(variant="small", repository="facebook/dinov2-small", dim=DIM),
        model,
        DEVICE,
        [f"S{index}" for index in range(SLOTS)],
    )
    plan = RunPlan(destination=destination, epochs=epochs, batch=4, seed=SEED, resume=resume)
    return Fitting(trainer, record, plan).run(_fold())


def _difference(left: Fitted, right: Fitted) -> float:
    """The largest disagreement between two fitted states, over every parameter they share."""
    return max(float((left.state[name] - right.state[name]).abs().max()) for name in left.state)


def test_a_resumed_run_lands_where_the_uninterrupted_run_would_have(tmp_path: Path) -> None:
    """Kill at epoch two, resume, and finish on the same weights the straight run finished on."""
    straight = _fitted(tmp_path / "straight.pt", EPOCHS, resume=False)

    interrupted = tmp_path / "interrupted.pt"
    _fitted(interrupted, KILLED_AT, resume=False)
    resumed = _fitted(interrupted, EPOCHS, resume=True)

    assert _difference(straight, resumed) < WEIGHT_TOLERANCE, (
        "a resumed run followed a different trajectory than the uninterrupted one - a stream was dropped"
    )
    assert resumed.epoch == straight.epoch
    assert resumed.holdout == straight.holdout


def test_the_epochs_after_the_kill_actually_move_the_weights(tmp_path: Path) -> None:
    """The equality above is only worth its runtime if the owed epochs change anything - so show they do.

    Without this, a resume that silently ran zero further epochs would satisfy the test above whenever
    the schedule had already converged, and the check would pass by measuring nothing.
    """
    straight = _fitted(tmp_path / "straight.pt", EPOCHS, resume=False)
    killed = _fitted(tmp_path / "killed.pt", KILLED_AT, resume=False)

    assert _difference(straight, killed) > WEIGHT_TOLERANCE, (
        "four epochs landed on the same weights as two, so this corpus cannot detect a restarted schedule"
    )
