"""The producer's testable half is the ORDERING — everything else is a card, a corpus and an encoder."""

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from core.data.corpus.folds import ReportFolds
from core.paths import DataRoot
from kneemri.inference.members import Member, MemberConfig
from kneemri.inference.teacher_readings import TeacherReadings

SLOTS = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"]


def test_teach(tmp_path: Path) -> None:
    """An empty member list is refused BEFORE the thirty-six minutes, not discovered after them.

    The rest of `teach` is a card, a corpus and an encoder, so this is the one clause a unit can reach —
    and it is the clause worth reaching: a mistyped checkpoint glob expands to nothing in a shell, and
    without the guard the pass would run to completion and write a teacher file ordering no studies.
    """
    readings = TeacherReadings(DataRoot(tmp_path), torch.device("cpu"))

    with pytest.raises(ValueError, match="no checkpoints to teach from"):
        readings.teach([], ["s0"], ReportFolds(5, 2026), 0)


def test_ordered() -> None:
    member = np.array([[0.9], [0.1], [0.5]])
    assert TeacherReadings.ordered([member])[:, 0].tolist() == [1.0, 0.0, 0.5]


def test_ordered_is_invariant_to_a_monotone_rewrite_of_a_member() -> None:
    """The claim the module's docstring makes about calibration, as an executable one."""
    member = np.array([[0.9], [0.1], [0.5]])
    assert TeacherReadings.ordered([member]).tolist() == TeacherReadings.ordered([member**3]).tolist()


def test_ordered_averages_the_members_ranks() -> None:
    """A member the others outvote moves the ordering without deciding it."""
    agreeing = np.array([[0.9], [0.1], [0.5]])
    dissenting = np.array([[0.1], [0.9], [0.5]])
    combined = TeacherReadings.ordered([agreeing, agreeing, dissenting])[:, 0]
    assert combined.tolist() == [1.0, 0.0, 0.5], "two votes for the first study beat one against"


def test_ordered_spans_the_unit_interval_for_every_label() -> None:
    scores = np.random.default_rng(0).random((7, 4))
    ordered = TeacherReadings.ordered([scores])
    assert ordered.min(axis=0).tolist() == [0.0] * 4
    assert ordered.max(axis=0).tolist() == [1.0] * 4


def test_over_corpus() -> None:
    """Both halves of the re-point land: the geometry through the regroup, the corpus through the scheme.

    They are separate transforms because each already has an owner, and a member re-pointed at only one
    of them is the failure with no symptom — the right shape read out of the wrong cache.
    """
    member = Member(
        identifier="keyless",
        file=Path("m_keyless.pt"),
        fold=0,
        config=cast(MemberConfig, {"img": 336, "group": 3, "slots": list(SLOTS), "slices": 20}),
        fingerprint=np.zeros((2, 12), np.float32),
        holdout=0.1,
        annot=0.1,
    )
    (repointed,) = TeacherReadings.over_corpus([member], "merged3", 12, [0.2, 0.8])

    assert repointed.scheme == "merged3"
    assert repointed.slices == 12
    assert repointed.band.recorded() == [0.2, 0.8]
    assert member.slices == 20, "the fitted member is untouched"
