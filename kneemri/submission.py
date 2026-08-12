"""Predicting a whole split and writing the submission — the part both kernels do identically.

The two kernels differ in exactly one thing: which weights package they mount. Everything after that
— loading members, decoding studies in chunks, predicting, rank-combining, writing the CSV — was
duplicated line for line between them, which is two places to fix a bug found once and two places for
them to drift silently apart.

What CANNOT move here is the bootstrap: locating the mounted kit and installing its wheels has to
happen before this module can be imported at all. That part stays in the kernel, and it is the only
part that should.
"""

import logging
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor

from kneemri.ensemble import Ensemble
from kneemri.members import Member
from kneemri.model import TARGETS
from kneemri.slot_reader import SlotReader
from kneemri.study import SlotPipeline, Study

logger = logging.getLogger(__name__)

STUDY_CHUNK = 48
"""Studies decoded before a prediction pass. Bounds peak memory: a chunk of 48 at twelve slices and
336 pixels is about 390 MB, against a Kaggle host with 32 GB and four CPUs."""

DECODE_THREADS = 8
"""Above the four available cores on purpose: decoding is dominated by waiting on the mount."""


class Submission:
    """Runs an ensemble over a split's studies and writes `submission.csv`."""

    def __init__(self, ensemble: Ensemble, competition: Path, working: Path, *, overlap: bool = True) -> None:
        self.ensemble = ensemble
        self.device = ensemble.device
        self.competition = competition
        self.working = working
        self.overlap = overlap

    def decode(
        self, identifiers: list[str], planes: dict[str, str], slices: int, members: list[Member]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode a chunk of studies into one batched bag and its presence mask.

        The pipeline is built at the MEMBERS' resolution rather than at `SlotReader`'s default, which
        was 336 whatever the members said. See `Ensemble.decoded_size`.
        """
        pipeline = SlotPipeline(
            scheme=self.ensemble.decoded_scheme(members),
            reader=SlotReader(image_size=self.ensemble.decoded_size(members)),
        )
        with ThreadPoolExecutor(max_workers=DECODE_THREADS) as pool:
            bags = list(
                pool.map(
                    lambda name: Study.from_directory(self.competition / "test_series" / name, planes, pipeline).bag(
                        slices
                    ),
                    identifiers,
                )
            )
        return torch.stack([bag for bag, _ in bags]), torch.stack([mask for _, mask in bags])

    def chunks(
        self, identifiers: list[str], planes: dict[str, str], slices: int, members: list[Member]
    ) -> Iterator[tuple[Tensor, Tensor]]:
        """Decoded chunks, yielded as they are read so prediction overlaps the next decode's I/O."""
        started = time.time()
        for begin in range(0, len(identifiers), STUDY_CHUNK):
            chunk = identifiers[begin : begin + STUDY_CHUNK]
            decode_started = time.time()
            images, mask = self.decode(chunk, planes, slices, members)
            logger.info(
                f"  studies {begin + 1}-{begin + len(chunk)}: decode {time.time() - decode_started:.0f}s, "
                f"elapsed {time.time() - started:.0f}s"
            )
            yield images.to(self.device), mask.to(self.device)

    def predict(self, identifiers: list[str], planes: dict[str, str]) -> NDArray[np.float64]:
        """Every member's predictions over every study, rank-combined into one table.

        The grouping and combining live on `Ensemble`, which knows which members can share pixels; this
        supplies the only thing it cannot know, which is how to turn a set of studies into chunks.
        """
        return self.ensemble.predict_groups(
            lambda members, slices: self.chunks(identifiers, planes, slices, members), overlap=self.overlap
        )

    def run(self) -> pd.DataFrame:
        """The whole submission: decode, predict, write — and NOTHING if any of it fails.

        THERE WAS A FALLBACK HERE AND IT COST A SUBMISSION. It copied `sample_submission.csv` out
        first, "so a crash later still leaves a scoreable file", and on 2026-08-12 that is exactly what
        it left: fusion v2 did not finish on the hidden test set, the sample's constant values were
        scored, and the leaderboard reported 0.500 — chance. A slot was spent to learn nothing, and the
        number looked like a model result rather than a crash.

        A kernel that fails must produce NO submission.csv. Kaggle then fails the submission itself,
        which costs no slot and says plainly that the run broke. A fallback converts a loud failure into
        a quiet wrong answer, and on a five-a-day budget that is the more expensive of the two.

        The slice depth comes from the members, not from a default. It used to be `slices: int = 12`,
        and every kernel called `run()` — so a member fitted on a deeper stack would have been scored on
        twelve slices without a word.
        """
        test = pd.read_csv(self.competition / "test.csv")
        series = pd.read_csv(self.competition / "test_series.csv")
        planes = series.set_index("SeriesInstanceUID")["Anatomical_Plane"].to_dict()
        identifiers = test["StudyInstanceUID"].tolist()
        logger.info(f"{len(identifiers)} studies, {len(series)} series")

        combined = self.predict(identifiers, planes)
        frame = pd.DataFrame(self.ensemble.submission(identifiers, combined))
        self.check(frame, identifiers)
        frame.to_csv(self.working / "submission.csv", index=False)
        logger.info(f"wrote {len(frame)} rows")
        return frame

    @staticmethod
    def check(frame: pd.DataFrame, identifiers: list[str]) -> None:
        """Refuse a table that cannot be a real answer, before it is written and scored.

        Two failures are indistinguishable from a working submission by eye and both score around
        chance: a table missing studies, and a table with no ordering in it at all.

        EVERY target constant, not any of them. A single flat column is possible on a small split
        where predictions tie — and the visible test set inside a kernel is three studies, so a
        per-column rule would fail the run that has to succeed before anything can be submitted. All
        twelve flat is not a tie, it is the sample submission, which is what the removed fallback wrote
        and what scored 0.500.
        """
        if len(frame) != len(identifiers):
            message = f"submission has {len(frame)} rows for {len(identifiers)} studies"
            raise ValueError(message)
        if len(frame) > 1 and all(frame[target].nunique() == 1 for target in TARGETS):
            message = "every target is constant — that scores as chance, not as a model"
            raise ValueError(message)
