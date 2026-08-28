"""Serving the raptor member on a Kaggle split, which our own runner cannot do.

`KernelRun` and `Submission` are bound to OUR decode: a `SlotPipeline`, a member manifest, a scheme
and an `Ensemble` that rank-combines members sharing that scheme. The raptor member shares none of it
— it carries its own five-slot volume, its own 42-window evaluation and its own head — so routing it
through that path would mean teaching every one of those objects a second reading. This is the
smaller thing: the ten lines between a study identifier and a row of twelve probabilities.

What it does NOT duplicate is the refusal. `Submission.check` is the rule that a table missing studies
or constant in all twelve columns is not an answer, and it is a static method precisely so a second
producer of submissions is held to it without a second copy of it.

ONE STUDY MUST NOT COST THE ENTRY. A study that cannot be read contributes a row of 0.5 and a logged
traceback, the same asymmetry `Submission.decoded` names: across the three visible studies a decode
failure never fires, and across a hidden test set of thousands it is close to certain.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from core.findings import STUDY, TARGETS
from core.mounts import Mounts
from kneemri.evaluation.raptor_feed import RaptorFeed, Records
from kneemri.evaluation.raptor_model import RaptorMember
from kneemri.kaggle.submission import Split, Submission

logger = logging.getLogger(__name__)

WORKING = Path("/kaggle/working")

FALLBACK_PROBABILITY = 0.5
"""What an unreadable study contributes: one weak row, never a missing one and never a crash."""

DEFAULT_WEIGHTS = "raptor_ft_coatnet_v4_full_swa.pt"
"""The arm this entry serves, mounted as the public CC0-1.0 dataset it already is."""

PROGRESS_EVERY = 100


@dataclass(frozen=True)
class RaptorEntry:
    """One raptor submission: where the split's tables and pixels are, and where the CSV goes."""

    competition: Path
    working: Path = WORKING
    split: Split = field(default_factory=Split)

    @staticmethod
    def weights(filename: str = DEFAULT_WEIGHTS, mounts: Mounts | None = None) -> Path:
        """The mounted checkpoint, found by the same pruned walk everything else on a kernel uses."""
        finder = mounts or Mounts()
        return finder.one(filename, f"the raptor checkpoint {filename}") / filename

    def readings(
        self, identifiers: list[str], records: Records, members: Mapping[str, RaptorMember], device: str
    ) -> dict[str, NDArray[np.float64]]:
        """Score every study through EVERY member, decoding each study exactly once.

        The decode is the cost: a pass over our corpus measured the card at 0% while pydicom worked, so
        thirteen members scored one at a time would pay for thirteen decodes of the same pixels. The
        members share the volume instead, which turns a fleet pass from thirteen passes into one.

        A failed DECODE costs one row in every member's array; a failed FORWARD costs one row in that
        member's alone. Keeping the two apart matters here in a way it does not for a single member: a
        checkpoint that cannot run must not silently hand the other twelve a corpus full of 0.5s.
        """
        feed = RaptorFeed(self.split.series(self.competition))
        rows = {name: np.full((len(identifiers), len(TARGETS)), FALLBACK_PROBABILITY) for name in members}
        for index, study in enumerate(identifiers):
            try:
                volume, mask = feed.study(study, records)
            # Broad on purpose, and logged with its traceback: pydicom, the filesystem and the slot
            # match each fail in their own way, and none of them may cost the other studies.
            except Exception:
                logger.exception("study %s could not be decoded — contributing %s", study, FALLBACK_PROBABILITY)
                continue
            for name, member in members.items():
                try:
                    rows[name][index] = member.ordered(volume, mask, device)
                # A member's failure is that member's alone — the other twelve keep their reading.
                except Exception:
                    logger.exception("member %s could not score study %s", name, study)
            if (index + 1) % PROGRESS_EVERY == 0:
                logger.info("scored %d of %d studies", index + 1, len(identifiers))
        return rows

    def predictions(
        self, identifiers: list[str], records: Records, member: RaptorMember, device: str
    ) -> NDArray[np.float64]:
        """Score every study, in `TARGETS` column order, a failed decode costing one row and no more."""
        return self.readings(identifiers, records, {DEFAULT_WEIGHTS: member}, device)[DEFAULT_WEIGHTS]

    def table(self, identifiers: list[str], probabilities: NDArray[np.float64]) -> pd.DataFrame:
        """The wide submission frame, held to the same refusal every other entry is held to."""
        frame = pd.DataFrame(probabilities, columns=list(TARGETS))
        frame.insert(0, STUDY, identifiers)
        Submission.check(frame, identifiers)
        return frame

    def run(self, member: RaptorMember, device: str) -> pd.DataFrame:
        """Read the split's tables, score it, and write `submission.csv` — or write nothing at all."""
        test = pd.read_csv(self.competition / f"{self.split.name}.csv")
        identifiers = [str(study) for study in test[STUDY].tolist()]
        records = RaptorFeed.records(self.competition / f"{self.split.name}_series.csv")
        logger.info("%d studies, %d with series rows", len(identifiers), len(records))

        frame = self.table(identifiers, self.predictions(identifiers, records, member, device))
        frame.to_csv(self.working / "submission.csv", index=False)
        logger.info("wrote %d rows", len(frame))
        return frame
