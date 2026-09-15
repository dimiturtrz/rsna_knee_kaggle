"""Serve a slot-dropout member over its own TRAINING masks: `python -m kneemri.evaluation.slotdrop_tta`.

`python -m kneemri.evaluation.slotdrop_tta --holdouts <dir> --member <stem> --masks K --rate 0.4
[--studies N] [--device cpu|cuda] [--out npz]`

WHY. The drop40 seats were fitted with whole acquisition slots withheld at 0.40 and are served with every
slot present, so the served input is the one corner of the training distribution each study reaches
least often. knee-burv's proposal is to serve it over the distribution it was fitted on instead: K
seeded draws of the very mask `SlotDropout.dropped` applies in training, averaged (knee-ilc58).

THE MASK IS THE TRAINING RULE, CALLED, NOT COPIED. An absent slot is never withheld and no study is
emptied, because both are properties of `SlotDropout.dropped` and this module only hands it a rate.

PROBABILITY SPACE. `HoldoutPredictions.predict` returns the sigmoid averaged over the member's own windows
and views, and the draws are averaged in that same space — one more axis of the average the member is
already served under, not a rank vote. AUC is rank-invariant per draw, but a mean of probabilities is not
a mean of ranks, and probabilities are what the ensemble downstream of a seat consumes.

THE PLAIN READING IS A CHECK, NOT A CONTROL. It must reproduce the member's `.holdout.npz` within
`RULER_TOLERANCE`; when it does not (a CPU float32 read of a member fitted in half precision will not)
the gap is printed so the dropout delta is read against the right baseline — the plain column here, never
the stored one. A mask is drawn per batch of `BATCH` studies, so draws are reproducible for a seed and
that batch size.
"""

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import override

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.data.corpus.study_cache import CachedCorpus
from kneemri.evaluation.ledger75 import LedgerSeventyFive
from kneemri.inference.holdout_predictions import HoldoutPredictions
from kneemri.inference.holdout_slab import HoldoutSlab
from kneemri.inference.member_readings import MemberReadings
from kneemri.inference.members import Member
from kneemri.models.backbone import ENCODERS
from kneemri.models.slot_dropout import SlotDropout
from kneemri.runs.holdout_record import DERIVED, RULER_TOLERANCE, SUFFIX, HoldoutReading

logger = logging.getLogger(__name__)


class SlotDropSlab(HoldoutSlab):
    """A holdout slab whose every batch carries one seeded draw of the training slot-dropout mask."""

    def __init__(self, corpus: CachedCorpus, holdout: NDArray[np.int64], rate: float, seed: int) -> None:
        super().__init__(corpus, holdout)
        self.dropout = SlotDropout(rate)
        self.seed = seed

    @override
    def batches(self, size: int) -> Iterator[tuple[NDArray[np.uint8], NDArray[np.float32]]]:
        """The plain batches in holdout order, each masked by the SAME rule training applies."""
        generator = torch.Generator().manual_seed(self.seed)
        for images, mask in super().batches(size):
            dropped, presence = self.dropout.dropped(torch.from_numpy(images), torch.from_numpy(mask), generator)
            yield dropped.numpy(), presence.numpy()


@dataclass(frozen=True)
class SlotDropReading:
    """One member's stored, plain and per-draw probabilities over the same studies and the same truth."""

    stored: HoldoutReading
    plain: Float[NDArray[np.float64], "studies labels"]
    draws: tuple[Float[NDArray[np.float64], "studies labels"], ...]

    @staticmethod
    def positions(identifiers: list[str], studies: list[str]) -> NDArray[np.int64]:
        """Where each recorded study sits in the cache, in the RECORD's order; a missing one is refused."""
        rows = {name: position for position, name in enumerate(identifiers)}
        missing = [name for name in studies if name not in rows]
        if missing:
            message = f"{len(missing)} recorded studies are not in the member's cache, first {missing[0]}"
            raise ValueError(message)
        return np.array([rows[name] for name in studies], dtype=np.int64)

    def averaged(self) -> Float[NDArray[np.float64], "studies labels"]:
        """The draws' mean, in the probability space the member is served in."""
        return np.mean(self.draws, axis=0)

    def drift(self) -> float:
        """The largest gap between the plain read and the stored record over the rows scored."""
        rows = len(self.plain)
        return float(np.max(np.abs(self.plain - self.stored.predictions[:rows])))

    def lines(self) -> list[str]:
        """Plain and dropout-averaged under both rulers, and whether plain reproduces the record."""
        ruler = LedgerSeventyFive()
        truth = self.stored.truth[: len(self.plain)]
        lines = [f"{'reading':28s} {'LEDGER':>8s} {'LEDGER75':>9s}"]
        for name, predictions in (("stored", self.stored.predictions[: len(truth)]), ("plain", self.plain)):
            lines.append(self._row(ruler, name, predictions, truth))
        lines.append(self._row(ruler, f"slotdrop mean of {len(self.draws)}", self.averaged(), truth))
        drift = self.drift()
        verdict = "MATCHES" if drift <= RULER_TOLERANCE else "DIFFERS FROM"
        lines.append(f"plain {verdict} the stored record: max |gap| {drift:.2e} (tolerance {RULER_TOLERANCE:.0e})")
        lines.append(f"over {len(truth)} studies of {len(self.stored.studies)}")
        return lines

    def _row(
        self,
        ruler: LedgerSeventyFive,
        name: str,
        predictions: Float[NDArray[np.float64], "studies labels"],
        truth: Float[NDArray[np.float32], "studies labels"],
    ) -> str:
        reading = HoldoutReading(self.stored.studies[: len(truth)], predictions, truth, self.stored.epoch)
        midpoint = float(np.nanmean(list(ruler.midpoint.per_label(predictions, truth).values())))
        return f"{name:28s} {midpoint:8.4f} {ruler.macro(reading):9.4f}"


def main() -> None:
    """Read one member plain and over K training masks, on the studies and truth its record names."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--holdouts", type=Path, required=True, help="directory of checkpoints and records")
    command.parser.add_argument("--member", required=True, help="checkpoint stem")
    command.parser.add_argument("--masks", type=int, required=True, help="seeded dropout draws to average")
    command.parser.add_argument("--rate", type=float, default=0.4, help="the member's training slot dropout")
    command.parser.add_argument("--studies", type=int, default=0, help="first N recorded studies; 0 is all")
    command.parser.add_argument("--device", default="cpu")
    command.parser.add_argument("--out", type=Path, default=None, help="write the averaged column as a record")
    arguments = command.parsed()

    checkpoint = arguments.holdouts / f"{arguments.member}.pt"
    stored = LedgerSeventyFive().reading(arguments.holdouts / f"{arguments.member}{SUFFIX}")
    studies = stored.studies[: arguments.studies or None]
    member = Member.from_checkpoint(checkpoint)
    readings = MemberReadings([member], command.rooted(arguments).processed())
    corpus, identifiers = readings.reference()
    holdout = SlotDropReading.positions(identifiers, studies)

    predictions = HoldoutPredictions(torch.device(arguments.device))
    encoder = ENCODERS[member.variant]
    model = predictions.load(checkpoint, encoder.load(), encoder.dim, member.slot_count)
    window = member.window(corpus.slices, overlap=False)
    began = time.perf_counter()
    plain = predictions.predict(model, HoldoutSlab(corpus, holdout), window, member.precision)
    logger.info("plain read: %.3f s a study on %s", (time.perf_counter() - began) / len(holdout), arguments.device)
    draws = []
    for seed in range(arguments.masks):
        began = time.perf_counter()
        draws.append(
            predictions.predict(model, SlotDropSlab(corpus, holdout, arguments.rate, seed), window, member.precision)
        )
        logger.info(
            "draw %d/%d: %.3f s a study", seed + 1, arguments.masks, (time.perf_counter() - began) / len(holdout)
        )
    reading = SlotDropReading(stored, plain, tuple(draws))
    if arguments.out:
        np.savez(
            arguments.out,
            studies=np.array(studies),
            predictions=reading.averaged().astype(np.float32),
            truth=stored.truth[: len(studies)],
            epoch=np.array(stored.epoch),
            source=np.array(DERIVED),
        )
        logger.info("wrote %s", arguments.out)
    command.report(reading.lines())


if __name__ == "__main__":
    main()
