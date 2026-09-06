"""Average one foreign family over the BAND it is served at, which is the only serving knob it exposes.

`python -m kneemri.evaluation.band_tta`.

WHY A BAND AVERAGE IS THE TTA HERE. Their `Net` takes `N_SLICE` as a constructor argument read from the
checkpoint cfg, so a served count other than 16 builds a model their state dict does not fit: the slice
COUNT axis does not exist for this family. The band does — and the coarse sweep reads non-monotone in
coverage (0.08-0.92 at 0.8582 and 0.16-0.84 at 0.8601 both below 0.12-0.88 at 0.8812, with their own
training band 0.20-0.80 lowest of all at 0.8492), so what the band moves is WHERE the sixteen slice
positions land, not how much anatomy they cover. A model sensitive to its grid offset is a model that
gains from averaging over offsets. That is test-time augmentation on the one handle we have.

RANK SPACE, NOT SCORE SPACE. Two bands are two different reads of the same weights and their score
scales need not agree; `Vote.ranks` is the project's one rank definition and this calls it rather than
writing a second. Ranking is per (band, label) and the mean is over bands, which leaves the output on
the same [0, 1] per-label scale each input had.

WHAT IT COSTS IF IT WORKS. Foreign serving time scales with the number of bands — about 1.1 h each on
the hidden split — so a three-band average is 3.3 h. Beside the champion's two read groups that fits
inside the 9 h limit ONLY with `raptor_v4_swa` out of the entry, and raptor is part of the cross-family
diversity the foreign seat is bought for. So a band-TTA seat has to beat the single-band seat by enough
to pay for raptor's own seat, which is why the pre-registered bar is on the SEAT delta over the served
base of eight and not on the solo.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from kneemri.evaluation.foreign_price import ForeignPrice
from kneemri.inference.vote import Vote
from kneemri.runs.holdout_record import FOREIGN, RULER_TOLERANCE, UNKNOWN_EPOCH

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BandTTA:
    """Several servings of one family, each at its own band, over the SAME studies and the same truth."""

    bands: tuple[Float[NDArray[np.float32], "studies labels"], ...]
    truth: Float[NDArray[np.float32], "studies labels"]

    @staticmethod
    def aligned(records: list[Path]) -> "BandTTA":
        """Load the emitted fold-matched columns, refusing any two that are not the same reading.

        Study ORDER and truth are both contracts here, not conveniences: a rank-mean over two arrays whose
        rows are different studies is arithmetic on two questions, and it would produce a number rather
        than an error. The truth check is the same one `HoldoutRecord.aligned` makes, at the same
        tolerance, because these arrays came through the same float32 round trip.
        """
        loaded = [np.load(record, allow_pickle=True) for record in records]
        studies = [str(one) for one in loaded[0]["studies"]]
        truth = loaded[0]["truth"].astype(np.float32)
        for record, served in zip(records, loaded, strict=True):
            if [str(one) for one in served["studies"]] != studies:
                message = f"{record} is not the same studies, in order, as {records[0]}"
                raise ValueError(message)
            if not np.allclose(served["truth"].astype(np.float32), truth, atol=RULER_TOLERANCE, equal_nan=True):
                message = f"{record} was scored against a different truth table than {records[0]}"
                raise ValueError(message)
        return BandTTA(tuple(served["predictions"].astype(np.float32) for served in loaded), truth)

    def combined(self) -> Float[NDArray[np.float32], "studies labels"]:
        """The per-label rank of every band, averaged over bands — one column, on the same rank scale."""
        ranked = [Vote.ranks(band.astype(np.float64)) for band in self.bands]
        return np.mean(ranked, axis=0).astype(np.float32)

    def lines(self) -> list[str]:
        """Each band's own macro AUC and the average's, so the reading carries what it is compared to."""
        solo = [f"band {index}: {ForeignPrice.macro(band, self.truth):.4f}" for index, band in enumerate(self.bands)]
        return [
            *solo,
            f"rank-mean over {len(self.bands)} bands: {ForeignPrice.macro(self.combined(), self.truth):.4f}",
            f"over {len(self.truth)} studies, every one fold-matched in each band",
        ]


def main() -> None:
    """Rank-mean several band servings of one foreign family into one column, and say what it is worth."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument(
        "--records", type=Path, nargs="+", required=True, help="the holdout npz files foreign_price --emit wrote"
    )
    command.parser.add_argument("--out", type=Path, default=None, help="write the averaged column as a seat record")
    arguments = command.parsed()

    tta = BandTTA.aligned(arguments.records)
    if arguments.out:
        np.savez(
            arguments.out,
            studies=np.load(arguments.records[0], allow_pickle=True)["studies"],
            predictions=tta.combined(),
            truth=tta.truth,
            epoch=np.array(UNKNOWN_EPOCH),
            source=np.array(FOREIGN),
        )
        logger.info("wrote %s", arguments.out)
    command.report(tta.lines())


if __name__ == "__main__":
    main()
