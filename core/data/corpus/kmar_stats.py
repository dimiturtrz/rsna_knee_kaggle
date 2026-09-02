"""Measure what motion actually does to a knee: `python -m core.data.corpus.kmar_stats --root PATH`.

THE NUMBER A MOTION-ARTIFACT OPERATOR HAS TO REPRODUCE, taken off real pairs rather than chosen. This
runs before any operator exists, and that order is the whole point — a severity knob set to taste is a
fitted constant with no measurement behind it, and it would be indistinguishable from one that adds
plain noise. See `kmar_corpus` for why the pairs are the only honest use of this corpus.

THREE STATISTICS, and the third is the one that discriminates:

  * `residual` — mean absolute difference between the motion and motion-free volumes, both scaled by the
    CLEAN volume's own percentiles so the difference is not renormalised away. This is how big the
    perturbation is, and any transform at all can be tuned to match it.
  * `residual_p99` — the tail. Motion is not uniform over an image; a transform matching the mean with a
    flat perturbation would sit far below this.
  * `background_share` — the fraction of the total residual landing OUTSIDE the knee. Motion ghosting
    deposits replicas of the object into empty space along the phase-encode axis, and blur and noise do
    not. An operator that matches the first two numbers and misses this one is not simulating motion.

It reports per plane, because ghosting runs along the phase-encode direction and the three planes are
acquired with different ones — a single pooled number would average over the axis the effect has.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from core import cli
from core.data.corpus.kmar_corpus import BACKGROUND_PERCENTILE, KmarCorpus, KmarPair

logger = logging.getLogger(__name__)

SCALE_PERCENTILES = (1.0, 99.0)
"""The window both halves of a pair are scaled by, taken from the CLEAN volume.

From the clean one, not from each separately. Scaling each by its own percentiles would absorb exactly
the intensity redistribution being measured, and a violently corrupted pair would read as a small
residual because both halves were stretched to the same range first.
"""

QUANTILES = (10, 50, 90)
DEFAULT_SAMPLE = 200


@dataclass(frozen=True)
class Residual:
    """What one pair's motion costs, in the scaled space our cache holds images in."""

    residual: float
    residual_p99: float
    background_share: float

    @classmethod
    def scaled(cls, volume: NDArray[np.float32], window: tuple[float, float]) -> NDArray[np.float32]:
        """A volume mapped onto [0, 1] by a window taken from elsewhere, clipped at both ends."""
        low, high = window
        if high <= low:
            return np.zeros_like(volume)
        return np.clip((volume - low) / (high - low), 0.0, 1.0)

    @classmethod
    def measured(cls, artifact: NDArray[np.float32], clean: NDArray[np.float32]) -> "Residual | None":
        """One pair's three statistics, or None when the two volumes are not the same acquisition.

        A shape mismatch is not a measurement to skip quietly: the filenames matched and the arrays did
        not, which means the join is wrong somewhere. It is counted and reported by the caller.
        """
        if artifact.shape != clean.shape:
            return None
        window = (
            float(np.percentile(clean, SCALE_PERCENTILES[0])),
            float(np.percentile(clean, SCALE_PERCENTILES[1])),
        )
        difference = np.abs(cls.scaled(artifact, window) - cls.scaled(clean, window))
        total = float(difference.sum())
        normalised = cls.scaled(clean, window)
        # `<=`, not `<`. Background pixels sit at the bottom of the window and a great many of them hold
        # the SAME value, so the percentile lands exactly on that value and a strict comparison selects
        # none of them — reporting a background share of zero, which reads as "no ghosting" and means
        # "not measured".
        background = normalised <= np.percentile(normalised, BACKGROUND_PERCENTILE)
        return cls(
            residual=float(difference.mean()),
            residual_p99=float(np.percentile(difference, 99.0)),
            background_share=0.0 if total == 0 else float(difference[background].sum() / total),
        )

    @classmethod
    def summarised(cls, residuals: "list[Residual]") -> dict[str, dict[str, float]]:
        """Each statistic at its 10th, 50th and 90th percentile over the pairs measured.

        Quantiles rather than a mean and a standard deviation: these distributions are bounded below at
        zero and a corpus built by corrupting some volumes harder than others has no reason to be
        symmetric.
        """
        columns = {
            "residual": [one.residual for one in residuals],
            "residual_p99": [one.residual_p99 for one in residuals],
            "background_share": [one.background_share for one in residuals],
        }
        return {
            name: {f"p{quantile}": float(np.percentile(column, quantile)) for quantile in QUANTILES}
            for name, column in columns.items()
        }


class Measurement:
    """The three statistics taken over a corpus of pairs, grouped by acquisition plane."""

    def __init__(self, corpus: KmarCorpus) -> None:
        self.corpus = corpus

    def report(self, pairs: list[KmarPair]) -> dict[str, object]:
        """Measure every pair given, carrying the counts every rate in it is over.

        The mismatched count is reported rather than dropped: a pair whose filenames joined and whose
        arrays did not is a broken join, and a summary that silently measured fewer pairs would be a
        rate over a denominator nobody chose.
        """
        planes: dict[str, list[Residual]] = {}
        mismatched = 0
        for pair in pairs:
            residual = Residual.measured(self.corpus.volume(pair.artifact), self.corpus.volume(pair.clean))
            if residual is None:
                mismatched += 1
                continue
            planes.setdefault(pair.plane, []).append(residual)
        measured = sum(len(group) for group in planes.values())
        logger.info(f"{measured} pairs measured, {mismatched} shape-mismatched")
        return {
            "measured": measured,
            "mismatched": mismatched,
            "planes": {
                plane: {"pairs": len(group)} | Residual.summarised(group) for plane, group in sorted(planes.items())
            },
        }


def main() -> None:
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--root", type=Path, required=True, help="the directory holding KMAR-50K*")
    command.parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE,
        help="how many pairs to measure, evenly spaced through the corpus (0 measures all of them)",
    )
    command.parser.add_argument("--out", type=Path, default=None, help="write the report as JSON here as well")
    arguments = command.parsed()

    corpus = KmarCorpus(arguments.root)
    pairs = corpus.pairs()
    # Evenly spaced rather than random or leading: the filenames sort by year and subject, so the first N
    # are one cohort's worth of scanners and a seed would make the number depend on one.
    if arguments.sample and arguments.sample < len(pairs):
        pairs = pairs[:: max(1, len(pairs) // arguments.sample)][: arguments.sample]
    report = Measurement(corpus).report(pairs)

    logger.info(json.dumps(report, indent=2))
    out: Path | None = arguments.out
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info(f"report -> {out}")


if __name__ == "__main__":
    main()
