"""The plumbing oracle for a foreign member: `python -m kneemri.evaluation.foreign_gate`.

WHY IT IS ITS OWN MODULE. `kneemri.inference.foreign_member` SERVES a published checkpoint through our
corpus; this SCORES what it serves. They were one file until import-linter said so, and the contract is
right — a serving layer that imports the layer measuring it is how a member ends up fitted to its own
ruler.

WHAT IT ASKS. Their pipeline fed bags of three physically-consecutive slices from a corpus we do not have,
in [0,1] with NO ImageNet normalisation; ours are slot-routed, cropped and cached at another resolution.
A foreign model read through our feed can score low because the FEED is wrong, and that is indistinguishable
from a fact about the model — this project's recurring defect in its most expensive form, since the
conclusion would be drawn on the 909-study holdout at a cost of card hours. Every published archive carries
`gold_auc` on the same 58 radiologist-annotated studies, so that number is an ORACLE rather than a claim:
run our feed on those 58 and compare. Reproduces it, the feed is right and a 909-study read is trustworthy;
reads materially low, the feed is wrong and the fix is in `WindowFeed`, not on the holdout.

IT PRICES A FEED, NOT A MODEL. A number from here says whether our pixels reach their weights intact.
Whether the member is worth a seat is a question for the holdout, and after that for the board. Which is
why `--key` is the flag that matters: the same checkpoint through two caches is a controlled comparison
of two feeds, and that is the only way the shortfall gets attributed to anything.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.findings import TARGETS
from kneemri.evaluation.protocol_prior import Ordering
from kneemri.evaluation.supervision_gap import SupervisionGap
from kneemri.inference.foreign_member import ForeignMember, ForeignReadings, WindowFeed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateReading:
    """Per-finding AUCs through one feed, beside the number the checkpoint publishes for itself."""

    aucs: NDArray[np.float64]
    gold: float

    @property
    def macro(self) -> float:
        """The unweighted mean the published `gold_auc` is also a mean of."""
        return float(self.aucs.mean())

    @property
    def shortfall(self) -> float:
        """How much our feed loses against theirs. Zero-ish means our pixels reach their weights intact."""
        return self.macro - self.gold

    def lines(self, title: str) -> list[str]:
        """The reading as a report, findings first so a single collapsed label is visible as a cause."""
        rows = [title, ""]
        rows.extend(f"{target:22s} {self.aucs[index]:7.4f}" for index, target in enumerate(TARGETS))
        rows.append("")
        rows.append(f"macro {self.macro:.4f} against a published {self.gold:.4f} -> {self.shortfall:+.4f}")
        return rows


@dataclass(frozen=True)
class FeedGate:
    """One published member and one of our caches, held together so the pair can be scored."""

    member: ForeignMember
    feed: WindowFeed
    device: torch.device

    def reading(self, rows: list[int], annotations: Float[np.ndarray, "studies targets"]) -> GateReading:
        """Score the member's probabilities for `rows` against the radiologists, one AUC per finding."""
        values = ForeignReadings(self.device).predict(self.member.to(self.device), self.feed, rows)
        ordering = Ordering()
        aucs = np.array([ordering.auc(values[:, index], annotations[:, index]) for index in range(len(TARGETS))])
        return GateReading(aucs, self.member.spec.gold)


def main() -> None:
    """Score one published checkpoint through our feed and set it beside the AUC it publishes."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--checkpoint", type=Path, required=True, help="a published foreign .pt")
    command.parser.add_argument("--cache", type=Path, required=True, help="the processed directory")
    command.parser.add_argument("--key", required=True, help="the cache key, e.g. 336px_12sl")
    command.parser.add_argument("--train", type=Path, required=True, help="train.csv, which carries the gold rows")
    command.parser.add_argument("--device", default="cuda")
    arguments = command.parsed()

    member = ForeignMember.rebuild(arguments.checkpoint)
    feed = WindowFeed.of(arguments.cache, arguments.key, member.spec.res)
    identifiers, annotations = SupervisionGap().annotated(arguments.train)
    positions = {name: index for index, name in enumerate(WindowFeed.order(arguments.cache, arguments.key))}
    rows = [positions[name] for name in identifiers]

    gate = FeedGate(member, feed, torch.device(arguments.device))
    title = f"{arguments.checkpoint.stem} through OUR {arguments.key} feed, {len(rows)} annotated studies"
    command.report(gate.reading(rows, annotations).lines(title))


if __name__ == "__main__":
    main()
