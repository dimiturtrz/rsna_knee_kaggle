"""How much of the training target is recoverable from the report alone:
`python -m kneemri.evaluation.label_invertibility --vectors <npz> --labels <csv> --holdout <stem>`.

THE QUESTION THIS DECIDES. The public baseline for this competition concatenates a frozen sentence
embedding of the report into the per-finding head and trains BCE, training-only, with the text branch
dropped at inference (`research/2026-09-02_public_baseline_text_branch_objective.md`). That shape is the
one text-supervision mechanism this project has never run. Whether it can help at all turns on a number
nobody had taken: how much of the target a report embedding predicts BY ITSELF.

WHY THE ANSWER IS NOT OBVIOUS IN OUR FAVOUR. `train.csv` carries the twelve columns on 58 of 4407 rows;
every member on the shelf trains on `report_labels_v2.csv`, which three extractors built by READING THE
REPORTS. So the input the public shape would concatenate and the target it would fit are two reads of one
document. A head handed a near-invertible copy of its own target does not enrich the image branch, it
starves it — the text path takes the gradient, and then it is gone at test time, which is the one moment
the image branch has to carry the prediction alone.

WHAT IT READ, AND WHY THE ANSWER IS THE MIDDLE BAND. 0.8553 mean over the reference member's own 909
held-out studies, per finding 0.8037 (Contusion) to 0.9208 (Synovitis). That is BELOW the 0.9053 the best
image member reads on the same rows under the same ruler — so the prediction that framed this run, that a
target extracted from the reports would be near-invertible from them, is REFUTED at this representation.
The mechanism of the shortfall is not mysterious: the extractors are lexical, keying on the words a
radiologist used, and a 384-wide mean-pooled sentence vector is exactly the transform that destroys word
identity.

THE MIDDLE BAND IS NOT "OPEN", AND THAT IS THE FINDING. Gradient starvation does not need invertibility of
1.0; it needs the text path to be CHEAPER than the image path. 0.8553 is available from a frozen vector
through ONE linear layer, against 0.9053 that costs a ViT trunk and an eighteen-epoch schedule. A head
given both from step zero takes the cheap one. This is why the public baseline STAGES the text branch
(`use_text_branch=False` first) — the research note recorded the staging without a reason, and this number
is the reason. So the verdict here is CONTESTED: an arm is licensed, and only a staged, ablated one.

WHAT IT CANNOT SAY, AND THE CAVEAT IS LOAD-BEARING AT THIS READING. A probe's AUC is a statement about
THIS embedding under a LINEAR read; the extractor itself scores 1.000 by construction, so a stronger
reader lies somewhere between and the number is a FLOOR on invertibility, never a ceiling. It also says
nothing about the Kaggle test labels, which the field believes were made from images rather than reports
(knee-7oep) — this is a statement about our own training target and the shortcut a text branch finds in
it.

THE SPLIT AND THE CUT ARE BOTH TAKEN, NOT DERIVED. The held-out studies come from a member's own
`.holdout.npz`, and the target is binarised by `Ledger75`'s own mixed cut — ten findings at 0.75, Medial
Meniscus and Baker's at the midpoint. The label table is GRADED on five rungs (0.08, 0.28, 0.68, 0.82,
0.94), so a cut is not optional: `> 0` calls every one of the 4406 rows positive on every finding, which
is what this module's first run did. Taking both from the ruler is what makes the probe's mean and the
champion's 0.9053 the same statistic on the same rows.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from core import cli
from kneemri.evaluation.ledger75 import MIDPOINT_COLUMNS, RECUT
from kneemri.evaluation.scores import MIDPOINT
from kneemri.runs.holdout_record import SUFFIX

LOGGER = logging.getLogger(__name__)

STUDY = "StudyInstanceUID"
"""The column every label table and every holdout record joins on."""

REFERENCE = 0.9053
"""`labelwise_depth24_drop40_long_s2026` at `ledger75` over these same 909 rows — the best image member
this project owns, and the bar the text has to clear to be a shortcut rather than a second view. A text
read at or above it is not supervision, it is the extractor's own rule coming back through the input."""

COMPLEMENT = 0.70
"""Below this the report is a weak read of its own target and a text branch is plausibly complementary.
Set at the top of the WEAK band the tail screen already names — seven members between 0.4593 and 0.7704
`ledger75` are the ones this project declines to seat — so a text path scoring inside that band is worth
no more attention than an image member scoring there."""

ITERATIONS = 2000
"""Enough for the probe to converge on 384 standardised dimensions; it is a ruler, not a model."""


@dataclass(frozen=True)
class ColumnProbe:
    """One finding's recoverability from the report embedding."""

    label: str
    area: float
    positives: int

    def line(self) -> str:
        """The row, carrying the positive count so a rare column cannot hide behind an AUC."""
        return f"{self.area:.4f}  {self.positives:4d} pos  {self.label}"


@dataclass(frozen=True)
class Invertibility:
    """Every finding's recoverability, and the verdict the mean licenses."""

    rows: tuple[ColumnProbe, ...]
    studies: int

    def mean(self) -> float:
        """The unweighted mean over findings, which is the ruler the competition board uses."""
        return sum(row.area for row in self.rows) / max(len(self.rows), 1)

    def lines(self) -> list[str]:
        """The per-finding reading, its denominator, and what the mean says about the text branch."""
        if not self.rows:
            return ["no finding could be probed: the label table and the vectors share no study"]
        return [
            f"{len(self.rows)} finding(s) probed on {self.studies} held-out studies",
            *[row.line() for row in self.rows],
            f"mean {self.mean():.4f}",
            self.verdict(),
        ]

    def verdict(self) -> str:
        """Which of the three bands the mean lands in, and what each licenses about a text branch."""
        mean = self.mean()
        if mean >= REFERENCE:
            return (
                f"SHORTCUT — the report predicts its own extracted target at {mean:.4f}, at or above the "
                f"{REFERENCE:.4f} the best image member reads on these rows, so concatenating it hands the "
                "head a copy of the label and starves the branch that answers alone at test time"
            )
        if mean < COMPLEMENT:
            return (
                f"COMPLEMENTARY — at {mean:.4f} the report is a weak read of its own target, so a text "
                "branch carries signal the extractor did not take and the concatenation lane is open"
            )
        return (
            f"CONTESTED — at {mean:.4f} the report is a PARTIAL shortcut: under the "
            f"{REFERENCE:.4f} an image member reads, but far above chance and available from a frozen "
            "vector through one linear layer, where the image path costs a trunk and a long schedule. So "
            "an arm must stage the text branch on and ablate it, never train both from step zero"
        )


class LabelInvertibility:
    """A linear probe from the report embedding to the target every member on the shelf trains on."""

    def __init__(self, vectors: Path, labels: Path, models: Path) -> None:
        self.vectors = vectors
        self.labels = labels
        self.models = models

    def embedded(self) -> tuple[list[str], Float[NDArray[np.float32], "studies dims"]]:
        """The report vectors and the studies they belong to, in the order the file stored them."""
        stored = np.load(self.vectors, allow_pickle=True)
        return [str(name) for name in stored["identifiers"]], stored["vectors"].astype(np.float32)

    def held(self, stem: str) -> set[str]:
        """The studies a member held out, taken from its own record so the ruler matches `ledger75`."""
        record = np.load(self.models / f"{stem}{SUFFIX}", allow_pickle=True)
        return {str(study) for study in record["studies"]}

    def findings(self, table: pd.DataFrame) -> list[str]:
        """The finding columns, which are the ones carrying no extractor metadata suffix."""
        return [column for column in table.columns if column != STUDY and "__" not in column]

    def cut(self, label: str) -> float:
        """`Ledger75`'s own threshold for this finding, so the probe and the champion share a ruler."""
        return MIDPOINT if label in MIDPOINT_COLUMNS else RECUT

    def probe(self, stem: str) -> Invertibility:
        """Fit one probe per finding on the training studies and score it on the member's holdout."""
        identifiers, embeddings = self.embedded()
        table = pd.read_csv(self.labels).set_index(STUDY)
        holdout = self.held(stem)
        place = {name: index for index, name in enumerate(identifiers)}
        known = [name for name in table.index.astype(str) if name in place]
        rows = []
        train = np.array([name not in holdout for name in known])
        features = embeddings[[place[name] for name in known]]
        for label in self.findings(table):
            graded = table.loc[known, label].to_numpy(dtype=np.float64)
            seen = np.isfinite(graded)
            truth = graded >= self.cut(label)
            fitted = LogisticRegression(max_iter=ITERATIONS).fit(
                features[train & seen], truth[train & seen].astype(np.int64)
            )
            scored = ~train & seen
            rows.append(
                ColumnProbe(
                    label=label,
                    area=float(self.area(fitted.predict_proba(features[scored])[:, 1], truth[scored])),
                    positives=int(truth[scored].sum()),
                )
            )
        return Invertibility(rows=tuple(rows), studies=int((~train).sum()))

    def area(self, scores: Float[NDArray[np.float64], " studies"], truth: NDArray[np.bool_]) -> float:
        """The rank statistic, computed here so the reading owes nothing to a metric import."""
        positives, negatives = int(truth.sum()), int((~truth).sum())
        if not positives or not negatives:
            return float("nan")
        order = np.argsort(scores)
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
        return float((ranks[truth].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def main() -> None:
    """Report how much of the extracted target a report embedding recovers on its own."""
    command = cli.Command(__doc__, LOGGER)
    command.parser.add_argument("--vectors", type=Path, required=True, help="npz from report_embeddings")
    command.parser.add_argument("--labels", type=Path, required=True, help="the label table members train on")
    command.parser.add_argument("--holdout", type=str, required=True, help="member stem whose split is the ruler")
    arguments = command.parsed()
    root = command.rooted(arguments)
    probe = LabelInvertibility(arguments.vectors, arguments.labels, root.models())
    command.report(probe.probe(arguments.holdout).lines())


if __name__ == "__main__":
    main()
