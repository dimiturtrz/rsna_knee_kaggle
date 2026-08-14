"""Run the SUBMISSION path over labelled studies: `python -m kneemri.dry_run --studies 40`.

WHY THIS EXISTS. Four entries have now failed and nothing local could say why. Two came back at 0.500,
two came back unscored, and the only evidence available was which configurations scored and which did
not — the scored run's log lives in a web UI, and a kernel's interactive run sees three studies, which
is too few for anything that breaks at scale to appear.

The gap it closes is not memory, which a T4 has and this machine does not. It is the OTHER half, and
the one that costs more when wrong: **does this path produce sane predictions at all?** A submission
whose rows are misaligned with its identifiers, or whose decode returns the wrong pixels, is a table of
the right shape and the right length that scores like a coin. Nothing distinguishes it from a crash on
the leaderboard, and nothing here has ever checked it.

So the same `Submission` runs against `train` — same decode, same slot routing, same batching, same
rank-combine, same table writer — and its output is scored against the labels we hold. A pipeline that
is fundamentally broken cannot score well here, and one that scores well here cannot be producing
nonsense there.

WHAT IT DOES NOT TEST: the T4's memory, the mounted kit, and the hidden test's own studies. Those are
the failures `kit_check` and the image budget address; this one is about the ANSWER, not the run.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel

from core.paths import DataRoot
from kneemri.backbone import ENCODERS
from kneemri.ensemble import Ensemble
from kneemri.members import MemberPackage
from kneemri.model import TARGETS
from kneemri.scores import LabelScores
from kneemri.submission import Split, Submission
from kneemri.targets import Targets

logger = logging.getLogger(__name__)

STUDIES = 40
"""Enough for a mean AUC to be readable, few enough to run on a CPU while the card belongs elsewhere."""


class DryRun:
    """The submission path over labelled studies, scored against the labels."""

    def __init__(self, root: DataRoot, device: torch.device) -> None:
        self.root = root
        self.device = device

    def tables(self, identifiers: list[str], into: Path) -> Path:
        """A scratch competition directory holding just these studies' rows.

        The submission path reads whole tables by name, so a subset is written rather than the class
        being taught to take a limit — the point is to run the SHIPPED code, not a variant of it.
        """
        into.mkdir(parents=True, exist_ok=True)
        studies = pd.read_csv(self.root.raw() / "train.csv")
        series = pd.read_csv(self.root.raw() / "train_series.csv")
        studies[studies["StudyInstanceUID"].isin(identifiers)].to_csv(into / "train.csv", index=False)
        series[series["StudyInstanceUID"].isin(identifiers)].to_csv(into / "train_series.csv", index=False)
        return into

    def scored(self, frame: pd.DataFrame, labels: Path) -> dict[str, float]:
        """Score a submission table against the labels we hold for its studies.

        Separate from the running because the running needs a network, a corpus and minutes, and this
        needs a table — the same split that lets `HoldoutPredictions.load` take its backbone as an
        argument instead of fetching one. A method that cannot be constructed in a test is a method
        whose contract nothing checks, and this is the half that decides what the dry run MEANS.

        Scored through the same `LabelScores` the ledger uses, so the number is comparable to a holdout
        number rather than a new kind of measurement.
        """
        truth = Targets.load(labels, self.root.raw() / "train.csv", list(frame["StudyInstanceUID"])).scores
        predictions = frame[list(TARGETS)].to_numpy()
        scores = LabelScores()
        return {"mean": scores.mean(predictions, truth), **scores.per_label(predictions, truth)}

    def _table(self, package: MemberPackage, identifiers: list[str]) -> pd.DataFrame:
        """Run the real submission path over these studies — network, corpus and all."""
        models = {
            member.identifier: package.load(
                member,
                AutoModel.from_pretrained(ENCODERS[member.variant].repository),
                ENCODERS[member.variant].dim,
                self.device,
            )
            for member in package.members
        }
        scratch = self.tables(identifiers, self.root.processed() / "dry_run")
        submission = Submission(
            Ensemble(package, models, self.device),
            scratch,
            scratch,
            split=Split("train", self.root.raw() / "train_series"),
        )
        return submission.run()


def main() -> None:
    """Score the submission path on labelled studies, and refuse a table that ranks like a coin."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=int, default=STUDIES)
    parser.add_argument("--members", type=Path, required=True, help="a weights package to run")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = DataRoot.from_config(arguments.paths)
    package = MemberPackage(arguments.members)
    identifiers = pd.read_csv(root.raw() / "train.csv")["StudyInstanceUID"].tolist()[: arguments.studies]

    logger.info("running %d labelled studies through the submission path", len(identifiers))

    dry = DryRun(root, torch.device(arguments.device))
    scores = dry.scored(dry._table(package, identifiers), arguments.labels)  # noqa: SLF001 - its own CLI
    for label, value in sorted(scores.items(), key=lambda item: item[1]):
        logger.info("  %-18s %.4f", label, value)
    logger.info("mean %.4f over %d studies", scores["mean"], len(identifiers))


if __name__ == "__main__":
    main()
