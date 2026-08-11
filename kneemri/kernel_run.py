"""Everything a Kaggle kernel does once it can import us.

A kernel script cannot be thin all the way down: finding the mounted kit and installing its wheels has
to happen in the script, because it runs *before* the package it is finding can be imported. Everything
after that point belongs here, so a second kernel is a few lines of parameters rather than a copy of the
first — the copy being exactly what the duplication gate exists to stop, and what made extracting
`Submission` necessary the first time.

Two kernels differ only in what they pass: the accuracy entry runs every member with overlapping
windows, and the efficiency entry runs the single strongest member with disjoint ones. That difference
is worth having, because the two leaderboards reward opposite things and a team may hold both — 347 of
897 teams carry an efficiency score strictly below their public best.
"""

import logging
from pathlib import Path

import torch

from kneemri.backbone import ENCODERS
from kneemri.ensemble import Ensemble
from kneemri.members import Member, MemberPackage
from kneemri.submission import Submission

INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")

logger = logging.getLogger(__name__)


class KernelRun:
    """One kernel's work: mount the weights, load the members it wants, write the submission."""

    def __init__(self, members: int | None = None, *, overlap: bool = True) -> None:
        self.members = members
        self.overlap = overlap

    def find_dinov2(self, variant: str) -> Path:
        """The mounted DINOv2 weights for one variant.

        The variant is matched, not merely found. Mounting both small and base and taking whichever
        sorted first would build a member at the wrong width, which surfaces as a shape error during
        load if we are lucky and as a silently wrong model if the head happens to accept it.
        """
        candidates = [
            config.parent
            for config in sorted(INPUT.rglob("config.json"))
            if not {"train_series", "test_series"} & set(config.parts) and "dinov2" in str(config).lower()
        ]
        for mount in candidates:
            if variant in str(mount).lower().split("dinov2")[-1]:
                return mount
        message = f"no DINOv2 {variant} mount among {candidates}"
        raise FileNotFoundError(message)

    def chosen(self, package: MemberPackage) -> list[Member]:
        """Which members to run: all of them, or the strongest few for the efficiency entry."""
        return package.members if self.members is None else package.strongest(self.members)

    def run(self) -> None:
        """Load, predict, write `/kaggle/working/submission.csv`."""
        from transformers import AutoModel  # noqa: PLC0415 - only importable after the kit is on sys.path

        package = MemberPackage.find()
        if package is None:
            message = "no weights package mounted"
            raise FileNotFoundError(message)
        package.members = self.chosen(package)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = {}
        for member in package.members:
            encoder = ENCODERS[member.variant]
            weights = self.find_dinov2(encoder.variant)
            models[member.identifier] = package.load(member, AutoModel.from_pretrained(weights), encoder.dim, device)
        logger.info("loaded and fingerprint-checked %d members from %s", len(models), package.root)

        competition = next(iter(sorted(INPUT.rglob("sample_submission.csv")))).parent
        frame = Submission(Ensemble(package, models, device), competition, WORKING, overlap=self.overlap).run()
        logger.info("%s", frame.head())
