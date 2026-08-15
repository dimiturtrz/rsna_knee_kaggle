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

import json
import logging
from pathlib import Path

import torch

from kneemri.backbone import ENCODERS, Encoder
from kneemri.ensemble import Ensemble
from kneemri.members import Member, MemberPackage
from kneemri.mounts import Mounts
from kneemri.submission import Submission

WORKING = Path("/kaggle/working")

logger = logging.getLogger(__name__)


class KernelRun:
    """One kernel's work: mount the weights, load the members it wants, write the submission."""

    def __init__(self, members: int | None = None, *, overlap: bool = True, mounts: Mounts | None = None) -> None:
        self.members = members
        self.overlap = overlap
        # One object for the whole run, because it caches: the cost this removes is a repeated walk,
        # and a fresh finder per call would keep paying it while looking like it did not.
        self.mounts = mounts or Mounts()

    def find_encoder(self, encoder: Encoder) -> Path:
        """The mounted weights for one encoder.

        The encoder owns its own recognition (`Encoder.locates`) rather than this method knowing the
        word 'dinov2' twice and splitting on it — which is what it did, and which cannot see a DINOv3
        mount at all, nor tell `vits16` from `vitb16`, because those are not what our variants are
        called.

        The search itself is the mounts' job, and it is CACHED there: this is called once per member, so
        an unpruned `rglob` of `/kaggle/input` — 248 seconds, measured — was being paid twenty-two times
        by the fusion entry before a single study decoded.
        """
        candidates = self.mounts.every("config.json")
        found = [mount for mount in candidates if encoder.locates(str(mount))]
        if not found:
            message = f"no {encoder.variant} mount among {candidates}"
            raise FileNotFoundError(message)
        if len(found) > 1:
            # Loading whichever sorted first builds a member on weights it was not fitted on, which
            # surfaces as a shape error if we are lucky and as a silently wrong model if not.
            message = f"{encoder.variant} matches {len(found)} mounts: {found}"
            raise FileNotFoundError(message)
        return found[0]

    def chosen(self, package: MemberPackage) -> list[Member]:
        """Which members to run: all of them, or the strongest few for the efficiency entry."""
        return package.members if self.members is None else package.strongest(self.members)

    def package(self) -> MemberPackage:
        """The mounted weights package this kernel runs.

        A method rather than a line inside `run` so an entry can mount something else — the fusion
        entry merges OUR members with the published ones, which are two packages under two mounts and
        one decode, because both were fitted on the same pixels (336 px, 12 slices, group 3, canonical
        slots) and `pixel_group` says so.
        """
        found = MemberPackage.find(self.mounts)
        if found is None:
            message = "no weights package mounted"
            raise FileNotFoundError(message)
        return found

    def revision(self) -> str:
        """The commit the mounted kit was built from, or a plain statement that it does not say.

        A kit built before the stamp existed carries no `kit.json`, and that is reported as UNSTAMPED
        rather than guessed at — an unknown version is exactly the state that cost a submission.
        """
        stamp = Path(__file__).resolve().parent.parent / "kit.json"
        if not stamp.exists():
            return "UNSTAMPED (built before build_kit recorded its commit)"
        recorded = json.loads(stamp.read_text(encoding="utf-8"))
        return f"{recorded['commit']}{' DIRTY' if recorded.get('dirty') else ''}"

    def run(self) -> None:
        """Load, predict, write `/kaggle/working/submission.csv`."""
        from transformers import AutoModel  # noqa: PLC0415 - only importable after the kit is on sys.path

        # WHICH CODE IS THIS? Members are versioned and kernels are versioned; the source between them
        # was not, and on 2026-08-13 a submission ran on a kit 54 commits stale — without
        # Member.precision, so a bf16 member was served in fp16 on a T4. The run's own log now names the
        # commit that produced it, which is the only place the drift is visible from.
        logger.info("kit %s", self.revision())

        package = self.package()
        package.members = self.chosen(package)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = {}
        usable = []
        for member in package.members:
            encoder = ENCODERS[member.variant]
            try:
                weights = self.find_encoder(encoder)
                models[member.identifier] = package.load(
                    member, AutoModel.from_pretrained(weights), encoder.dim, device
                )
            # A MEMBER THAT WILL NOT LOAD IS A MEMBER, NOT AN ENTRY. A DINOv3 member cost a whole
            # submission this way: its weights did not match the class the runtime built, and the run
            # died with twenty-five other members loaded and ready.
            except Exception:
                logger.exception(
                    "member %s (%s) would not load — dropping it from this entry", member.identifier, member.variant
                )
            else:
                usable.append(member)
        if not usable:
            message = "no member loaded"
            raise RuntimeError(message)
        package.members = usable
        logger.info(
            "loaded and fingerprint-checked %d of %d members from %s", len(usable), len(models) or 0, package.root
        )

        competition = self.mounts.one("sample_submission.csv", "competition data")
        frame = Submission(Ensemble(package, models, device), competition, WORKING, overlap=self.overlap).run()
        logger.info("%s", frame.head())
