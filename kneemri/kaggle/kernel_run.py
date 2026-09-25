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
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from core.mounts import Mounts
from kneemri.device import VisibleCards
from kneemri.inference.budget import Budget
from kneemri.inference.ensemble import Ensemble, Serving
from kneemri.inference.member_package import MemberPackage
from kneemri.inference.members import Member
from kneemri.kaggle.submission import Submission
from kneemri.models.backbone import ENCODERS
from kneemri.models.encoder import Encoder

WORKING = Path("/kaggle/working")

logger = logging.getLogger(__name__)


class KernelRun:
    """One kernel's work: mount the weights, load the members it wants, write the submission."""

    def __init__(
        self,
        members: int | None = None,
        *,
        names: Sequence[str] | None = None,
        overlap: bool = True,
        serving: Serving | None = None,
        mounts: Mounts | None = None,
    ) -> None:
        self.members = members
        self.names = names
        self.overlap = overlap
        # HOW the members are served, which until now no kernel could say. `Serving` carries the two
        # axes an entry can vary without retraining anything — the test-time views and the vote — and
        # both were reachable from `member_curve` and from nothing that produces a submission. An axis
        # measurable only offline is an axis whose board transfer cannot be asked about, and this
        # project's last three surprises were all the board disagreeing with the offline ruler.
        self.serving = serving or Serving()
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
        """Which members to run: the ones this entry NAMES, else the strongest few, else all of them.

        Naming comes first because it is the only selection that can carry a joint offline score into a
        kernel: `strongest` ranks members by what each scores alone, and an ensemble is not the sum of
        marginal rankings (`MemberPackage.named` records the 0.0024 that costs on our own holdout).

        And "else all of them" is a fallback, not a default worth taking. Over the thirteen checkpoints
        sharing the corpus split, the best subset peaks at FIVE (0.8617) and every larger one scores
        lower, down to 0.8547 flat over all thirteen — members past the fifth are negative returns, and
        they cost a forward pass each inside the kernel's nine-hour budget to be so.
        """
        if self.names is not None:
            return package.named(self.names)
        return package.members if self.members is None else package.strongest(self.members)

    def identity(self, asked: Sequence[Member], usable: Sequence[Member]) -> str:
        """What to say when the entry that ran is not the entry that was designed — empty when it is.

        A DROPPED MEMBER IS A DIFFERENT EXPERIMENT WHEN THE MEMBERS WERE NAMED. `strongest(n)` asks for
        a quality, so losing one leaves the same kind of entry; a NAMED list is a subset someone chose
        jointly offline, and running three of its four is another entry entirely — on 2026-08-17
        `kneemri-encoder`, whose whole design was one DINOv3 member the others do not share an encoder
        with, dropped exactly that member and wrote the 0.8608 trio under the encoder entry's name. It
        scored, so nothing looked wrong; the slot bought a control that had already been measured.

        The submission is still written, because a run that refuses costs the slot outright and an
        ensemble minus one member is a real entry. What must never happen is that it goes unrecorded.
        """
        if self.names is None:
            return ""
        survived = {member.identifier for member in usable}
        lost = [member.identifier for member in asked if member.identifier not in survived]
        if not lost:
            return ""
        return (
            f"ENTRY IDENTITY CHANGED: named {[member.identifier for member in asked]} and ran "
            f"{sorted(survived)} — {lost} would not load. Score this as the smaller ensemble, not as the "
            f"entry that was designed."
        )

    def package(self) -> MemberPackage:
        """The mounted weights package this kernel runs.

        A method rather than a line inside `run` so an entry can mount something else — the fusion
        entry merges OUR members with the published ones, which are two packages under two mounts and
        one decode, because both were fitted on the same pixels (336 px, 12 slices, group 3, canonical
        slots) and `pixel_group` says so.

        It SPANS every mounted root rather than taking the first. The shipped package is ~10 GB, so a
        member trained today can only reach the board beside it, in a dataset of its own; taking the
        first manifest made that impossible and made re-pushing ten gigabytes the price of one member.
        A name held by two roots refuses.
        """
        found = MemberPackage.spanning(self.mounts)
        if found is None:
            message = "no weights package mounted"
            raise FileNotFoundError(message)
        return found

    def revision(self) -> str:
        """The commit the mounted kit was built from, or a plain statement that it does not say.

        A kit built before the stamp existed carries no `kit.json`, and that is reported as UNSTAMPED
        rather than guessed at — an unknown version is exactly the state that cost a submission.

        The stamp is SEARCHED FOR up the ancestors rather than addressed by a fixed number of `.parent`
        hops. It sits at `kneemri_src/kit.json` while this module is `kneemri_src/kneemri/kaggle/…`, so
        the two hops that the unit test's flattened layout made look right were one short of the real
        one, and every kernel run since has logged UNSTAMPED while mounting a perfectly stamped kit —
        the drift guard silently reporting the state it exists to detect.
        """
        here = Path(__file__).resolve()
        stamps = [parent / "kit.json" for parent in here.parents]
        found = next((stamp for stamp in stamps if stamp.exists()), None)
        if found is None:
            return "UNSTAMPED (built before build_kit recorded its commit)"
        recorded = json.loads(found.read_text(encoding="utf-8"))
        return f"{recorded['commit']}{' DIRTY' if recorded.get('dirty') else ''}"

    def served(self) -> Serving:
        """This entry's serving, with the wall-clock ceiling this path alone is subject to.

        A BUDGET ONLY HERE: a Kaggle batch session is the one place a run is killed for taking too
        long, and a dry run or a witness is not. Composed rather than constructed, so an entry that
        declares views or a vote does not have to also remember to declare the budget — forgetting it
        is a nine-hour kill, and the kernel is the only caller that knows it is inside one.

        AN ENTRY THAT DID DECLARE ONE KEEPS IT. This used to overwrite unconditionally, which is right
        for every entry that is the session's only producer and wrong for one that is not: a kernel
        running a second pass first would hand the members a fresh nine hours with part of the session
        already gone, so both passes aim at the same ceiling and the run is killed at the real one with
        no submission written. That is the exact failure the budget exists to prevent, reintroduced by
        the budget's own assumption that it starts at zero. Defaulting still catches the forgetful case.

        AND IT SAYS WHAT IT SERVES UNDER, because the log is the only place a submission's serving is
        visible after the fact. The first `kneemri-vote` run logged the kit, the members and the
        studies and said nothing about the vote — so the one thing that entry was bought to vary was
        the one thing its log could not confirm had landed, which is a slot spent on an unasserted flag.
        """
        budget = self.serving.budget or Budget()
        served = replace(self.serving, budget=budget)
        logger.info(
            "serving: vote=%s views=%s precision=%s deadline=%.0fs",
            type(served.vote).__name__ if served.vote else "RankAverage (default)",
            served.views or "identity only (default)",
            served.precision or "each member's own (default)",
            budget.seconds,
        )
        return served

    def decode(self, members: Sequence[Member]) -> str:
        """What this entry's `overlap` will ACTUALLY do, member by member.

        `overlap` is a request, not an outcome: `Member.window` hands a whole-stack member one centred
        window and never reads the flag, so on an all-whole-stack pool the two values predict
        identically. The log used to state the request — `overlap=%s` straight off the attribute — and
        two entries whose only difference was that inert flag read as a confounded member-count curve,
        which cost a re-push and a submission slot's worth of doubt to find out it had varied nothing.
        """
        windowed = [member.identifier for member in members if not member.whole_stack]
        if not windowed:
            return f"overlap={self.overlap} requested, INERT — all {len(members)} members are whole-stack"
        return f"overlap={self.overlap} honoured by {len(windowed)} of {len(members)} windowed members: {windowed}"

    def run(self) -> None:
        """Load, predict, write `/kaggle/working/submission.csv`."""

        # WHICH CODE IS THIS? Members are versioned and kernels are versioned; the source between them
        # was not, and on 2026-08-13 a submission ran on a kit 54 commits stale — without
        # Member.precision, so a bf16 member was served in fp16 on a T4. The run's own log now names the
        # commit that produced it, which is the only place the drift is visible from.
        logger.info("kit %s", self.revision())

        package = self.package()
        asked = self.chosen(package)
        package.members = asked

        device = VisibleCards().device()
        models = {}
        usable = []
        for member in asked:
            encoder = ENCODERS[member.variant]
            try:
                weights = self.find_encoder(encoder)
                models[member.identifier] = package.load(member, encoder.load(weights), encoder.dim, device)
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
        logger.info("loaded and fingerprint-checked %d of %d members from %s", len(usable), len(asked), package.root)
        changed = self.identity(asked, usable)
        if changed:
            logger.warning("%s", changed)
        logger.info("%s", self.decode(usable))

        competition = self.mounts.one("sample_submission.csv", "competition data")
        ensemble = Ensemble(package, models, device, self.served())
        frame = Submission(ensemble, competition, WORKING, overlap=self.overlap).run()
        logger.info("%s", frame.head())
