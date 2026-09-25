"""Does every kernel declare the mounts its members need? `python -m kneemri.kaggle.mount_check`.

`verify_members --kernel` already answers this for ONE kernel, and it is opt-in: a submission built
without the flag gets no check, and none of the committed pre-flight checks pass it. This sweeps the
whole `comp/kernels` directory instead, so the question is asked about every entry at once and about the
entry nobody thought to ask about — which is the one that will be edited next.

WHAT IT CANNOT SEE, and says so per kernel rather than guessing: an entry that does not build a
`KernelRun` picks its members some other way (`kneemri-graft` runs whatever package the mount holds,
`kneemri-probe` runs none), and an entry whose selection is `members=N` ranks by recorded holdout, which
is data the parse does not have. The first is skipped; the second reads as the whole package, since
over-declaring costs mount seconds and under-declaring costs the slot.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from core import cli
from core.denominator import Denominator
from kneemri.inference.member_package import MemberPackage, WeightsError
from kneemri.inference.members import Member
from kneemri.kaggle.kernel_script import KernelScript
from kneemri.kaggle.kernel_sources import KernelSources
from kneemri.kaggle.live_dataset import LiveDataset, LiveModel, TruncatedListingError
from kneemri.models.backbone import MountKind
from kneemri.models.encoder import MODEL_MOUNT

logger = logging.getLogger(__name__)

METADATA = "kernel-metadata.json"
MEMBERS_DATASET = "dimiturnt/kneemri-members"
"""The Kaggle dataset the SHIPPED package is pushed to — the fallback when a package does not say.

A kernel may deliberately mount a DIFFERENT weights package: `kneemri-orthorehearse` attaches
`dimiturnt/kneemri-ortho-pkg`, a one-member dataset, precisely so `MemberPackage.find` cannot pick the
big manifest instead. Checked against the shipped package its single member is absent, so the sweep
refused a kernel that is correct — and a gate that is permanently red is how the next real refusal gets
read as noise. What this sweep can answer is "does a kernel that mounts THE PACKAGE IT IS HOLDING
declare what that package needs", so a kernel mounting another one is a skip with the reason.

WHICH package it is holding is `--root`'s answer, not this constant's, and reading the constant made
`--root` a flag that could not do the thing it exists for: `kneemri-relabel7` mounts a seven-member
package of its own, and both invocations skipped it — the default sweep because the kernel names
another dataset, and `--root <that package>` because the comparison was still against this string. An
entry nothing checks is exactly the hole `farseat_v1` went through, and it went through it with every
gate green. A package declares its own id in the `dataset-metadata.json` that is pushed beside it, so
the sweep asks the package; this string is what a package built before that file existed means.
"""
SETTLED_READS = 2
SETTLE_ATTEMPTS = 8
SETTLE_SECONDS = 20.0
SETTLE_PAUSE = partial(time.sleep, SETTLE_SECONDS)
"""What `settled` waits between two reads — a partial rather than a def, which is a top-level function."""


class MountCheck:
    """Every kernel's declared sources, against the mounts the members it names actually need."""

    def __init__(self, package: MemberPackage, live: set[str] | None = None) -> None:
        self.package = package
        self.live = live
        self.datasets = [self.identify(root) for root in package.roots]
        self.dataset = self.datasets[0]
        """The FIRST shelf — the one a kernel must mount to have a package at all. A kernel mounting a
        different one is another package's business; a kernel mounting this one and not the second is
        answered by `verdict`, per member, because only some members live on the second."""

    @staticmethod
    def identify(root: Path) -> str:
        """The dataset id a shelf ships under, or the shipped id when the shelf never recorded one.

        A package built before `package_members` wrote the metadata beside it, and every package a unit
        test builds. The shipped id is the only honest guess, and guessing it wrong can only SKIP a
        kernel — never pass one whose members were not looked at.
        """
        try:
            return LiveDataset.identified(root)
        except (FileNotFoundError, KeyError):
            return MEMBERS_DATASET

    def shelf(self, member: "Member") -> str:
        """Which dataset serves one member — its file is resolved against its OWN root, so the path says.

        A member on the second shelf needs a SECOND mount declared, and a kernel that declares only the
        first drops it silently at run time and scores as a smaller ensemble under the larger name.
        """
        for root, dataset in zip(self.package.roots, self.datasets, strict=True):
            if root in member.file.parents:
                return dataset
        return self.dataset

    def entries(self, kernels: Path) -> list[Path]:
        """Every directory under a root that is TRYING to be a kernel, in a stable order.

        A directory qualifies on its metadata OR on holding code, and the second half is the fix for a
        silence. Keying only on `kernel-metadata.json` made a metadata-less directory vanish from the
        sweep entirely — `kneemri-dark7` was in none of the ok / skipped / refused buckets and drew no
        line at all, so a check that had NOT looked at it read exactly like a check that had. That is
        also not an exotic state: it is what a kernel looks like between being written and being pushed,
        which is precisely when this check runs. A directory with neither metadata nor code is still not
        a kernel and is still not this tool's business.
        """
        return sorted(
            kernel
            for kernel in kernels.iterdir()
            if kernel.is_dir() and ((kernel / METADATA).is_file() or any(kernel.glob("*.py")))
        )

    def verdict(self, kernel: Path) -> str:
        """One kernel's answer: ok, skipped with the reason, or a refusal naming what is missing."""
        scripts = sorted(kernel.glob("*.py"))
        if not scripts:
            return "skipped - no script to read a member selection from"
        if not (kernel / METADATA).is_file():
            # A REFUSAL rather than a skip: `push.sh kernel` cannot push this at all, and the slot clock
            # is running by the time the CLI says so.
            return f"REFUSED - has a script and no {METADATA}, so it cannot be pushed"
        script = KernelScript(scripts[0])
        if script.call() is None:
            return "skipped - builds no KernelRun, so its package is not ours to check"
        sources = KernelSources(kernel / METADATA)
        if not sources.covers(self.dataset):
            return f"skipped - mounts a weights package other than {self.dataset}, which this sweep is not holding"
        names = script.member_names()
        try:
            wanted = self.package.named(names) if names else self.package.members
            mounts = self.package.mounts(wanted)
        except WeightsError as unusable:
            # The message is passed through rather than framed, because there are now two ways to be
            # unusable — a member the package does not hold, and a member on a variant no dataset
            # mirrors — and only the raiser knows which one this is.
            return f"REFUSED - {unusable}"
        # ONLY when the kernel NAMES its members. An entry that ranks whatever mounted has no identity
        # to break: a shelf it does not declare is a smaller pool, not a dropped seat, and refusing it
        # would make seven correct kernels permanently red — which is how a real refusal stops being read.
        shelves: list[str] = sorted({self.shelf(member) for member in wanted}) if names else []
        unmounted = [shelf for shelf in shelves if not sources.covers(shelf)]
        if unmounted:
            return f"REFUSED - serves members from {', '.join(shelves)} and does not declare {', '.join(unmounted)}"
        undeclared = sources.undeclared(mounts)
        if undeclared:
            return f"REFUSED - needs {', '.join(mounts)} and does not declare {', '.join(undeclared)}"
        unpushed = self.unpushed(names)
        if unpushed:
            return f"REFUSED - names {', '.join(unpushed)}, which the LIVE members dataset does not hold"
        return f"ok - declares all of {', '.join(mounts)}"

    def absent(self) -> list[str]:
        """Every member the manifest names that the LIVE dataset does not hold — the package question.

        PACKAGE-LEVEL, and that is the whole point. `MemberPackage.__init__` refuses a manifest whose
        files it cannot all find, so ONE dropped file takes down every kernel that mounts the dataset,
        including kernels whose own tuple never names it. `unpushed` asks the per-kernel question and
        cannot see that: on 2026-08-25 it answered `kneemri-farseat: ok`, and the run then died with
        "manifest listing 40 members but 1 of their files are absent (first 'small_accum_s2026.pt')" —
        a member farseat does not name. A kernel slot bought that distinction.
        """
        if self.live is None:
            return []
        return [member.identifier for member in self.package.members if member.identifier not in self.live]

    def settled(
        self,
        listing: Callable[[], set[str]],
        *,
        reads: int = SETTLED_READS,
        attempts: int = SETTLE_ATTEMPTS,
        pause: Callable[[], None] = SETTLE_PAUSE,
    ) -> list[str]:
        """Poll the live listing until the whole manifest reads present twice running; name what is not.

        THE UPLOAD HALF of the 2026-08-25 failure, and the half `absent` cannot close. A Kaggle dataset
        version can land INCOMPLETE while the CLI reports the push a success — the marker line "Dataset
        version is being created" is printed before the files are all there — and one dropped `.pt` then
        refuses `MemberPackage` for every kernel that mounts the dataset, including the ones whose tuple
        never names it. That cost a kernel run and twenty minutes of a critical path with two slots left.

        TWICE RUNNING, because a single green read is not proof: a version that is still materialising
        answers differently read to read, so a package that is complete once is not therefore complete.
        An incomplete read on the way there is EXPECTED — that is what polling is for — and settling
        returns nothing at all. What is reported when it never settles is the UNION of every member seen
        absent at any point, not the last read: on a listing that flip-flops, the last read is the one
        answer known to be unreliable, and the union is the only honest account of what to re-push.
        """
        seen: set[str] = set()
        clear = 0
        for attempt in range(max(attempts, reads)):
            live = listing()
            missing = [member.identifier for member in self.package.members if member.identifier not in live]
            seen.update(missing)
            clear = 0 if missing else clear + 1
            if clear >= reads:
                return []
            logger.info("live listing holds %d, %d of the manifest absent - re-reading", len(live), len(missing))
            if attempt < attempts - 1:
                pause()
        return sorted(seen)

    def unpushed(self, names: list[str] | None) -> list[str]:
        """The members a kernel needs that the live dataset does not hold — empty when nothing was asked.

        A kernel that names no members ranks the WHOLE package by recorded holdout, so every member the
        package holds has to be live for it: the selection is made at run time from whatever mounted.
        """
        if self.live is None:
            return []
        needed = names if names else [member.identifier for member in self.package.members]
        return [name for name in needed if name not in self.live]


@dataclass(frozen=True)
class LiveReading:
    """What asking Kaggle what the members dataset holds ANSWERED — a state, not an exit.

    The acquisition used to sit inline in `main`, where the truncated case called `sys.exit(2)` from the
    middle of the function. That made the whole live path unreachable from a unit test: the only way to
    exercise it was to run the CLI against real credentials, which is why the half of this tool that
    cost a kernel slot on 2026-08-25 had no test at all. As a state, UNKNOWN is a value the caller reads
    and turns into an exit code at the one place exit codes are decided.
    """

    live: set[str] | None
    unsettled: list[str]
    unknown: str = ""
    """Why the live listing could not be believed, empty when it could. A partial listing answers the
    question wrongly in BOTH directions, so it is neither a refusal nor a pass."""

    @classmethod
    def taken(
        cls,
        package: MemberPackage,
        *,
        live: bool,
        settle: bool,
        listing: Callable[[str], str] | None = None,
        pause: Callable[[], None] = SETTLE_PAUSE,
    ) -> "LiveReading":
        """Read the live dataset when either flag asked for it; otherwise the not-asked reading.

        `--settle` implies `--live`, and the poll runs INSIDE the same try: a listing can truncate on
        any read, not only the first, and a truncation part-way through a poll is the same UNKNOWN.

        `listing` and `pause` are injected for the same reason `settled` takes them: without them the
        only way to reach this code is a real Kaggle round trip, and this is the half of the tool that
        cost a kernel slot untested.
        """
        if not (live or settle):
            return cls(None, [])
        # EVERY shelf, unioned: a member the kernel will mount from the second dataset is present as
        # far as the kernel is concerned, and reading only the first calls it unpushed.
        shelves = [LiveDataset(MountCheck.identify(root), listing) for root in package.roots]

        def held_now() -> set[str]:
            return set().union(*(shelf.members() for shelf in shelves))

        try:
            held = held_now()
            unsettled = MountCheck(package, held).settled(held_now, pause=pause) if settle else []
        except TruncatedListingError as truncated:
            return cls(None, [], str(truncated))
        logger.info("live %s hold %d members", ", ".join(shelf.identifier for shelf in shelves), len(held))
        return cls(held, unsettled)


DATASET_SEGMENTS = 2
"""`owner/name` — what a Kaggle dataset id is, and where a mount's in-dataset subpath starts."""


@dataclass(frozen=True)
class EncoderMounts:
    """Do the ENCODER datasets publish the paths their mounts name? The other half of the live question.

    `absent` asks whether the members dataset holds the `.pt` files, which is a question about
    FILENAMES. An encoder is addressed by a PATH instead: `Encoder.locates` matches its leaf as a path
    segment, so `dimiturnt/dinov3-hf/vits16` needs that directory to exist INSIDE the dataset. Nothing
    asked, and on 2026-08-24 `dimiturnt/vjepa2-hf` was pushed with `--dir-mode zip`, which unpacks a
    subdirectory into the root and dropped the `vjepa2-vitl` segment the mount declared.

    Every other gate stayed green through that, correctly and uselessly: the kernel DECLARED the
    dataset, the dataset EXISTED, the member was packaged and rostered and witness-checked. The kernel
    dropped the member, printed `ENTRY IDENTITY CHANGED`, and `farseat_v1` scored as six members under a
    seven-member name — a flat read that was then recorded as closing the cross-family diversity axis.
    Five days and one closed axis, for a check that is one listing per encoder dataset. See knee-nbk3.
    """

    missing: dict[str, str]
    """Mount string to the dataset id it resolves to, for every mount the live dataset does not hold."""

    @classmethod
    def over(
        cls,
        mounts: list[str],
        listing: Callable[[str], str] | None = None,
    ) -> "EncoderMounts":
        """Ask each distinct encoder source once whether it publishes the subpath its mount names.

        A MODEL mount is the whole slug and has no subpath inside it; a DATASET mount is `owner/name`
        plus the directory the encoder lives in. Which one a mount is comes from `MountKind.of`,
        never from counting its segments — Kaggle's URL grammar is not ours to encode here.
        """
        missing: dict[str, str] = {}
        for mount in sorted(set(mounts)):
            segments = mount.strip("/").split("/")
            model = MountKind.of(mount) == MODEL_MOUNT
            reader = LiveModel if model else LiveDataset
            identifier = mount if model else "/".join(segments[:DATASET_SEGMENTS])
            subpath = "" if model else "/".join(segments[DATASET_SEGMENTS:])
            if not reader(identifier, listing).holds(subpath):
                missing[mount] = identifier
        return cls(missing)


@dataclass(frozen=True)
class MountSweep:
    """Every kernel's verdict, bucketed, carrying the count of directories the sweep WALKED.

    The walked count is a field rather than a derivation because it is the only number that can disagree
    with the others: the buckets are computed from the verdicts, so they always sum to the verdicts and
    can never show a directory that never became one. The number that CAN is the one the sweep walked
    past — which is the whole of what `kneemri-dark7`'s silence looked like.
    """

    verdicts: dict[str, str]
    walked: int
    checked: int
    """Directories that became entries. `walked - checked` is what the sweep declined to look at."""

    @classmethod
    def over(cls, check: MountCheck, kernels: Path) -> "MountSweep":
        """Every entry under a root, judged, with the directory count read from the same directory."""
        entries = check.entries(kernels)
        walked = len([entry for entry in kernels.iterdir() if entry.is_dir()])
        return cls({kernel.name: check.verdict(kernel) for kernel in entries}, walked, len(entries))

    @property
    def refused(self) -> list[str]:
        """The kernels that could not run what they name — the reason this tool exits non-zero."""
        return [name for name, verdict in self.verdicts.items() if verdict.startswith("REFUSED")]

    @property
    def skipped(self) -> list[str]:
        """The kernels this tool cannot answer about, which is not the same as answering ok."""
        return [name for name, verdict in self.verdicts.items() if verdict.startswith("skipped")]

    @property
    def passed(self) -> int:
        """Kernels actually checked and cleared — the denominator a pre-flight is honest about."""
        return len(self.verdicts) - len(self.skipped) - len(self.refused)

    def lines(self, kernels: Path) -> list[str]:
        """The per-kernel rows and the tally, in the order they are read."""
        rows = [f"  {name}: {verdict}" for name, verdict in self.verdicts.items()]
        rows.append(
            f"{self.passed} ok, {len(self.skipped)} skipped, {len(self.refused)} refused of "
            f"{self.checked} kernel directories ({self.walked - self.checked} others under {kernels} "
            f"are not kernels)"
        )
        return rows


def main() -> None:
    """Report every kernel's mount coverage and refuse if any entry could not run what it names."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument(
        "--root", type=Path, default=None, help="a package to check; default is the one that ships"
    )
    command.parser.add_argument("--kernels", type=Path, default=Path("comp/kernels"))
    command.parser.add_argument(
        "--live",
        action="store_true",
        help="also ask Kaggle what the members dataset actually holds (needs credentials)",
    )
    command.parser.add_argument(
        "--settle",
        action="store_true",
        help="implies --live, and POLLS it until the whole manifest reads present twice running",
    )
    arguments = command.parsed()

    package = MemberPackage.shipped(arguments.root, arguments.paths)
    # OPT-IN, because it needs credentials and a round trip, and REQUIRED before a submission: the
    # local answer cannot see a package that leads the pushed one, which is exactly the failure that
    # killed a kernel run on 2026-08-19. A pre-flight without --live checks the kernels; --live checks
    # the SHIPMENT.
    reading = LiveReading.taken(package, live=arguments.live, settle=arguments.settle)
    if reading.unknown:
        # UNKNOWN, not REFUSED and not a clean pass: a partial listing answers the question wrongly
        # in BOTH directions, so the only honest report is that the live check did not run.
        command.unknown(f"live check - {reading.unknown}")
    if reading.unsettled:
        logger.error(
            "the live dataset never read complete: %d of %d members were absent at some point in the "
            "poll (%s). A version that is still materialising answers differently read to read, so this "
            "refuses the package until a re-push settles.",
            len(reading.unsettled),
            len(package.members),
            ", ".join(reading.unsettled),
        )

    check = MountCheck(package, reading.live)
    # BEFORE the per-kernel loop, because it invalidates every verdict that loop is about to print: a
    # manifest the live dataset cannot satisfy refuses the package for ALL of them.
    absent = check.absent()
    if absent:
        logger.error(
            "live dataset does not hold %d of the %d members the manifest names (%s) — this refuses the "
            "package for EVERY kernel, not only the ones naming them. Re-push the members dataset.",
            len(absent),
            len(package.members),
            ", ".join(absent),
        )
    # THE SAME QUESTION ONE LEVEL DOWN, and only under --live because it is a listing per encoder
    # dataset. A mount whose leaf was never published refuses every kernel that names a member on that
    # encoder, and does it SILENTLY at run time by dropping the member rather than failing the run.
    unresolved: dict[str, str] = EncoderMounts.over(package.mounts(None)).missing if reading.live is not None else {}
    for mount, identifier in unresolved.items():
        logger.error(
            "mount %s names a path %s does not publish — a kernel declaring this dataset will still fail "
            "to find the encoder and will DROP the member, scoring as a smaller ensemble. Re-push the "
            "dataset with that directory, or name the mount for the dataset itself (see rad-dino).",
            mount,
            identifier,
        )
    sweep = MountSweep.over(check, arguments.kernels)
    for line in sweep.lines(arguments.kernels):
        logger.info(line)
    # PASSED, not len(verdicts): a --kernels path that names nothing, and a root where every kernel
    # skips, both leave a pre-flight that checked no mount and said so in the words of a clean one.
    empty = f"no kernel under {arguments.kernels} was checkable — every entry skipped, or none was found"
    Denominator(sweep.passed, empty).refuse_empty()
    if sweep.refused or absent or reading.unsettled or unresolved:
        raise SystemExit(cli.Exit.FOUND)


if __name__ == "__main__":
    main()
