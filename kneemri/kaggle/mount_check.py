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
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

from core import cli
from core.denominator import Denominator
from kneemri.inference.members import MemberPackage, WeightsError
from kneemri.kaggle.kernel_script import KernelScript
from kneemri.kaggle.kernel_sources import KernelSources
from kneemri.kaggle.live_dataset import LiveDataset, TruncatedListingError

logger = logging.getLogger(__name__)

METADATA = "kernel-metadata.json"
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
        names = script.member_names()
        try:
            mounts = self.package.mounts(self.package.named(names) if names else None)
        except WeightsError as unusable:
            # The message is passed through rather than framed, because there are now two ways to be
            # unusable — a member the package does not hold, and a member on a variant no dataset
            # mirrors — and only the raiser knows which one this is.
            return f"REFUSED - {unusable}"
        undeclared = KernelSources(kernel / METADATA).undeclared(mounts)
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

    package = MemberPackage.at(arguments.root, arguments.paths)
    # OPT-IN, because it needs credentials and a round trip, and REQUIRED before a submission: the
    # local answer cannot see a package that leads the pushed one, which is exactly the failure that
    # killed a kernel run on 2026-08-19. A pre-flight without --live checks the kernels; --live checks
    # the SHIPMENT.
    live = None
    unsettled: list[str] = []
    if arguments.live or arguments.settle:
        identifier = LiveDataset.identified(package.root)
        dataset = LiveDataset(identifier)
        try:
            live = dataset.members()
            if arguments.settle:
                unsettled = MountCheck(package, live).settled(dataset.members)
        except TruncatedListingError as truncated:
            # UNKNOWN, not REFUSED and not a clean pass: a partial listing answers the question wrongly
            # in BOTH directions, so the only honest report is that the live check did not run.
            logger.error("live check UNKNOWN - %s", truncated)
            sys.exit(2)
        logger.info("live %s holds %d members", identifier, len(live))
    if unsettled:
        logger.error(
            "the live dataset never read complete: %d of %d members were absent at some point in the "
            "poll (%s). A version that is still materialising answers differently read to read, so this "
            "refuses the package until a re-push settles.",
            len(unsettled),
            len(package.members),
            ", ".join(unsettled),
        )

    check = MountCheck(package, live)
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
    entries = check.entries(arguments.kernels)
    verdicts = {kernel.name: check.verdict(kernel) for kernel in entries}
    for name, verdict in verdicts.items():
        logger.info("  %s: %s", name, verdict)

    refused = [name for name, verdict in verdicts.items() if verdict.startswith("REFUSED")]
    skipped = [name for name, verdict in verdicts.items() if verdict.startswith("skipped")]
    passed = len(verdicts) - len(skipped) - len(refused)
    # AGAINST THE DIRECTORY COUNT, not against len(verdicts): the buckets are computed from the verdicts,
    # so they always sum to the verdicts and can never show a directory that never became one. The number
    # that CAN is the one the sweep walked past — which is the whole of what dark7's silence looked like.
    walked = len([entry for entry in arguments.kernels.iterdir() if entry.is_dir()])
    logger.info(
        "%d ok, %d skipped, %d refused of %d kernel directories (%d others under %s are not kernels)",
        passed,
        len(skipped),
        len(refused),
        len(entries),
        walked - len(entries),
        arguments.kernels,
    )
    # PASSED, not len(verdicts): a --kernels path that names nothing, and a root where every kernel
    # skips, both leave a pre-flight that checked no mount and said so in the words of a clean one.
    empty = f"no kernel under {arguments.kernels} was checkable — every entry skipped, or none was found"
    Denominator(passed, empty).refuse_empty()
    if refused or absent or unsettled:
        sys.exit(1)


if __name__ == "__main__":
    main()
