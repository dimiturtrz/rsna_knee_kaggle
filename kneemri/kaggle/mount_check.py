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

import argparse
import logging
import sys
from pathlib import Path

from core.denominator import Denominator
from kneemri.inference.members import MemberPackage, WeightsError
from kneemri.kaggle.kernel_script import KernelScript
from kneemri.kaggle.kernel_sources import KernelSources
from kneemri.kaggle.live_dataset import LiveDataset, TruncatedListingError

logger = logging.getLogger(__name__)

METADATA = "kernel-metadata.json"


class MountCheck:
    """Every kernel's declared sources, against the mounts the members it names actually need."""

    def __init__(self, package: MemberPackage, live: set[str] | None = None) -> None:
        self.package = package
        self.live = live

    def entries(self, kernels: Path) -> list[Path]:
        """The kernel directories under a root, in a stable order."""
        return sorted(kernel for kernel in kernels.iterdir() if (kernel / METADATA).is_file())

    def verdict(self, kernel: Path) -> str:
        """One kernel's answer: ok, skipped with the reason, or a refusal naming what is missing."""
        scripts = sorted(kernel.glob("*.py"))
        if not scripts:
            return "skipped - no script to read a member selection from"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="a package to check; default is the one that ships")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    parser.add_argument("--kernels", type=Path, default=Path("comp/kernels"))
    parser.add_argument(
        "--live",
        action="store_true",
        help="also ask Kaggle what the members dataset actually holds (needs credentials)",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    package = MemberPackage.at(arguments.root, arguments.paths)
    # OPT-IN, because it needs credentials and a round trip, and REQUIRED before a submission: the
    # local answer cannot see a package that leads the pushed one, which is exactly the failure that
    # killed a kernel run on 2026-08-19. A pre-flight without --live checks the kernels; --live checks
    # the SHIPMENT.
    live = None
    if arguments.live:
        identifier = LiveDataset.identified(package.root)
        try:
            live = LiveDataset(identifier).members()
        except TruncatedListingError as truncated:
            # UNKNOWN, not REFUSED and not a clean pass: a partial listing answers the question wrongly
            # in BOTH directions, so the only honest report is that the live check did not run.
            logger.error("live check UNKNOWN - %s", truncated)
            sys.exit(2)
        logger.info("live %s holds %d members", identifier, len(live))

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
    verdicts = {kernel.name: check.verdict(kernel) for kernel in check.entries(arguments.kernels)}
    for name, verdict in verdicts.items():
        logger.info("  %s: %s", name, verdict)

    refused = [name for name, verdict in verdicts.items() if verdict.startswith("REFUSED")]
    skipped = [name for name, verdict in verdicts.items() if verdict.startswith("skipped")]
    passed = len(verdicts) - len(skipped) - len(refused)
    logger.info("%d ok, %d skipped, %d refused", passed, len(skipped), len(refused))
    # PASSED, not len(verdicts): a --kernels path that names nothing, and a root where every kernel
    # skips, both leave a pre-flight that checked no mount and said so in the words of a clean one.
    empty = f"no kernel under {arguments.kernels} was checkable — every entry skipped, or none was found"
    Denominator(passed, empty).refuse_empty()
    if refused or absent:
        sys.exit(1)


if __name__ == "__main__":
    main()
