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

from kneemri.inference.members import MemberPackage, WeightsError
from kneemri.kaggle.kernel_script import KernelScript
from kneemri.kaggle.kernel_sources import KernelSources

logger = logging.getLogger(__name__)

METADATA = "kernel-metadata.json"


class MountCheck:
    """Every kernel's declared sources, against the mounts the members it names actually need."""

    def __init__(self, package: MemberPackage) -> None:
        self.package = package

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
        except WeightsError as absent:
            return f"REFUSED - names a member the package does not hold: {absent}"
        undeclared = KernelSources(kernel / METADATA).undeclared(mounts)
        if undeclared:
            return f"REFUSED - needs {', '.join(mounts)} and does not declare {', '.join(undeclared)}"
        return f"ok - declares all of {', '.join(mounts)}"


def main() -> None:
    """Report every kernel's mount coverage and refuse if any entry could not run what it names."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("comp/members"))
    parser.add_argument("--kernels", type=Path, default=Path("comp/kernels"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    check = MountCheck(MemberPackage(arguments.root))
    verdicts = {kernel.name: check.verdict(kernel) for kernel in check.entries(arguments.kernels)}
    for name, verdict in verdicts.items():
        logger.info("  %s: %s", name, verdict)

    refused = [name for name, verdict in verdicts.items() if verdict.startswith("REFUSED")]
    skipped = [name for name, verdict in verdicts.items() if verdict.startswith("skipped")]
    passed = len(verdicts) - len(skipped) - len(refused)
    logger.info("%d ok, %d skipped, %d refused", passed, len(skipped), len(refused))
    if refused:
        sys.exit(1)


if __name__ == "__main__":
    main()
