"""Splice the canonical bootstrap into every kernel that marks a region for it, or check that it is there.

`python -m comp.build_kernels` rewrites the marked region of each kernel from `kernel_bootstrap`;
`--check` writes nothing and exits non-zero naming the kernels whose copy has drifted, which is the
form the gate runs. A kernel with no markers is left alone and reported, because the probe kernel
genuinely has no bootstrap and a kernel that LOST its markers is the same silence.
"""

import argparse
import logging
import sys
from pathlib import Path

OPEN = "# >>> KERNEL-BOOTSTRAP"
CLOSE = "# <<< KERNEL-BOOTSTRAP"
KERNELS = Path(__file__).parent / "kernels"
SOURCE = Path(__file__).parent / "kernel_bootstrap.py"

logger = logging.getLogger(__name__)


class Bootstrap:
    """The canonical region, and the splice of it into one kernel."""

    def __init__(self, source: Path = SOURCE) -> None:
        self.text = self.region(source.read_text(encoding="utf-8"), source)

    @staticmethod
    def region(text: str, where: Path) -> str:
        """What sits between the markers, markers excluded, or a failure naming the file."""
        opened = text.find(OPEN)
        closed = text.find(CLOSE)
        if opened < 0 or closed < opened:
            message = f"{where} carries no {OPEN} … {CLOSE} region"
            raise ValueError(message)
        return text[opened + len(OPEN) : closed]

    def spliced(self, text: str) -> str:
        """That kernel's text with its region replaced by the canonical one."""
        opened = text.find(OPEN)
        closed = text.find(CLOSE)
        return text[: opened + len(OPEN)] + self.text + text[closed:]

    def marked(self, kernel: Path) -> bool:
        """Whether this kernel asks for the bootstrap at all."""
        text = kernel.read_text(encoding="utf-8")
        return OPEN in text and CLOSE in text

    def strays(self, text: str) -> list[str]:
        """Imports sitting in that kernel's region that the canonical text does not have.

        The region is a build product, so anything in it that this file did not put there is about to
        be deleted by the next splice — and one thing PUTS code there without anyone deciding to.
        `ruff check --select I --fix` treats the OPEN marker as a comment attached to the imports that
        follow it, so a kernel whose own imports sit just above the marker can have them sorted DOWN
        across it. That happened on 2026-08-25: `import pandas as pd` and `import torch` moved into
        `kneemri-graft-verify`'s region, the next splice overwrote them, and the loss surfaced as
        `Could not find name torch` from pyrefly rather than as anything about the region.

        Imports only, and that is the whole point of the narrowness: the region's OTHER lines
        legitimately differ from canonical whenever `kernel_bootstrap.py` itself has changed, which is
        the ordinary reason to run this. A stray import is the one drift that is never an edit anyone
        made on purpose.
        """
        current = {line.strip() for line in self.region(text, Path("<kernel>")).splitlines()}
        canonical = {line.strip() for line in self.text.splitlines()}
        return sorted(line for line in current - canonical if line.startswith(("import ", "from ")))

    def apply(self, kernel: Path, *, write: bool) -> bool:
        """Whether the kernel's copy already matched — writing it when it did not and may be written."""
        text = kernel.read_text(encoding="utf-8")
        strays = self.strays(text)
        if strays:
            message = f"{kernel.parent.name} has imports inside the bootstrap region: {strays}"
            raise ValueError(message)
        wanted = self.spliced(text)
        if wanted == text:
            return True
        if write:
            kernel.write_text(wanted, encoding="utf-8")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="write nothing; fail if any copy has drifted")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    bootstrap = Bootstrap()
    drifted = []
    for kernel in sorted(KERNELS.glob("*/*.py")):
        if not bootstrap.marked(kernel):
            logger.info(f"{kernel.parent.name}: no bootstrap region")
            continue
        if bootstrap.apply(kernel, write=not arguments.check):
            continue
        drifted.append(kernel.parent.name)
        logger.info(f"{kernel.parent.name}: {'DRIFTED' if arguments.check else 'rewritten'}")
    if drifted and arguments.check:
        logger.error(
            f"{len(drifted)} kernel(s) drifted from comp/kernel_bootstrap.py: run python -m comp.build_kernels"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
