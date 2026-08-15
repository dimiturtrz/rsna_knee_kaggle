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

    def apply(self, kernel: Path, *, write: bool) -> bool:
        """Whether the kernel's copy already matched — writing it when it did not and may be written."""
        text = kernel.read_text(encoding="utf-8")
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
