"""Refuse a packaged member that does not reproduce its own fingerprint: `python comp/verify_members.py`.

The sibling of `verify_kit.py`, for weights rather than source. A member records the answer it gave to
a seeded question at training time; rebuilding it and asking again is what catches a member being run
under a reading it was not fitted under.

That is not hypothetical. `members.py` reconstructed every member with `StudyDesign(pool=...)` and
never passed `slice_pooling`, so a member fitted with `--slice-pooling max` came back with channel
stacking — a different function on different pixels. Measured on a real checkpoint, the fingerprint
differed by **2.25** against a 2e-3 tolerance. The gate refused it, correctly; this script is what
makes that refusal happen BEFORE a submission slot is spent rather than inside a Kaggle kernel.
"""

import argparse
import logging
import sys
from pathlib import Path

from transformers import AutoModel

from kneemri.device import VisibleCards

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kneemri.inference.members import MemberPackage
from kneemri.kaggle.kernel_sources import KernelSources
from kneemri.models.backbone import ENCODERS

logger = logging.getLogger(__name__)


def main() -> None:
    """Load every member in a packaged directory and verify it against its stored fingerprint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("comp/members"))
    parser.add_argument("--kernel", type=Path, help="a kernel-metadata.json whose mounts must cover this package")
    parser.add_argument("--members", nargs="*", help="the members that kernel runs; default is the whole package")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    package = MemberPackage(arguments.root)
    mounts = package.mounts(package.named(arguments.members) if arguments.members else None)
    # BEFORE ANY WEIGHTS LOAD, because this costs a file read and the loop below costs a GPU minute per
    # member — and because a kernel that cannot attach an encoder fails no matter how well its members
    # verify. Only checked when a kernel is named: the package is also read by things that never ship.
    if arguments.kernel:
        undeclared = KernelSources(arguments.kernel).undeclared(mounts)
        logger.info("%s needs %s", arguments.kernel.parent.name, ", ".join(mounts))
        if undeclared:
            logger.info("NOT DECLARED by the kernel: %s — do not submit", ", ".join(undeclared))
            sys.exit(1)
    device = VisibleCards().device()
    failures = 0
    for member in package.members:
        encoder = ENCODERS[member.variant]
        try:
            model = package.load(member, AutoModel.from_pretrained(encoder.repository), encoder.dim, device)
            difference = package.check(model, member, device)
        except (ValueError, RuntimeError) as failure:
            logger.info("  %s: REFUSED - %s", member.identifier, failure)
            failures += 1
            continue
        logger.info(
            "  %s: ok (max|diff| %.3g, holdout %.4f, slice_pooling=%s, %d slices)",
            member.identifier,
            difference,
            member.holdout,
            member.slice_pooling,
            member.slices,
        )

    groups = {member.pixel_group for member in package.members}
    logger.info("%d members, %d distinct pixel group(s) — one decode serves each", len(package.members), len(groups))
    if failures:
        logger.info("%d member(s) REFUSED — do not submit", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
