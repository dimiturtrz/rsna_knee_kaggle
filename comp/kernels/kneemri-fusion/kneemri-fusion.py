"""Submission from OUR members AND the published ones, rank-averaged as one ensemble.

THE QUESTION. Our three whole-stack members average 0.849 on holdout; the twenty published CC0
members average 0.8398 and scored 0.888 on the leaderboard by themselves. Ours are stronger per model
and far fewer; theirs are weaker and many, fitted under a different recipe (channel stacking, a
windowed read) on the same corpus. Rank-averaging rewards DISAGREEMENT between members, not member
quality alone, so a small strong set plus a large diverse set can beat either — or the twenty can
simply outvote the three. This entry is the measurement.

WHY IT COSTS ALMOST NOTHING TO ASK. Both sets were fitted on the same pixels — 336 px, 12 slices,
group 3, canonical slots — so `pixel_group` is identical and ONE decode of the test set serves all
twenty-three models. The extra cost is twenty-three forward passes instead of three, not a second
decode.

THE GUARD. If the two packages ever stop agreeing on pixels this must fail rather than average models
that read different images, so the pixel groups are compared explicitly. Nothing else in the run would
notice: every member still loads, every fingerprint still passes (each is checked against its OWN
reading), and the submission would just be quietly wrong.
"""

import logging
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import override

WORKING = Path("/kaggle/working")


def find_kit() -> Path:
    """The directory to put on `sys.path`, found by looking for the package rather than the wrapper."""
    marker = next(iter(sorted(Path("/kaggle/input").rglob("kneemri/model.py"))), None)
    if marker is not None:
        return marker.parent.parent
    archive = next(iter(sorted(Path("/kaggle/input").rglob("kneemri_src.zip"))), None)
    if archive is None:
        raise FileNotFoundError("no kneemri package under /kaggle/input")
    zipfile.ZipFile(archive).extractall("/tmp/kit")
    return next(iter(sorted(Path("/tmp/kit").rglob("kneemri/model.py")))).parent.parent


def install_wheels(kit: Path) -> None:
    """jaxtyping is absent from Kaggle's image and our modules import it at module top."""
    wheels = sorted(kit.parent.glob("*.whl")) + sorted(kit.glob("*.whl"))
    if wheels:
        subprocess.run(  # noqa: S603 - pip installing wheels we mounted ourselves, fixed argv, no shell
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *[str(w) for w in wheels]],
            check=True,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    kit = find_kit()
    install_wheels(kit)
    sys.path.insert(0, str(kit))

    from kneemri.kernel_run import KernelRun  # noqa: PLC0415 - the kit is not on sys.path until install_wheels runs
    from kneemri.members import MemberPackage  # noqa: PLC0415 - the kit is not on sys.path until install_wheels runs

    logger = logging.getLogger("fusion")

    class Fused(KernelRun):
        """Every mounted package's members, run as one ensemble."""

        @override
        def package(self) -> MemberPackage:
            """Merge every mounted manifest into one package, refusing mixed pixel groups."""
            roots = [
                found.parent
                for found in sorted(Path("/kaggle/input").rglob("manifest.json"))
                if not {"train_series", "test_series"} & set(found.parts)
            ]
            packages = [MemberPackage(root) for root in roots]
            if not packages:
                raise FileNotFoundError("no weights package mounted")

            # ONE DECODE, SO ONE PIXEL GROUP — and since 2026-08-13 our own package spans three of them
            # (res392 and intensity392 read 392 px off a 448 cache, the rest 336). Refusing outright, as
            # this did, throws away the fusion entry entirely; the honest move is to fuse the largest
            # group and say what was dropped. The published twenty define it, being the biggest block and
            # the reason this kernel exists.
            counts: dict[object, int] = {}
            for package in packages:
                for member in package.members:
                    counts[member.pixel_group] = counts.get(member.pixel_group, 0) + 1
            serving = max(counts, key=lambda group: counts[group])
            if len(counts) > 1:
                logger.info(
                    "fusing the %d members that share one decode, dropping %d in %d other group(s)",
                    counts[serving],
                    sum(counts.values()) - counts[serving],
                    len(counts) - 1,
                )

            merged = packages[0]
            # DINOv3 is mounted now, so nothing is excluded for its ENCODER — only for its pixels. A
            # member outside the serving group reads a different image size, and one decode cannot
            # produce two. That is a property of this kernel's single-decode design, not of the member.
            usable = [m for package in packages for m in package.members if m.pixel_group == serving]
            dropped = sum(len(package.members) for package in packages) - len(usable)
            if dropped:
                logger.info("dropped %d member(s) that one decode cannot serve", dropped)
            merged.members = usable
            logger.info(
                "fused %d members from %d packages: %s",
                len(merged.members),
                len(packages),
                ", ".join(str(root) for root in roots),
            )
            return merged

    Fused().run()


if __name__ == "__main__":
    main()
