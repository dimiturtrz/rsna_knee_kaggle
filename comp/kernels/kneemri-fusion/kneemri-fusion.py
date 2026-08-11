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
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *[str(w) for w in wheels]],
            check=True,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    kit = find_kit()
    install_wheels(kit)
    sys.path.insert(0, str(kit))

    from kneemri.kernel_run import KernelRun
    from kneemri.members import MemberPackage

    logger = logging.getLogger("fusion")

    class Fused(KernelRun):
        """Every mounted package's members, run as one ensemble."""

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

            groups = {member.pixel_group for package in packages for member in package.members}
            if len(groups) != 1:
                message = f"{len(groups)} distinct pixel groups across {len(packages)} packages — one decode cannot serve them"
                raise ValueError(message)

            merged = packages[0]
            merged.members = [member for package in packages for member in package.members]
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
