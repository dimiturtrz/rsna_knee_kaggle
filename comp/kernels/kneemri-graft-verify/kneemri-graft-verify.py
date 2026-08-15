"""Verify that our ported architecture is byte-faithful to the published members.

A replication is worth nothing until something proves it. The published weights package carries, per
member, the output that member gave to a seeded synthetic bag — its fingerprint. If our `StudyModel` is
the same function theirs was, every fingerprint reproduces within 2e-3; if we got the pooling, the
normalisation, the resize or the head wrong, the difference is order one and the member is refused.

So this kernel is the oracle for the port: no pixels, no data layer, no submission worth scoring. It
loads all twenty members and prints how much slack each one had.
"""

import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # The kit is not on sys.path when this module is imported, so the finder is a name to the
    # type checker and an argument at runtime.
    from kneemri.mounts import Mounts

import pandas as pd
import torch

logger = logging.getLogger(__name__)


WORKING = Path("/kaggle/working")
DINOV2_DIM = 384  # facebook/dinov2-small hidden size
RULE = "=" * 78


def section(title: str) -> None:
    logger.info(f"\n{RULE}\n{title}\n{RULE}")


# >>> KERNEL-BOOTSTRAP
SERIES = {"train_series", "test_series", "train_images", "test_images"}
"""Competition mounts holding one directory per study. `rglob` walks every one of them."""

KIT_DEPTH = 5
"""How deep under /kaggle/input a mounted kit has ever sat, plus room. A bound rather than a guess: it
turns an unbounded walk into a bounded one even if a mount arrives with a name this does not know."""


def find_kit() -> Path:
    """The directory to put on `sys.path`, found WITHOUT walking the competition data.

    The obvious spelling — `Path("/kaggle/input").rglob("kneemri/model.py")` — cost **248 seconds** on
    2026-08-14, four minutes before a single line of ours ran. It descends into the competition mount,
    which holds one directory per study and one per series beneath that, and stats every DICOM on the
    way. The efficiency board scores total runtime, so this was our largest single expense and it was
    spent looking for a file we already knew the neighbourhood of.

    Pruning by name is what removes the cost; the depth bound is what keeps a differently-named mount
    from restoring it silently.
    """
    root = Path("/kaggle/input")
    for parent, directories, _ in os.walk(root):
        here = Path(parent)
        depth = len(here.relative_to(root).parts)
        directories[:] = [] if depth >= KIT_DEPTH else [name for name in directories if name not in SERIES]
        if here.name == "kneemri" and (here / "model.py").exists():
            return here.parent
        archive = here / "kneemri_src.zip"
        if archive.exists():
            zipfile.ZipFile(archive).extractall("/tmp/kit")  # our own kit, built by build_kit.py
            return next(iter(sorted(Path("/tmp/kit").rglob("kneemri/model.py")))).parent.parent
    raise FileNotFoundError("no kneemri package under /kaggle/input")


def install_wheels(kit: Path) -> None:
    """Install the kit's wheels offline, before anything imports our package.

    The runtime probe checked the libraries the PIPELINE needs (pydicom, timm, transformers…) and found
    them all present — but not the ones our own modules import. `jaxtyping` is not in Kaggle's image and
    `kneemri.model` imports it at module top, so the kit needs wheels after all. Probe what you import,
    not only what you plan to use.

    Names what it installs, because a kit shipped without the wheel it needed fails much later, inside
    an import, in a log nobody reads until the slot is spent.
    """
    wheels = sorted(kit.parent.glob("*.whl")) + sorted(kit.glob("*.whl"))
    if not wheels:
        logging.getLogger(__name__).info("no wheels in the kit — assuming every import is already in the image")
        return
    logging.getLogger(__name__).info(f"installing {[wheel.name for wheel in wheels]}")
    subprocess.run(  # noqa: S603 - pip installing wheels we mounted ourselves, fixed argv, no shell
        [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *[str(w) for w in wheels]],
        check=True,
    )


# <<< KERNEL-BOOTSTRAP


def find_dinov2(mounts: "Mounts") -> Path:
    """The mounted DINOv2 weights: a config.json under the model mounts, not under the image trees.

    The skip-list used to be applied to `rglob`'s results, which prunes nothing — the walk has already
    descended into every study by then. `Mounts` skips those directories inside the traversal instead.
    """
    found = [mount for mount in mounts.every("config.json") if "dinov2" in str(mount).lower()]
    if not found:
        raise FileNotFoundError("no DINOv2 model mount")
    return found[0]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    section("0. MOUNTS")
    for kind in sorted(Path("/kaggle/input").glob("*")):
        logger.info(f"  {kind.name}/")
        for owner in sorted(kind.glob("*")):
            logger.info(f"      {owner.name}/")
            for item in sorted(owner.glob("*"))[:6]:
                logger.info(f"          {item.name}{'/' if item.is_dir() else ''}")

    kit = find_kit()
    install_wheels(kit)
    sys.path.insert(0, str(kit))
    logger.info(f"kit: {kit}")

    from kneemri.members import (  # noqa: PLC0415 - the kit is not on sys.path until install_wheels runs
        MemberPackage,
        WeightsError,
    )
    from kneemri.model import TARGETS  # noqa: PLC0415 - the kit is not on sys.path until install_wheels runs
    from kneemri.mounts import Mounts  # noqa: PLC0415 - the kit is not on sys.path until install_wheels runs

    mounts = Mounts()
    package = MemberPackage.find(mounts)
    if package is None:
        raise FileNotFoundError("no weights package mounted")
    logger.info(f"package: {package.root} with {len(package.members)} members")

    section("1. SUBMISSION (passthrough, so the run leaves a valid file)")
    competition = sorted(Path("/kaggle/input").glob("*/*/sample_submission.csv"))
    if competition:
        pd.read_csv(competition[0]).to_csv(WORKING / "submission.csv", index=False)
        logger.info(f"wrote a passthrough submission from {competition[0]}")

    section("2. FINGERPRINTS")
    from transformers import AutoModel  # noqa: PLC0415 - the kit is not on sys.path until install_wheels runs

    weights = find_dinov2(mounts)
    logger.info(f"dinov2: {weights}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    passed: list[float] = []  # each member's fingerprint slack
    failed: list[str] = []  # the identifiers that were refused
    for member in package.members:
        backbone = AutoModel.from_pretrained(weights)
        try:
            model = package.load(member, backbone, DINOV2_DIM, device)
        except WeightsError as exc:
            failed.append(member.identifier)
            logger.info(f"  {member.identifier}  fold={member.fold}  REFUSED: {exc}")
            continue
        slack = package.check(model, member, device)
        passed.append(slack)
        logger.info(
            f"  {member.identifier}  fold={member.fold}  max|diff|={slack:.2e}  "
            f"holdout={member.holdout:.4f}  annot={member.annot:.4f}"
        )

    section("3. VERDICT")
    logger.info(f"targets: {len(TARGETS)}")
    logger.info(f"passed {len(passed)} / {len(package.members)}; refused {len(failed)}")
    if passed:
        logger.info(f"worst slack across members: {max(passed):.2e} (tolerance 2e-3)")
    if failed:
        raise SystemExit(f"the port is NOT faithful for: {failed}")
    logger.info("PORT IS FAITHFUL — our StudyModel computes what these weights computed when they were fitted.")


if __name__ == "__main__":
    main()
