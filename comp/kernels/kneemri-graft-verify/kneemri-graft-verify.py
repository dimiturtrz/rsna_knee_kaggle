"""Verify that our ported architecture is byte-faithful to the published members.

A replication is worth nothing until something proves it. The published weights package carries, per
member, the output that member gave to a seeded synthetic bag — its fingerprint. If our `StudyModel` is
the same function theirs was, every fingerprint reproduces within 2e-3; if we got the pooling, the
normalisation, the resize or the head wrong, the difference is order one and the member is refused.

So this kernel is the oracle for the port: no pixels, no data layer, no submission worth scoring. It
loads all twenty members and prints how much slack each one had.
"""

import logging
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import torch

logger = logging.getLogger(__name__)


WORKING = Path("/kaggle/working")
DINOV2_DIM = 384  # facebook/dinov2-small hidden size
RULE = "=" * 78


def section(title: str) -> None:
    logger.info(f"\n{RULE}\n{title}\n{RULE}")


def find_kit() -> Path:
    """The directory to put on `sys.path`, found by looking for the package rather than the wrapper.

    Three runs were spent on mount-shape assumptions: the competition nests under
    `/kaggle/input/competitions/<slug>`, datasets under `/kaggle/input/datasets/<owner>/<name>`, and a
    tree uploaded with `--dir-mode zip` is expanded on mount with the wrapping directory REMOVED — so
    `kneemri_src/` never appears and the mount holds `core/` and `kneemri/` directly.

    The lesson generalises past this kernel: search for the artefact you need (`kneemri/model.py`),
    never for the container you think holds it. The zip branch stays as a fallback for a mount that is
    not expanded.
    """
    marker = next(iter(sorted(Path("/kaggle/input").rglob("kneemri/model.py"))), None)
    if marker is not None:
        return marker.parent.parent

    archive = next(iter(sorted(Path("/kaggle/input").rglob("kneemri_src.zip"))), None)
    if archive is None:
        raise FileNotFoundError("no kneemri package and no kneemri_src.zip anywhere under /kaggle/input")
    destination = Path("/tmp/kit")
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    inner = next(iter(sorted(destination.rglob("kneemri/model.py"))), None)
    if inner is None:
        raise FileNotFoundError(f"{archive} does not contain a kneemri package")
    logger.info(f"expanded {archive.name} -> {destination}")
    return inner.parent.parent


def install_wheels(kit: Path) -> None:
    """Install the kit's wheels offline, before anything imports our package.

    The runtime probe checked the libraries the PIPELINE needs (pydicom, timm, transformers…) and found
    them all present — but not the ones our own modules import. `jaxtyping` is not in Kaggle's image and
    `kneemri.model` imports it at module top, so the kit needs wheels after all. Probe what you import,
    not only what you plan to use.
    """
    wheels = sorted(kit.parent.glob("*.whl")) + sorted(kit.glob("*.whl"))
    if not wheels:
        logger.info("no wheels in the kit — assuming every import is already in the image")
        return
    logger.info(f"installing {[w.name for w in wheels]}")
    subprocess.run(  # noqa: S603 - pip installing wheels we mounted ourselves, fixed argv, no shell
        [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *[str(w) for w in wheels]],
        check=True,
    )


def find_dinov2() -> Path:
    """The mounted DINOv2 weights: a directory holding a config.json, under the model mounts."""
    for config in sorted(Path("/kaggle/input").rglob("config.json")):
        if {"train_series", "test_series"} & set(config.parts):
            continue
        if "dinov2" in str(config).lower():
            return config.parent
    raise FileNotFoundError("no DINOv2 model mount found")


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

    package = MemberPackage.find()
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

    weights = find_dinov2()
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
