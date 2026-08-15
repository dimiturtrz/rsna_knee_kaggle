"""Submission from the PUBLISHED members, run through our pipeline — the graft.

Someone else's trained weights, our data layer, our submission path. It measures the field's floor on
our own harness and is a floor to build past, never a result to report as ours.

Two oracles stand behind it: the fingerprint proves the architecture is the function these weights were
fitted to, and the authors' own out-of-fold predictions prove the pixels (max deviation 0.025, per-study
correlation 0.99).

Everything below the bootstrap lives in `kneemri.submission`, shared with the graft kernel. Only the
bootstrap — finding the mounted kit and installing its wheels — has to be here, because it runs BEFORE
the package it is finding can be imported.
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

import torch

WORKING = Path("/kaggle/working")
DINOV2_DIM = 384


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
            zipfile.ZipFile(archive).extractall("/tmp/kit")
            return next(iter(sorted(Path("/tmp/kit").rglob("kneemri/model.py")))).parent.parent
    raise FileNotFoundError("no kneemri package under /kaggle/input")


def install_wheels(kit: Path) -> None:
    """jaxtyping is absent from Kaggle's image and our modules import it at module top."""
    wheels = sorted(kit.parent.glob("*.whl")) + sorted(kit.glob("*.whl"))
    if wheels:
        subprocess.run(  # noqa: S603 - pip installing wheels we mounted ourselves, fixed argv, no shell
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *[str(w) for w in wheels]],
            check=True,
        )


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
    kit = find_kit()
    install_wheels(kit)
    sys.path.insert(0, str(kit))

    from transformers import AutoModel  # noqa: PLC0415 - the kit is not on sys.path until install_wheels runs

    from kneemri.ensemble import Ensemble  # noqa: PLC0415 - the kit is not on sys.path until install_wheels runs
    from kneemri.members import MemberPackage  # noqa: PLC0415 - the kit is not on sys.path until install_wheels runs
    from kneemri.mounts import Mounts  # noqa: PLC0415 - the kit is not on sys.path until install_wheels runs
    from kneemri.submission import Submission  # noqa: PLC0415 - the kit is not on sys.path until install_wheels runs

    # ONE finder for the whole run: it caches, and each of these lookups used to walk the whole
    # competition mount on its own.
    mounts = Mounts()
    package = MemberPackage.find(mounts)
    if package is None:
        raise FileNotFoundError("no weights package mounted")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = find_dinov2(mounts)
    models = {
        member.identifier: package.load(member, AutoModel.from_pretrained(weights), DINOV2_DIM, device)
        for member in package.members
    }
    logging.info(f"loaded and fingerprint-checked {len(models)} members from {package.root}")

    competition = mounts.one("sample_submission.csv", "competition data")
    frame = Submission(Ensemble(package, models, device), competition, WORKING).run()
    logging.info("%s", frame.head())


if __name__ == "__main__":
    main()
