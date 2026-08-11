"""Submission from OUR OWN trained members.

Our weights, our data layer, our submission path. The members were fitted by `kneemri.train` on the
report-derived labels, selected on a holdout split by REPORT-TEXT HASH so the template reports that
recur across studies cannot straddle the divide.

The same fingerprint check guards these members as guards the published ones: a member run under a
different reading than it was fitted under loads cleanly and predicts from the wrong image, and nothing
else in the run would say so.

Everything below the bootstrap lives in `kneemri.submission`, shared with the graft kernel. Only the
bootstrap — finding the mounted kit and installing its wheels — has to be here, because it runs BEFORE
the package it is finding can be imported.
"""

import logging
import subprocess
import sys
import zipfile
from pathlib import Path

import torch

WORKING = Path("/kaggle/working")
DINOV2_DIM = 384


def find_kit() -> Path:
    """The directory to put on `sys.path`, found by looking for the package rather than the wrapper.

    Mounts nest by kind (`/kaggle/input/{competitions,datasets,models}/…`) and a tree uploaded with
    `--dir-mode zip` is expanded with its wrapping directory REMOVED. Three runs were spent on that;
    the rule is to search for the artefact you need, never the container assumed to hold it.
    """
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


def find_dinov2() -> Path:
    """The mounted DINOv2 weights: a config.json under the model mounts, not under the image trees."""
    for config in sorted(Path("/kaggle/input").rglob("config.json")):
        if not {"train_series", "test_series"} & set(config.parts) and "dinov2" in str(config).lower():
            return config.parent
    raise FileNotFoundError("no DINOv2 model mount")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    kit = find_kit()
    install_wheels(kit)
    sys.path.insert(0, str(kit))

    from transformers import AutoModel

    from kneemri.ensemble import Ensemble
    from kneemri.members import MemberPackage
    from kneemri.submission import Submission

    package = MemberPackage.find()
    if package is None:
        raise FileNotFoundError("no weights package mounted")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = find_dinov2()
    models = {
        member.identifier: package.load(member, AutoModel.from_pretrained(weights), DINOV2_DIM, device)
        for member in package.members
    }
    logging.info(f"loaded and fingerprint-checked {len(models)} members from {package.root}")

    competition = next(iter(sorted(Path("/kaggle/input").rglob("sample_submission.csv")))).parent
    frame = Submission(Ensemble(package, models, device), competition, WORKING).run()
    logging.info("%s", frame.head())


if __name__ == "__main__":
    main()
