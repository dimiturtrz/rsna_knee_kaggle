"""Assemble the kneemri-kit Kaggle dataset: our source tree + trained weights + offline dependency wheels.

The submission runs in a code competition with no internet, so everything the kernel imports has to be mounted.
This packs the current `kneemri`/`core` source (no vendoring — it is our own tree), the trained weights, and
the wheels for the dependencies missing from Kaggle's base image into a directory, writes the dataset
metadata, and prints the push command.

Which wheels are actually missing is a property of Kaggle's image, not of this script — probe it from a
kernel before assuming (torch/numpy/scipy/pandas ship; pydicom and the timm/DINOv2 path are the open ones).
"""

import argparse
import json
import shutil
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DATASET_ID = "dimiturnt/kneemri-kit"
_PACKAGES = ("kneemri", "core")
_SOURCE_DIR = "kneemri_src"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=_REPO / "comp" / "kit")
    ap.add_argument("--weights", nargs="*", default=[], help="weight files under processed/")
    ap.add_argument("--wheels", type=Path, required=True, help="dir holding the offline dependency wheels")
    ap.add_argument("--processed", type=Path, required=True, help="processed data dir holding the weight files")
    arguments = ap.parse_args()

    out = arguments.out
    if out.exists():
        shutil.rmtree(out)
    (out / _SOURCE_DIR).mkdir(parents=True)
    for package in _PACKAGES:
        shutil.copytree(_REPO / package, out / _SOURCE_DIR / package, ignore=shutil.ignore_patterns("__pycache__"))
    for wheel in sorted(arguments.wheels.glob("*.whl")):
        shutil.copy2(wheel, out / wheel.name)
    for weight in arguments.weights:
        shutil.copy2(arguments.processed / weight, out / weight)
    (out / "dataset-metadata.json").write_text(
        json.dumps({"title": "kneemri-kit", "id": _DATASET_ID, "licenses": [{"name": "CC0-1.0"}]}, indent=2)
    )
    wheels = [w.name for w in out.glob("*.whl")]
    weights = list(arguments.weights)
    print(f"kit at {out}: src={','.join(_PACKAGES)} wheels={wheels} weights={weights}")
    print(f"push a new version:\n  kaggle datasets version -p {out} -m '<message>' --dir-mode zip")


if __name__ == "__main__":
    main()
