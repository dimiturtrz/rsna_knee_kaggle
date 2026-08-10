"""Submission from the published members, run through OUR pipeline.

This is the graft: someone else's trained weights, our data layer, our submission path. It measures the
field's public plateau as a number on our own harness, and it is a FLOOR to build past — never a result
to report as ours.

Two oracles stand behind it. The fingerprint proves the architecture is the function these weights were
fitted to. The authors' own out-of-fold predictions prove the pixels: our pipeline reproduces them to a
max deviation of 0.025 with a per-study correlation of 0.99, under disjoint windows and probability
pooling.
"""

import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch

WORKING = Path("/kaggle/working")
DINOV2_DIM = 384
SLICES = 12
STUDY_CHUNK = 48
DECODE_THREADS = 8
RULE = "=" * 78


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}", flush=True)


def find_kit() -> Path:
    """Search for the package, never the container that is assumed to hold it."""
    marker = next(iter(sorted(Path("/kaggle/input").rglob("kneemri/model.py"))), None)
    if marker is not None:
        return marker.parent.parent
    archive = next(iter(sorted(Path("/kaggle/input").rglob("kneemri_src.zip"))), None)
    if archive is None:
        raise FileNotFoundError("no kneemri package under /kaggle/input")
    zipfile.ZipFile(archive).extractall("/tmp/kit")
    inner = next(iter(sorted(Path("/tmp/kit").rglob("kneemri/model.py"))))
    return inner.parent.parent


def install_wheels(kit: Path) -> None:
    """jaxtyping is not in Kaggle's image and our modules import it at module top."""
    wheels = sorted(kit.parent.glob("*.whl")) + sorted(kit.glob("*.whl"))
    if wheels:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *[str(w) for w in wheels]],
            check=True,
        )


def find_dinov2() -> Path:
    for config in sorted(Path("/kaggle/input").rglob("config.json")):
        if not {"train_series", "test_series"} & set(config.parts) and "dinov2" in str(config).lower():
            return config.parent
    raise FileNotFoundError("no DINOv2 model mount")


def main() -> None:
    started = time.time()
    kit = find_kit()
    install_wheels(kit)
    sys.path.insert(0, str(kit))

    from kneemri.ensemble import Ensemble
    from kneemri.members import MemberPackage
    from kneemri.model import StudyModel
    from kneemri.study import Study

    competition = next(iter(sorted(Path("/kaggle/input").rglob("sample_submission.csv")))).parent
    sample = pd.read_csv(competition / "sample_submission.csv")
    sample.to_csv(WORKING / "submission.csv", index=False)
    print(f"kit={kit}  competition={competition}  wrote a fallback submission ({len(sample)} rows)")

    package = MemberPackage.find()
    if package is None:
        raise FileNotFoundError("no weights package mounted")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    section("1. MEMBERS")
    from transformers import AutoModel

    weights = find_dinov2()
    models = {}
    for member in package.members:
        model = package.load(member, AutoModel.from_pretrained(weights), DINOV2_DIM, device)
        models[member.identifier] = model
    print(f"loaded and fingerprint-checked {len(models)} members in {time.time() - started:.0f}s")
    ensemble = Ensemble(package, models, device)

    section("2. TEST STUDIES")
    test = pd.read_csv(competition / "test.csv")
    series = pd.read_csv(competition / "test_series.csv")
    planes = series.set_index("SeriesInstanceUID")["Anatomical_Plane"].to_dict()
    identifiers = test["StudyInstanceUID"].tolist()
    print(f"{len(identifiers)} studies, {len(series)} series")

    per_member: dict[str, list] = {member.identifier: [] for member in package.members}
    order: list[str] = []
    for start in range(0, len(identifiers), STUDY_CHUNK):
        chunk = identifiers[start : start + STUDY_CHUNK]
        decode_started = time.time()
        with ThreadPoolExecutor(max_workers=DECODE_THREADS) as pool:
            bags = list(
                pool.map(
                    lambda name: Study.from_directory(competition / "test_series" / name, planes).bag(SLICES), chunk
                )
            )
        images = torch.stack([bag[0] for bag in bags]).to(device)
        mask = torch.stack([bag[1] for bag in bags]).to(device)
        decoded = time.time() - decode_started

        predict_started = time.time()
        for member in package.members:
            per_member[member.identifier].append(ensemble.predict_member(member, images, mask))
        order.extend(chunk)
        print(
            f"  studies {start + 1}-{start + len(chunk)}: decode {decoded:.0f}s, "
            f"predict {time.time() - predict_started:.0f}s, elapsed {time.time() - started:.0f}s",
            flush=True,
        )

    section("3. SUBMISSION")
    import numpy as np

    stacked = [np.concatenate(per_member[member.identifier]) for member in package.members]
    combined = ensemble.combine(stacked)
    frame = pd.DataFrame(ensemble.submission(order, combined))
    frame = frame.set_index("StudyInstanceUID").loc[identifiers].reset_index()
    frame.to_csv(WORKING / "submission.csv", index=False)
    print(frame.head())
    print(f"wrote {len(frame)} rows in {time.time() - started:.0f}s total")


if __name__ == "__main__":
    main()
