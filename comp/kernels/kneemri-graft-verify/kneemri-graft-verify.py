"""Verify that our ported architecture is byte-faithful to the published members.

A replication is worth nothing until something proves it. The published weights package carries, per
member, the output that member gave to a seeded synthetic bag — its fingerprint. If our `StudyModel` is
the same function theirs was, every fingerprint reproduces within 2e-3; if we got the pooling, the
normalisation, the resize or the head wrong, the difference is order one and the member is refused.

So this kernel is the oracle for the port: no pixels, no data layer, no submission worth scoring. It
loads all twenty members and prints how much slack each one had.
"""

import sys
from pathlib import Path

import pandas as pd
import torch

WORKING = Path("/kaggle/working")
DINOV2_DIM = 384  # facebook/dinov2-small hidden size
RULE = "=" * 78


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}", flush=True)


def find_kit() -> Path:
    """The mounted source tree. Discovered, not assumed — the probe already taught us that lesson."""
    matches = sorted(Path("/kaggle/input").glob("*/kneemri_src"))
    if not matches:
        raise FileNotFoundError("kneemri-kit is not mounted (no */kneemri_src under /kaggle/input)")
    return matches[0]


def find_dinov2() -> Path:
    """The mounted DINOv2 weights: a directory holding a config.json, under the model mounts."""
    for config in sorted(Path("/kaggle/input").rglob("config.json")):
        if {"train_series", "test_series"} & set(config.parts):
            continue
        if "dinov2" in str(config).lower():
            return config.parent
    raise FileNotFoundError("no DINOv2 model mount found")


def main() -> None:
    section("0. MOUNTS")
    for entry in sorted(Path("/kaggle/input").glob("*")):
        print(f"  {entry.name}")

    kit = find_kit()
    sys.path.insert(0, str(kit))
    print(f"kit: {kit}")

    from kneemri.members import MemberPackage, WeightsError
    from kneemri.model import TARGETS

    package = MemberPackage.find()
    if package is None:
        raise FileNotFoundError("no weights package mounted")
    print(f"package: {package.root} with {len(package.members)} members")

    section("1. SUBMISSION (passthrough, so the run leaves a valid file)")
    competition = sorted(Path("/kaggle/input").glob("*/*/sample_submission.csv"))
    if competition:
        pd.read_csv(competition[0]).to_csv(WORKING / "submission.csv", index=False)
        print(f"wrote a passthrough submission from {competition[0]}")

    section("2. FINGERPRINTS")
    from transformers import AutoModel

    weights = find_dinov2()
    print(f"dinov2: {weights}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    passed, failed = [], []
    for member in package.members:
        backbone = AutoModel.from_pretrained(weights)
        try:
            model = package.load(member, backbone, DINOV2_DIM, device)
        except WeightsError as exc:
            failed.append(member.identifier)
            print(f"  {member.identifier}  fold={member.fold}  REFUSED: {exc}")
            continue
        slack = package.check(model, member, device)
        passed.append(slack)
        print(
            f"  {member.identifier}  fold={member.fold}  max|diff|={slack:.2e}  "
            f"holdout={member.holdout:.4f}  annot={member.annot:.4f}"
        )

    section("3. VERDICT")
    print(f"targets: {len(TARGETS)}")
    print(f"passed {len(passed)} / {len(package.members)}; refused {len(failed)}")
    if passed:
        print(f"worst slack across members: {max(passed):.2e} (tolerance 2e-3)")
    if failed:
        raise SystemExit(f"the port is NOT faithful for: {failed}")
    print("PORT IS FAITHFUL — our StudyModel computes what these weights computed when they were fitted.")


if __name__ == "__main__":
    main()
