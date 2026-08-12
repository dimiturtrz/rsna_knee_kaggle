"""Package our trained members for Kaggle: `python -m kneemri.package_members --out comp/members`.

Emits the same manifest shape the published package uses, so `MemberPackage` reads ours and theirs
through one code path — including the fingerprint check, which is what stops a member being run under a
reading it was not fitted under.
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch

from core.paths import DataRoot

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("comp/members"))
    parser.add_argument(
        "--pattern",
        nargs="+",
        default=["member_fold*.pt"],
        help="one or more globs under the processed root. SEVERAL because a set of members "
        "chosen for their DIVERSITY shares no name prefix - whole-stack, windowed, large, "
        "attention - and squeezing that into one glob means widening it until it matches "
        "everything. `--pattern '*.pt'` once packaged all 54 checkpoints and 8 GB without "
        "a word.",
    )
    parser.add_argument("--title", default="kneemri-members")
    parser.add_argument("--owner", default="dimiturnt")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    source = DataRoot.from_config(arguments.paths).processed()
    checkpoints = sorted({path for pattern in arguments.pattern for path in source.glob(pattern)})
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints matching {arguments.pattern} under {source}")
    # Named, not counted: a glob that matched more than intended is invisible as a number and obvious
    # as a list, and this copies ~89 MB per member.
    logger.info("packaging %d checkpoint(s):", len(checkpoints))
    for path in checkpoints:
        logger.info("  %s", path.name)

    destination = arguments.out
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    members = []
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        identifier = checkpoint_path.stem.replace("member_", "")
        shutil.copy2(checkpoint_path, destination / checkpoint_path.name)
        members.append(
            {
                "id": identifier,
                "file": checkpoint_path.name,
                "fold": int(identifier.replace("fold", "")) if identifier.startswith("fold") else 0,
                "config": checkpoint["config"],
                "fingerprint": checkpoint["fingerprint"],
                "holdout": float(checkpoint["holdout"]),
                # Our members are selected on the report-derived holdout; the annotation check is
                # reported by the competition, not by us, so there is no honest number to put here.
                "annot": float("nan"),
            }
        )
        logger.info(f"  {identifier}: holdout {checkpoint['holdout']:.4f}  ({checkpoint_path.name})")

    (destination / "manifest.json").write_text(
        json.dumps({"format": 2, "members": members}, indent=2), encoding="utf-8"
    )
    (destination / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": arguments.title, "id": f"{arguments.owner}/{arguments.title}", "licenses": [{"name": "CC0-1.0"}]},
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"packaged {len(members)} members -> {destination}")
    logger.info(f"push with:\n  cd {destination.parent} && kaggle datasets create -p {destination.name} --dir-mode zip")


if __name__ == "__main__":
    main()
