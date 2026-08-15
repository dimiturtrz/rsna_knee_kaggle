"""Package our trained members for Kaggle: `python -m kneemri.kaggle.package_members --out comp/members`.

Emits the same manifest shape the published package uses, so `MemberPackage` reads ours and theirs
through one code path — including the fingerprint check, which is what stops a member being run under a
reading it was not fitted under.
"""

import argparse
import json
import logging
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from core.paths import DataRoot

logger = logging.getLogger(__name__)

FORMAT = 2
"""The manifest version `MemberPackage` reads — ours and the published package share it."""


@dataclass(frozen=True)
class PackagedMember:
    """A checkpoint on disk and the manifest row that addresses it.

    A row callable without a data root, because the row is what INFERENCE reads: while it was assembled
    inline in `main`, the only way to exercise it was to package 8 GB from a real processed root.
    """

    path: Path
    checkpoint: Mapping[str, object]

    @property
    def identifier(self) -> str:
        stem: str = self.path.stem
        return stem.replace("member_", "")

    def entry(self) -> dict[str, object]:
        """The row, at the keys `Member` reads back."""
        identifier = self.identifier
        return {
            "id": identifier,
            "file": self.path.name,
            "fold": int(identifier.removeprefix("fold")) if identifier.startswith("fold") else 0,
            "config": self.checkpoint["config"],
            "fingerprint": self.checkpoint["fingerprint"],
            "holdout": float(cast(float, self.checkpoint["holdout"])),
            # Our members are selected on the report-derived holdout; the annotation check is
            # reported by the competition, not by us, so there is no honest number to put here.
            "annot": float("nan"),
        }


@dataclass(frozen=True)
class MemberManifest:
    """The document a mounted package is read from, and the check that it addresses its own members."""

    members: Sequence[PackagedMember]

    def document(self) -> str:
        """The manifest text, refusing a set of members that two rows would address as one.

        Two patterns can match two files of the SAME name under different directories: the copy
        overwrites and the second row wins, so the package would list a member whose weights are
        another member's — which loads, fingerprints and predicts without a word.
        """
        identifiers = [member.identifier for member in self.members]
        repeated = sorted({name for name in identifiers if identifiers.count(name) > 1})
        if repeated:
            message = f"two checkpoints package under one id, so one would overwrite the other: {repeated}"
            raise ValueError(message)
        return json.dumps({"format": FORMAT, "members": [member.entry() for member in self.members]}, indent=2)


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

    members = []
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        member = PackagedMember(checkpoint_path, checkpoint)
        members.append(member)
        logger.info(f"  {member.identifier}: holdout {checkpoint['holdout']:.4f}  ({checkpoint_path.name})")
    # Assembled before anything is copied, so a set that cannot address its own members is refused
    # without first deleting the previous package and writing ~89 MB per checkpoint.
    document = MemberManifest(members).document()

    destination = arguments.out
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for checkpoint_path in checkpoints:
        shutil.copy2(checkpoint_path, destination / checkpoint_path.name)

    (destination / "manifest.json").write_text(document, encoding="utf-8")
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
