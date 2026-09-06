"""Package our trained members for Kaggle: `python -m kneemri.kaggle.package_members`.

Emits the same manifest shape the published package uses, so `MemberPackage` reads ours and theirs
through one code path — including the fingerprint check, which is what stops a member being run under a
reading it was not fitted under.

WHICH members is read from a COMMITTED roster rather than passed as a glob, because the glob lived in
shell history and the package it produced did not. On 2026-08-16 there were two local member
directories — `comp/members` and `comp/members_ours` — both stamping the dataset id
`dimiturnt/kneemri-members`, holding three members and four, while the LIVE package held six. Either
one could be pushed over the live package, and nothing on disk said which set the leaderboard entries
had actually run. The roster is the fix: one file, in git, that a `git log` can date.
"""

import json
import logging
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from core import cli
from kneemri.inference.member_package import MemberPackage
from kneemri.models.backbone import ENCODERS, SMALL

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

    @property
    def variant(self) -> str:
        """The encoder this member's head was fitted on, read the same way `Member.variant` reads it.

        Defaults to small for a checkpoint written before variants existed, matching the manifest side
        rather than guessing differently on the two ends of one round trip.
        """
        config = cast(Mapping[str, object], self.checkpoint["config"])
        return str(config.get("variant", SMALL))

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
class MemberRoster:
    """The committed statement of WHICH members the live package holds.

    Lines, because a set of members chosen for their DIVERSITY shares no name prefix — whole-stack,
    windowed, large, attention — and squeezing that into one glob means widening it until it matches
    everything. `--pattern '*.pt'` once packaged all 54 checkpoints and 8 GB without a word.

    A line that matches NOTHING is refused rather than skipped. A roster is edited by hand, so a typo is
    the expected failure, and a silently-dropped name packages an ensemble one member short of the one
    that was measured — which loads, fingerprints and scores without saying anything is missing.
    """

    text: str

    COMMENT = "#"

    @classmethod
    def read(cls, path: Path) -> "MemberRoster":
        return cls(path.read_text(encoding="utf-8"))

    def patterns(self) -> list[str]:
        """The named members, without the prose that says why each is in the package."""
        lines = [line.split(self.COMMENT)[0].strip() for line in self.text.splitlines()]
        return [line for line in lines if line]

    def checkpoints(self, source: Path) -> list[Path]:
        """Every named member, resolved under the processed root, in roster order."""
        resolved = []
        for pattern in self.patterns():
            matched = sorted(source.glob(pattern))
            if not matched:
                message = f"roster names {pattern!r}, which matches nothing under {source}"
                raise FileNotFoundError(message)
            resolved.extend(matched)
        return resolved


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
        self.shippable()
        identifiers = [member.identifier for member in self.members]
        repeated = sorted({name for name in identifiers if identifiers.count(name) > 1})
        if repeated:
            message = f"two checkpoints package under one id, so one would overwrite the other: {repeated}"
            raise ValueError(message)
        return json.dumps({"format": FORMAT, "members": [member.entry() for member in self.members]}, indent=2)

    def shippable(self) -> None:
        """Refuse a member whose encoder no Kaggle dataset mirrors, before any weights are copied.

        THE CHECK EXISTED AND FIRED TOO LATE. `MemberPackage.mounts` already raises on an unmounted
        variant, and every caller of it is a kernel-metadata check — which runs after this package has
        been built, and after the card that trained the member has been spent. A check in the right
        place at the wrong TIME is a receipt, not a check.

        Packaging is the right time because it is where a member enters the shippable set: three of the
        nine encoders carry no mount, two of them already have trained checkpoints on disk (`cnx_dense`,
        the V-JEPA2 members), and a roster line naming one of those used to package and push without a
        word. The fix for a refusal is a dataset upload, which is why the message names the variant
        rather than the member alone.
        """
        unmounted = sorted({member.variant for member in self.members if ENCODERS[member.variant].mount is None})
        if unmounted:
            names = sorted(member.identifier for member in self.members if ENCODERS[member.variant].mount is None)
            message = (
                f"{names} sit on {unmounted}, which no Kaggle dataset mirrors: a kernel cannot attach "
                "their encoder weights, so packaging them would ship an entry that cannot run. Publish "
                "the weights as a dataset and give the variant a mount, or drop the roster line."
            )
            raise ValueError(message)


def main() -> None:
    command = cli.Command(__doc__, logger)
    command.parser.add_argument(
        "--out", type=Path, default=None, help="where to write; default is the package that ships"
    )
    command.parser.add_argument(
        "--roster",
        type=Path,
        default=Path("comp/members.roster"),
        help="the committed list of members this package holds, one name or glob per line. A FILE "
        "rather than a flag, because the glob that built the live package lived in shell history and "
        "the package it produced did not: two local directories both stamped the live dataset id, "
        "holding three members and four, while the live package held six.",
    )
    command.parser.add_argument("--title", default="kneemri-members")
    command.parser.add_argument("--owner", default="dimiturnt")
    arguments = command.parsed()

    source = command.rooted(arguments).models()
    checkpoints = MemberRoster.read(arguments.roster).checkpoints(source)
    # Named, not counted: a roster line that matched more than intended is invisible as a number and
    # obvious as a list, and this copies ~89 MB per member.
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

    destination = arguments.out or MemberPackage.shipped_root(arguments.paths)
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
