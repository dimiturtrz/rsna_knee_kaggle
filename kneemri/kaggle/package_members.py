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
        """The row, at the keys `Member` reads back.

        A checkpoint with no fingerprint is REFUSED BY NAME rather than by `KeyError`. `weight_soup`
        drops the key deliberately and nothing restored it, so every soup raised here with every other
        gate green and the traceback named neither the soup nor the remedy (knee-emyky).
        """
        identifier = self.identifier
        if "fingerprint" not in self.checkpoint:
            message = (
                f"{self.path.name} carries no weight fingerprint, so it cannot be packaged - the "
                "fingerprint is what stops a member being run under a reading it was not fitted "
                "under. A weight soup arrives this way by design; give it one with\n"
                f"  python -m kneemri.inference.holdout_backfill --device cuda --checkpoints {self.path}"
            )
            raise ValueError(message)
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

    LICENCE = "licence:"
    """The trailing-comment tag naming the external table a member trained on, when that table's terms
    are not settled against competition rule 2.5's obligation to license the winning submission.

    It lives in the COMMENT because packaging a tagged member is not the thing being refused — SEATING
    one in a submitted entry is. `patterns` splits on `#` and never sees this, so the shelf still holds
    the member and every offline reading of it stays valid; only a `chosen` tuple is refused.
    """

    SHELF_DIR = Path("comp")
    SHELF_GLOB = "members*.roster"

    @classmethod
    def read(cls, path: Path) -> "MemberRoster":
        return cls(path.read_text(encoding="utf-8"))

    @classmethod
    def shelves(cls, directory: Path | None = None) -> tuple[Path, ...]:
        """Every shelf that ships, discovered rather than listed, in shelf order.

        A shelf exists because `kaggle datasets version` re-pushes a whole dataset: at 156 members and
        ~13.8 GB, putting one new member on the first shelf taxes its first submission with all of it.
        So shelves ROTATE — a new one opens whenever the current one stops being cheap — and the count
        is a fact about upload cost, never about what is documented or shippable.

        That rotation is why the list is a glob. Three modules each held their own literal tuple, and
        opening the third shelf made all three wrong at once: `roster_check` reported its two members
        undocumented, `champion_check` could not have rebuilt a pin that seated one, and
        `submit_check` — whose copy is the LICENCE gate — would have read a foreign member's
        `# licence:` tag on any shelf but the first as absent, which is the one direction that fails
        silently. Shelf 2 carries no tag today, so that hole was latent; it would not have stayed that
        way.

        Ordered by the shelf's own number so shelf 10 follows shelf 9 rather than shelf 1, which plain
        lexicographic sorting of the filenames gets wrong. Order is presentation only — `spanning`
        merges every mounted root and REFUSES a name two shelves both hold.
        """
        found = (directory or cls.SHELF_DIR).glob(cls.SHELF_GLOB)
        return tuple(sorted(found, key=lambda shelf: int("".join(c for c in shelf.stem if c.isdigit()) or "1")))

    def patterns(self) -> list[str]:
        """The named members, without the prose that says why each is in the package."""
        lines = [line.split(self.COMMENT)[0].strip() for line in self.text.splitlines()]
        return [line for line in lines if line]

    def licence_refusals(self, names: set[str]) -> list[str]:
        """Every seat in `names` whose roster line carries a `# licence:` tag, as the line a check prints.

        Keyed on the name a kernel spells — a `chosen` tuple never carries the `.pt` the roster does.
        """
        tagged = {}
        for line in self.text.splitlines():
            member, _, comment = line.partition(self.COMMENT)
            if member.strip() and self.LICENCE in comment:
                tagged[member.strip().removesuffix(".pt")] = comment.split(self.LICENCE)[1].split()[0]
        return [
            f"{name} trains on the {tagged[name]} table, whose terms are unresolved against rule 2.5"
            for name in sorted(names & set(tagged))
        ]

    def shelf(self) -> "RosterShelf":
        """Which shelf this roster's members belong to, declared in the roster's own header."""
        return RosterShelf.read(self.text)

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
class RosterShelf:
    """WHERE a roster's members are written and WHICH dataset id stamps them — declared by the roster.

    THE DEFAULT WAS THE FOOTGUN. `--out` used to default to the shipped package and `--title` to
    `kneemri-members`, so `package_members --roster comp/members2.roster` — the SECOND shelf, every flag
    spelled deliberately except the two that decide the destination — wrote a two-member package into
    the shipped root (which is rmtree'd first) stamped with the LIVE dataset id, and it pushed. For
    about fifteen minutes the shelf every champion kernel mounts held two files.

    The roster is what decides which shelf, so the roster is what declares it. A default that is right
    for the common case is a footgun on the rare one, and the rare one is the one being typed.
    """

    title: str
    directory: str

    TITLE = "# shelf:"
    DIRECTORY = "# out:"

    @classmethod
    def read(cls, text: str) -> "RosterShelf":
        """Both header directives, refusing a roster that declares neither, either twice, or one alone."""
        declared = []
        for marker in (cls.TITLE, cls.DIRECTORY):
            found = [line.removeprefix(marker).strip() for line in text.splitlines() if line.startswith(marker)]
            if len(found) != 1:
                message = f"a roster declares its shelf with exactly one {marker!r} line; this one has {len(found)}"
                raise ValueError(message)
            declared.append(found[0])
        return cls(title=declared[0], directory=declared[1])

    def root(self, models: Path) -> Path:
        """The directory this shelf is built in, under the models root both shelves live beneath."""
        return models / self.directory

    def agrees(self, out: Path | None, title: str | None, models: Path) -> None:
        """Refuse a flag that points somewhere other than where the roster says these members go."""
        wanted = self.root(models)
        if out is not None and out.resolve() != wanted.resolve():
            message = f"--out {out} contradicts the roster, which declares {wanted}"
            raise ValueError(message)
        if title is not None and title != self.title:
            message = f"--title {title!r} contradicts the roster, which declares {self.title!r}"
            raise ValueError(message)


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
        "--out", type=Path, default=None, help="where to write; the ROSTER decides, this may only agree with it"
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
    command.parser.add_argument("--title", default=None, help="the dataset id; the ROSTER decides, see --out")
    command.parser.add_argument("--owner", default="dimiturnt")
    arguments = command.parsed()

    source = command.rooted(arguments).models()
    roster = MemberRoster.read(arguments.roster)
    shelf = roster.shelf()
    shelf.agrees(arguments.out, arguments.title, source)
    checkpoints = roster.checkpoints(source)
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

    destination = shelf.root(source)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for checkpoint_path in checkpoints:
        shutil.copy2(checkpoint_path, destination / checkpoint_path.name)

    (destination / "manifest.json").write_text(document, encoding="utf-8")
    (destination / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": shelf.title, "id": f"{arguments.owner}/{shelf.title}", "licenses": [{"name": "CC0-1.0"}]},
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"packaged {len(members)} members -> {destination}")
    logger.info(f"push with:\n  cd {destination.parent} && kaggle datasets create -p {destination.name} --dir-mode zip")


if __name__ == "__main__":
    main()
