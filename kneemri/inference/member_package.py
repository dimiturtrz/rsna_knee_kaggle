"""A mounted weights package: which members a root holds, and proving each is the model it was.

Split from `members.py`, which describes ONE member — its reading, its geometry, the config it was
fitted under. This describes a SHELF of them: what a mounted root holds, which mounts those members
need, how several roots become one package, and the fingerprint check that refuses weights that load
cleanly and compute something else.

The two are different subjects with different consumers. A trainer or a curve reader wants `Member`
and never a package; a kernel, a packager and every pre-flight check want the package. Keeping them in
one file put a god-file ceiling between the project and its next member.
"""

import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from core.mounts import Mounts
from core.paths import DataRoot
from kneemri.inference.members import Member
from kneemri.models.backbone import ENCODERS
from kneemri.models.model import StudyModel

logger = logging.getLogger(__name__)

SHIPPED_PACKAGE = "all_ours"
"""The directory under the processed root that `package_members` writes and `push.sh` uploads."""

SECOND_SHELF = "members_2"
"""The second shelf, pushed as its own Kaggle dataset and mounted BESIDE the first, in mount order.

The shipped shelf holds 127 members and about 10 GB, so a member trained after the last push can only
reach the board by re-pushing all of it — which is why every offline gain since was unshippable. A
kernel sees both roots through `spanning`; a local check that reads only the first answers a different
question than the kernel asks, and answers it wrongly in the direction that refuses a valid entry. The
same fixed pair, in the same order, that `roster_check` documents against."""


class WeightsError(RuntimeError):
    """Raised when attached weights cannot be trusted to be the ones that were fitted.

    Deliberately fatal: a run that predicts from a mismatched model completes, writes a plausible
    submission, and differs from a correct one only in a number no output of the run reveals.
    """


FINGERPRINT_TOLERANCE = 2e-3
"""Above GPU-to-GPU numeric noise (~1e-5), far below what any preprocessing change moves (order one)."""


class MemberPackage:
    """A mounted weights package: the manifest, its members, and the check that they are trustworthy."""

    @staticmethod
    def shipped_root(paths: Path = Path("paths.yaml")) -> Path:
        """Where the package that actually ships lives — the ONE directory allowed to carry its dataset id.

        Five CLIs defaulted `--root` to `comp/members`, an in-repo mirror of the same package, and
        `package_members --out` defaulted there too. Both directories stamped `dimiturnt/kneemri-members`
        while only the other one was pushed, so the mirror silently aged: on 2026-08-18 it held seven
        members against the live package's nine, and `mount_check`'s DEFAULT invocation refused a valid
        kernel because the member it names was absent from the copy nobody uploads. A check whose default
        answer is wrong teaches its operator to pass `--root` and stop reading the refusal.

        The weights are gitignored, so the mirror was never version control's copy of anything — it was a
        second home for a thing that can only have one.
        """
        return DataRoot.from_config(paths).models() / SHIPPED_PACKAGE

    @classmethod
    def at(cls, root: Path | None, paths: Path = Path("paths.yaml")) -> "MemberPackage":
        """The package a CLI was pointed at, or the one that ships when it was pointed nowhere.

        Every `--root` defaults to None and resolves through here, so the fallback is written once
        rather than as an `or` at each of four call sites that could each drift to a different home.
        """
        return cls(root if root is not None else cls.shipped_root(paths))

    @classmethod
    def shipped(cls, root: Path | None = None, paths: Path = Path("paths.yaml")) -> "MemberPackage":
        """The local package AS A KERNEL WILL MOUNT IT — the shipped shelf plus the shelf beside it.

        `at` answers "the one package a CLI was pointed at"; a kernel calls `spanning` and gets both
        roots. A pre-flight built on `at` therefore checks a package no kernel runs: it refused every
        entry naming `oof_teacher_s2026` while the entry was correct, because the member is on the
        second shelf. An explicit `--root` is still exactly one shelf — that flag exists to name a
        package, not to grow one.
        """
        package = cls.at(root, paths)
        beside = package.root.parent / SECOND_SHELF
        if root is None and (beside / "manifest.json").is_file():
            package.absorb(cls(beside).members, beside)
        return package

    def __init__(self, root: Path, tie: str | None = None) -> None:
        """A mounted package, optionally under a tie rule WE measured rather than one it declares.

        THE OVERRIDE IS AN INFERENCE ABOUT SOMEONE ELSE'S WEIGHTS, and it is an argument rather than a
        default so that it can never be mistaken for something the package said. The published manifest
        carries no `tie` field, so its members would otherwise be routed under OUR first-maximal rule —
        and they reproduce their own recorded predictions markedly better under last-maximal (4 of 4
        members, 3 seeds, mean residual 15-25% lower; knee-evn). Passing it here keeps the claim, its
        evidence and its owner in one place instead of buried in a kernel.

        It applies to every member of the package, which is right for a package whose members share one
        author and one pipeline, and wrong for anything else — so it is opt-in per construction.
        """
        self.root = root
        self.roots = [root]
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.members = [
            Member(
                identifier=entry["id"],
                file=root / entry["file"],
                fold=int(entry["fold"]),
                config=entry["config"] if tie is None else {**entry["config"], "tie": tie},
                fingerprint=np.asarray(entry["fingerprint"], np.float32),
                holdout=float(entry["holdout"]),
                annot=float(entry["annot"]),
            )
            for entry in manifest["members"]
        ]
        missing = [member.file.name for member in self.members if not member.file.is_file()]
        if missing:
            raise WeightsError(
                f"{root} holds a manifest listing {len(self.members)} members but "
                f"{len(missing)} of their files are absent (first {missing[0]!r})"
            )

    def strongest(self, count: int) -> list[Member]:
        """The `count` members with the best recorded holdout, best first.

        For the Efficiency Prize, where the accuracy leaderboard and the efficiency one reward opposite
        things: the best accuracy anywhere on the efficiency board sits at rank 638, while our accuracy
        run cheaply is worth ranks 29-50.

        HOW MANY TO ASK FOR is not a free choice, and the holdout answer this docstring used to give was
        wrong on the leaderboard. On a SEED ensemble's holdout, members two and three bought +0.0018 and
        +0.0007 and the best single outscored the three-member average, which reads as "one is enough".
        Scored entries say otherwise: one member 0.868, two whole-stack 0.882, three whole-stack 0.882,
        five windowed 0.888. Ask for two — one costs 0.020, and a third buys nothing.

        Ranking by each member's OWN holdout is the only ordering available at submission time, and it
        is a weak one: those numbers come from different folds and, for anything trained before the
        label table was recovered, from different rulers. It picks a good member, not provably the best.

        MEASURED 2026-08-17, and the ordering is not merely weak — it is not the right question. Over
        the thirteen checkpoints that share the corpus split, `deep24` scores 0.8352 alone, BELOW the
        pool's floor of 0.8459, and joint selection takes it at every subset size from four up; `crop80`
        scores 0.8345, within 0.0007 of it, and joint selection takes it at none. Solo score does not
        separate the two. Prefer `named` with a subset chosen jointly offline whenever one exists, and
        read this sort as the fallback for when nobody has done that work.

        A member with NO recorded holdout is ranked LAST rather than compared, because a full-data fit
        has none by construction and NaN compares False against every number — under a plain descending
        sort it lands FIRST and `strongest(1)` returns the one member that was never scored. Naming an
        unranked member for a submission stays a decision someone makes, not one a sort makes for them.
        """
        ranked = sorted(self.members, key=lambda member: (not math.isnan(member.holdout), member.holdout), reverse=True)
        unranked = [member.identifier for member in ranked if math.isnan(member.holdout)]
        if unranked and count > len(ranked) - len(unranked):
            logger.warning("members with no recorded holdout, ranked last: %s", ", ".join(unranked))
        return ranked[:count]

    def mounts(self, members: Sequence["Member"] | None = None) -> list[str]:
        """Every Kaggle source a kernel must attach to run THESE members, deduplicated and sorted.

        WHICH members is an argument because a kernel need not run the package: an entry that names a
        two-member subset needs those two encoders and nothing else, so asking about the whole package
        would flag mounts it is right not to declare. Measured over the ten kernels in `comp/kernels`
        on 2026-08-17: three of them run subsets and would read as undeclared against the full package.

        The weights a member was fitted on are not in its checkpoint — only the head is, and the
        backbone is reconstructed from `variant` at load time. So a package's real dependency is the set
        of encoder mounts its members name, and that set is derived from the members rather than typed
        into a kernel's metadata by hand.

        A variant with no mount is refused HERE, as a `WeightsError` rather than the `ValueError`
        `mounted_source` raises on its own. The distinction is not cosmetic: every caller of this method
        is a pre-flight check that already handles `WeightsError` by returning a refusal and carrying on
        to the next kernel, and handles nothing else at all — so an unmounted variant used to escape as
        a traceback that left the REMAINING kernels unchecked. A shippability failure and a
        weights-mismatch failure are the same kind of answer to the same question: these weights cannot
        be trusted to run under this entry.
        """
        chosen = self.members if members is None else members
        adrift = [member for member in chosen if ENCODERS[member.variant].mount is None]
        if adrift:
            unmounted = sorted({member.identifier for member in adrift})
            variants = sorted({member.variant for member in adrift})
            message = (
                f"{unmounted} sit on {variants}, which no Kaggle dataset mirrors, so they can be trained "
                "here and not submitted until their weights are published"
            )
            raise WeightsError(message)
        return sorted({ENCODERS[member.variant].mounted_source for member in chosen})

    def named(self, wanted: Sequence[str]) -> list["Member"]:
        """Exactly these members, in the order asked for, or a refusal naming what is not mounted.

        WHY AN ENTRY NAMES ITS MEMBERS RATHER THAN ASKING FOR THE STRONGEST FEW. `strongest` sorts by
        each member's own solo holdout, which is a MARGINAL ranking, and an ensemble is not marginal:
        measured 2026-08-16 on the 909-study fold-0 holdout over eight of our cached members, the four
        that sort strongest score 0.8588 together, while the best four chosen by their JOINT score
        reach 0.8612 — the sort drops `intensity392_ws` (solo 0.8459) for `extended_ws` (solo 0.8462)
        and loses 0.0024 doing it. A member earns its seat by what it adds, not by what it scores alone.

        Missing names are FATAL rather than skipped. Dropping silently would ship a smaller entry than
        the one whose offline score justified the slot, and nothing in the run's output would say which
        ensemble actually predicted — the same class of failure the fingerprint check exists to stop.
        """
        available = {member.identifier: member for member in self.members}
        absent = [name for name in wanted if name not in available]
        if absent:
            raise WeightsError(f"{self.root} holds {sorted(available)} and was asked for {absent}")
        return [available[name] for name in wanted]

    @classmethod
    def find(cls, mounts: "Mounts | None" = None, name: str = "manifest.json") -> "MemberPackage | None":
        """Locate a mounted package, or return None when none is attached.

        The manifest decides, not the filenames beside it: testing for a naming convention makes the
        search agree with whatever the packager last called its files, which is a second definition of
        what a package is.

        `Mounts` does the walking because the competition's image directories have to be PRUNED, not
        filtered: this read `base.rglob(name)` and skipped the study paths it yielded, which pays the
        entire cost of descending into them and then throws the result away.
        """
        finding = mounts or Mounts()
        if not finding.root.is_dir():
            return None
        holding = finding.every(name)
        return cls(holding[0]) if holding else None

    @classmethod
    def spanning(cls, mounts: "Mounts | None" = None, name: str = "manifest.json") -> "MemberPackage | None":
        """Every mounted package at once, as one package, or None when none is attached.

        WHY THIS EXISTS AND `find` IS NOT ENOUGH. The shipped package holds 127 members and about 10 GB.
        `find` takes `holding[0]` and discards the rest, so the only way a NEW member reaches the board
        is to re-push all ten gigabytes — which nobody does per member, which means every offline gain
        is a board gain never. A second dataset holding only new checkpoints, mounted beside the first,
        costs one upload of one file. Nothing else had to change to allow it: `Mounts.every` already
        returns every root holding the name, and a member's `file` is resolved against ITS OWN root, so
        the paths were absolute per root before this method existed.

        A DUPLICATE IDENTIFIER IS FATAL, not first-wins or last-wins. Two roots naming one member is
        either a stale copy or a rebuild, and both resolve to "a vote ran with weights nobody chose".
        Silence there is the `farseat_v1` failure with a different cause: complete kernel, real
        submission, a board row under an entry that was never assembled.
        """
        finding = mounts or Mounts()
        if not finding.root.is_dir():
            return None
        holding = finding.every(name)
        if not holding:
            return None
        spanned = cls(holding[0])
        for root in holding[1:]:
            spanned.absorb(cls(root).members, root)
        return spanned

    def absorb(self, members: list[Member], root: Path) -> None:
        """Take a second mounted root's members onto this shelf, refusing any identifier both hold.

        Takes the MEMBERS and the root rather than the other package: what is merged here is a shelf of
        weights, and none of the other package's behaviour — its loading, its fingerprint check — is
        wanted or used. Passing the object would make this a method more interested in its argument
        than in itself.
        """
        shared = sorted({member.identifier for member in self.members} & {member.identifier for member in members})
        if shared:
            raise WeightsError(
                f"{self.roots} and {root} both hold {shared}, so which weights a seat would load is "
                "undecided — remove the stale copy rather than letting a mount order pick"
            )
        self.members.extend(members)
        self.roots.append(root)

    def load(self, member: Member, backbone: nn.Module, dim: int, device: torch.device) -> StudyModel:
        """Build the member's model, load its weights, and refuse it if the fingerprint disagrees."""
        # The member records BOTH its encoder and the size it read at, and nothing until now checked
        # that the two are compatible: DINOv2 tiles 392 and DINOv3 does not, so a member could name a
        # combination no patch grid can express and only fail inside a kernel.
        ENCODERS[member.variant].require_tiling(member.read_size)
        # THE SHARED REBUILD — including the adapters, which must exist before the weights arrive: a
        # LoRA member's state dict carries adapter keys that only exist once `low_rank` has wrapped the
        # encoder's linears, and loading into the plain architecture fails on unexpected keys at best,
        # at worst returning the pretrained base with no adaptation at all. This path spelled its own
        # subset of the config and so did the ranking tool's; the anatomical prior was the one this
        # half had never heard of.
        model = StudyModel.from_config(member.config, backbone, dim, member.slot_count).to(device)
        model.load_trained(self.state_dict_of(member, device))
        self.check(model, member, device)
        return model

    def state_dict_of(self, member: Member, device: torch.device) -> dict[str, Tensor]:
        """The member's tensors, unwrapped from whatever the checkpoint wraps them in.

        A published checkpoint is a training artefact, not a bare state dict: it carries the weights
        under `model` alongside the epoch, the fold scores, the config, the out-of-fold predictions and
        its own copy of the fingerprint. Loading the wrapper directly fails loudly on unexpected keys —
        which is the good case, and the reason to unwrap explicitly rather than to load non-strictly.
        """
        checkpoint = torch.load(member.file, map_location=device, weights_only=True)
        return checkpoint.get("model", checkpoint)

    def check(self, model: StudyModel, member: Member, device: torch.device) -> float:
        """Compare the model's answer to the seeded question against the stored one.

        Returns the largest absolute difference, so a caller can report how much slack was left.
        """
        got = model.fingerprint(member.read_size, member.group, device)
        if got.shape != member.fingerprint.shape:
            raise WeightsError(
                f"{member.identifier}: fingerprint shape {got.shape} != stored "
                f"{member.fingerprint.shape}: the architecture is not the one these weights were fitted to"
            )
        difference = float(np.abs(got - member.fingerprint).max())
        if difference > FINGERPRINT_TOLERANCE:
            raise WeightsError(
                f"{member.identifier}: fingerprint differs by {difference:.4g} (tolerance "
                f"{FINGERPRINT_TOLERANCE:g}). The weights load but do not compute what they computed "
                f"when fitted — preprocessing, resolution or architecture has moved between the two runs."
            )
        return difference
