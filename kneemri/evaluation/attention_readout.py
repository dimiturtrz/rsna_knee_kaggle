"""Where each diagnosis actually looked, against where the anatomy table says it should have.

THIS IS THE ONE ATTRIBUTION IN THIS PROJECT THAT HAS A GROUND TRUTH. `attribution.py` declines pixel
saliency because nothing in the supervision says WHERE a finding is, so a heatmap cannot be checked
against anything. The slot axis is different: the six slots are NAMED acquisitions — a plane, a
weighting, a fat-suppression state — and `core.data.corpus.anatomy` already states which of them each
finding is read on. So an attention weight here is not a picture, it is a claim with an answer:
attention mass on `COR_T1` for a finding the table calls sagittal fluid-weighted is a falsifiable fact
about the model, and `SAG_T1` mass on a marrow-oedema finding says the model is reading a sequence on
which that finding is largely absent.

WHAT THE PRIOR DOES AND DOES NOT MAKE THIS CIRCULAR. A member fitted with `--anatomical-prior` had the
same table added to its attention logits, so agreement is partly built in and only DISAGREEMENT is
informative for it. A member fitted without the prior — which is most of the pool — learned its slot
attention from study-level labels alone, and for those the agreement is the measurement. The report
names which case it is reading rather than leaving it to be remembered.

TWO DENOMINATORS, both stated rather than assumed. Attention renormalises over the slots a study
actually contains, so a slot absent from a study contributes a structural zero: an unconditional mean
confounds "the model does not look there" with "that sequence was not acquired". Per-slot mass is
therefore averaged over the studies where the slot is PRESENT, and the presence rate is reported beside
it. Off-table mass — attention on slots the table gives no affinity at all — is averaged only over the
studies that had somewhere on-table to look, because a study whose only present slots are off-table has
no choice and says nothing about the model.
"""

import logging
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.data.corpus.anatomy import ANATOMY, FindingAnatomy
from core.data.corpus.slots import SLOTS, Slot
from core.findings import TARGETS

logger = logging.getLogger(__name__)

PRESENT = 0.5
"""Midpoint of the {0, 1} presence mask, matching `SlotHead`'s own threshold."""

NO_AFFINITY = 0.0
"""What the anatomy table gives a slot whose PLANE does not show the structure at all.

Not "less conspicuous" — `FindingAnatomy.affinity` returns this only when the plane is neither primary
nor secondary, i.e. the structure is not being viewed. That is what makes off-table mass a claim rather
than a preference.
"""

CONCENTRATED = 0.5
"""Share of one finding's attention on a single slot above which the read counts as collapsed.

Half of everything a diagnosis reads, sitting on one of six acquisitions, is the degenerate-attention
failure attention-MIL is known for and which `attribution.py`'s slot diagnostic already watches for from
the other side (score lost when a slot is withheld). Here it is visible without a second inference pass.
"""


@dataclass(frozen=True)
class SlotMass:
    """One finding's attention over the named slots, beside what the anatomy table says it reads on."""

    finding: str
    mass: dict[str, float]
    """Slot name to mean attention weight, over the studies in which that slot was PRESENT."""
    affinity: dict[str, float]
    """Slot name to `FindingAnatomy.affinity` in [0, 1] — the prior this is being checked against."""
    offtable: float
    """Mean attention on slots of zero affinity, over studies that had an on-table slot to look at."""
    studies: int
    """The denominator of `offtable` — how many studies could have looked on-table and were counted."""

    def dearest(self) -> str:
        """The slot this finding put the most attention on."""
        return max(self.mass, key=lambda slot: self.mass[slot])

    def expected(self) -> list[str]:
        """The slots the anatomy table gives the highest affinity — where the reading should have gone."""
        best = max(self.affinity.values(), default=NO_AFFINITY)
        return [slot for slot, value in self.affinity.items() if value == best and value > NO_AFFINITY]

    def line(self) -> str:
        """This finding's row: where it looked, where the table says it should, and the off-table mass."""
        read = " ".join(f"{slot}={self.mass[slot]:.3f}" for slot in sorted(self.mass, key=lambda s: -self.mass[s]))
        agrees = "agrees" if self.dearest() in self.expected() else "DISAGREES"
        return (
            f"  {self.finding:<17} {read}\n"
            f"    dearest {self.dearest()} vs table {'/'.join(self.expected()) or 'none'} — {agrees}; "
            f"off-table {self.offtable:.3f} over {self.studies} studies"
        )


class AttentionReadout:
    """The attention weights of a member, summarised per finding and set against the anatomy table.

    Pure arithmetic over arrays, so it is testable without a model, a card or a corpus — the same split
    `Attribution` makes between the half that needs a GPU and the half that is a function of numbers.
    """

    def __init__(self, slots: tuple[Slot, ...] = SLOTS, anatomy: dict[str, FindingAnatomy] | None = None) -> None:
        self.slots = slots
        self.anatomy = anatomy if anatomy is not None else ANATOMY

    def affinity(self, finding: str) -> dict[str, float]:
        """What the table says each named slot is worth for this finding.

        A finding absent from the table gets zeros rather than an error, which is `SlotAffinity.matrix`'s
        rule for the same case: a thirteenth target must not silently declare every slot off-table.
        """
        read = self.anatomy.get(finding)
        return {slot.name: read.affinity(slot) if read is not None else NO_AFFINITY for slot in self.slots}

    def presence(self, mask: Float[NDArray[np.float32], "b s"]) -> dict[str, float]:
        """How often each slot was acquired at all — the denominator every mass below is read against."""
        present = mask >= PRESENT
        return {slot.name: float(present[:, index].mean()) for index, slot in enumerate(self.slots)}

    def rows(
        self,
        weights: Float[NDArray[np.float32], "b o s"],
        mask: Float[NDArray[np.float32], "b s"],
    ) -> list[SlotMass]:
        """One row per finding, in target order, each carrying its own two denominators."""
        present = mask >= PRESENT
        return [self._row(finding, index, weights, present) for index, finding in enumerate(TARGETS)]

    def _row(
        self,
        finding: str,
        index: int,
        weights: Float[NDArray[np.float32], "b o s"],
        present: NDArray[np.bool_],
    ) -> SlotMass:
        """This finding's per-slot mass, conditioned on presence, plus its off-table share."""
        affinity = self.affinity(finding)
        mass = {
            slot.name: self._conditional(weights[:, index, position], present[:, position])
            for position, slot in enumerate(self.slots)
        }
        offtable = np.array([affinity[slot.name] <= NO_AFFINITY for slot in self.slots])
        eligible = (present & ~offtable).any(axis=1)
        counted = weights[eligible][:, index][:, offtable]
        return SlotMass(
            finding=finding,
            mass=mass,
            affinity=affinity,
            offtable=float(counted.sum(axis=1).mean()) if counted.size else float("nan"),
            studies=int(eligible.sum()),
        )

    @staticmethod
    def _conditional(column: Float[NDArray[np.float32], " b"], present: NDArray[np.bool_]) -> float:
        """The mean of a slot's weights over the studies that HAVE that slot, or nan if none do."""
        return float(column[present].mean()) if present.any() else float("nan")

    def record(
        self,
        weights: Float[NDArray[np.float32], "b o s"],
        mask: Float[NDArray[np.float32], "b s"],
    ) -> dict[str, object]:
        """The whole reading as plain data, for the artefact the CLI writes outside the repo."""
        presence = self.presence(mask)
        return {
            "slots": [slot.name for slot in self.slots],
            "presence": presence,
            "studies": len(mask),
            "findings": [
                {
                    "finding": row.finding,
                    "mass": row.mass,
                    "affinity": row.affinity,
                    "offtable": row.offtable,
                    "offtable_studies": row.studies,
                    "dearest": row.dearest(),
                    "expected": row.expected(),
                }
                for row in self.rows(weights, mask)
            ],
        }


@dataclass(frozen=True)
class AttentionReadoutReport:
    """The rows as prose, with the verdict each one supports stated rather than left to be inferred."""

    rows: list[SlotMass]
    presence: dict[str, float]
    prior: bool
    """Whether the member was fitted WITH the anatomical prior, which is what agreement means here."""

    def lines(self) -> list[str]:
        """The table, the presence denominators, and what the agreement count is worth for this member."""
        lines = ["slot presence (share of studies in which the acquisition exists):"]
        lines.extend(f"  {slot}: {rate:.3f}" for slot, rate in self.presence.items())
        lines.append("")
        lines.append("attention mass per finding (mean over the studies where the slot is present):")
        lines.extend(row.line() for row in self.rows)
        lines.append("")
        lines.extend(self._verdict())
        return lines

    def _verdict(self) -> list[str]:
        """What the counted rows say, said with the one caveat that decides how to read them."""
        agreeing = [row for row in self.rows if row.dearest() in row.expected()]
        collapsed = [row for row in self.rows if max(row.mass.values(), default=0.0) >= CONCENTRATED]
        offtable = float(np.nanmean([row.offtable for row in self.rows]))
        return [
            f"{len(agreeing)} of {len(self.rows)} findings read most heavily on a slot the table calls primary",
            f"mean off-table mass {offtable:.3f} — attention on planes that do not show the structure",
            f"{len(collapsed)} of {len(self.rows)} findings put over {CONCENTRATED:.0%} of their attention "
            "on one slot, which is the degenerate-attention failure of attention-MIL",
            self._caveat(),
        ]

    def _caveat(self) -> str:
        """Whether agreement is a measurement or a restatement of what the member was given."""
        if self.prior:
            return (
                "THIS MEMBER WAS FITTED WITH THE ANATOMICAL PRIOR: the same table was added to its "
                "attention logits, so agreement is partly built in and only the DISAGREEING rows and the "
                "off-table mass are informative"
            )
        return (
            "this member was fitted WITHOUT the anatomical prior, so its slot attention came from "
            "study-level labels alone and the agreement above is a measurement rather than a restatement"
        )


def main() -> None:
    """Read one member's slot attention over its fold's holdout and check it against the anatomy table."""
    import json  # noqa: PLC0415 - a main() imports what only it needs, as every entrypoint here does
    import logging as log  # noqa: PLC0415 - as above
    from pathlib import Path  # noqa: PLC0415 - as above

    import torch  # noqa: PLC0415 - as above

    from core.data.corpus.folds import ReportFolds  # noqa: PLC0415 - as above
    from core.data.corpus.slots import SCHEMES  # noqa: PLC0415 - as above
    from kneemri.device import VisibleCards  # noqa: PLC0415 - as above
    from kneemri.evaluation.attention_scan import AttentionScan  # noqa: PLC0415 - as above
    from kneemri.inference.holdout_predictions import HoldoutPredictions, ServedModel  # noqa: PLC0415 - as above
    from kneemri.inference.holdout_slab import HoldoutSlab  # noqa: PLC0415 - as above
    from kneemri.inference.members import Member  # noqa: PLC0415 - as above
    from kneemri.models.backbone import ENCODERS  # noqa: PLC0415 - as above

    command = cli.Command(__doc__, log.getLogger(__name__))
    command.parser.add_argument("--checkpoint", type=Path, required=True)
    command.parser.add_argument("--fold", type=int, default=0)
    command.parser.add_argument("--folds", type=int, default=5)
    command.parser.add_argument("--split-seed", type=int, default=2026)
    command.parser.add_argument("--studies", type=int, default=200, help="holdout studies read, for wall-clock")
    command.parser.add_argument("--device", default="auto", help="auto, cpu, or a cuda index")
    arguments = command.parsed()

    root = command.rooted(arguments)
    # THE MEMBER'S OWN READING, never a constant — the defect this repo's other analysis CLIs each had
    # to be fixed for separately: a member names its cache, its window and its slot scheme, and a tool
    # that supplies any of them itself reads different pixels than the member was fitted on.
    member = Member.from_checkpoint(arguments.checkpoint)
    corpus, identifiers = member.cache(root.processed()).load()
    folds = ReportFolds(arguments.folds, arguments.split_seed)
    _, holdout = folds.holdout(folds.assign(str(root.train_table()), identifiers), arguments.fold)
    holdout = holdout[: arguments.studies]

    device = VisibleCards().device() if arguments.device == "auto" else torch.device(arguments.device)
    encoder = ENCODERS[member.variant]
    model = HoldoutPredictions(device).load(arguments.checkpoint, encoder.load(), encoder.dim, member.slot_count)
    images, mask = corpus.images[holdout], corpus.mask[holdout]
    window = member.window(corpus.slices, overlap=True)

    readout = AttentionReadout(SCHEMES[member.scheme])
    weights = AttentionScan(device).weights(ServedModel(model, window, member.precision), HoldoutSlab.of(images, mask))
    record = readout.record(weights, mask)
    # Out of the repo, beside the caches and the checkpoints: this is a run output, not source.
    destination = root.processed() / f"attention_readout_{arguments.checkpoint.stem}.json"
    destination.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

    prior = bool(member.config.get("anatomical_prior", False))
    command.report(AttentionReadoutReport(readout.rows(weights, mask), readout.presence(mask), prior).lines())
    command.report([f"written: {destination}"])


if __name__ == "__main__":
    main()
