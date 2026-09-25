"""Where each study's evidence sits, measured by blanking a grid of patches one at a time.

`python -m kneemri.evaluation.patch_maps --checkpoint <member>.pt --labels <csv> --out <dir>` writes
`<member>_g<grid>.patches.npz` — a `(studies, labels, grid, grid)` array of per-study probability drops
— and prints the gate that decides whether the array is worth anything.

WHY THIS AND NOT THE SWEEPS NEXT DOOR: `attribution.region_importance` already asks centre-versus-border
and answered it once, on an August member, at 0.8021 against 0.8006 — the centre and the border were
worth the same. That reading is either "the model cannot resolve the joint and leans on coarse context"
or "the signal is simply spread out", and a two-cell measurement cannot tell them apart. Sixteen cells
can, and unlike an AUC delta the result carries a per-study coordinate, which is the only thing a
locator could be fitted on.

THE GATE, PRE-REGISTERED: the share of a label's map mass lying in the central half of the side — a
quarter of the area, so `CENTRE_SHARE_NULL` under a diffuse read. Well above it, with per-study
centroids that agree, means the model's own opinion about WHERE is worth distilling. At the null the
opinion is noise, and a locator trained on it would crop to skin, coil and air while every downstream
number stayed green.

The maps are a WEAK label and nothing here pretends otherwise: this supervision never says where a
finding is, so there is no ground truth in this repository to check a map against. The check that does
exist is external — `kneemri.evaluation.locator_audit` measures a locator's landmark centroid error in
millimetres on SKM-TEA, which is instrument use of a corpus nothing may ship a derivative of.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy.typing import NDArray

from core import cli
from core.data.corpus.folds import ReportFolds
from core.data.labels.targets import Targets
from core.findings import TARGETS
from core.wakelock import Wakelock
from kneemri.device import VisibleCards
from kneemri.evaluation.attribution import GRID, Attribution, Holdout
from kneemri.inference.holdout_predictions import HoldoutPredictions
from kneemri.inference.members import Member
from kneemri.models.backbone import ENCODERS
from kneemri.models.variants import SMALL

logger = logging.getLogger(__name__)

CENTRE_SHARE_NULL = 0.25
"""Share of the mass the central half of the side holds when a map is diffuse — it is a quarter of the area."""

FINEST_GRID = 4
"""Coarsest grid that HAS a central half.

`centre_mass` takes the middle `grid - 2 * (grid // 4)` cells of the side, and at `grid = 2` that
floor is zero: the centre becomes the whole map and every label reads a share of exactly 1.000. That
is not a pass, it is the question failing to be asked — and it printed as twelve of twelve labels
above the null on 2026-09-16 before anything caught it."""

POSITIVE = 0.5
"""Truth above this counts the study as carrying the finding, for `label_contrast` alone.

It splits at the midpoint rather than at a tuned operating point because nothing here is a decision:
the two groups only have to be the studies that have the finding and the studies that do not."""

STUDIES = 200
"""Studies to sweep by default.

The sweep is `grid ** 2 + 1` full inference passes over this many studies, and one pass costs 4.276 s a
study on cpu (measured 2026-09-16, `labelwise_dense_drop40_long_s2026`, 336 px, six slots, twenty
slices). Two hundred at a four-cell grid is four hours unattended; the whole 909-study holdout at eight
cells is sixty-nine, which is a card's question rather than a processor's."""


@dataclass(frozen=True)
class TissueNull:
    """What a centre share reads when nothing has been learned — the denominator `CENTRE_SHARE_NULL` is not.

    `CENTRE_SHARE_NULL` is the share a map holds if every cell is worth the SAME, and no map over a knee
    can be, because the cells are not the same: a middle cell is joint and a corner is air, coil and
    skin. Blanking the middle therefore removes more signal than blanking a corner for a model that
    learned nothing at all, so a centre share above 0.25 is the expected reading rather than a finding.
    Measured against the pixels instead, the question becomes the one worth asking — does the model
    concentrate MORE than the anatomy already does.

    This is a PROXY and is not sold as anything else: it is the share of image intensity a cell holds,
    not the share of held-out score, and those coincide only if the model's reliance on a region is
    linear in how much tissue is there. The exact control is the same sweep under a randomly initialised
    head, which costs a second run of the hours this one cost. The proxy is free — it touches no model
    and runs no forward pass — and it is strictly better than comparing against a flat field.
    """

    shares: Float[NDArray[np.float64], "g g"]

    @classmethod
    def of(cls, images: UInt8[NDArray[np.uint8], "b s d h w"], grid: int) -> "TissueNull":
        """The share of total image intensity lying in each cell of a `grid` x `grid` tiling."""
        edges = np.linspace(0, images.shape[-1], grid + 1).astype(int)

        # Summed per cell rather than per study: the confound is a property of where tissue sits in the
        # field of view, which is the same question for every study in a registered crop.
        def cell(row: int, col: int) -> float:
            return float(images[..., edges[row] : edges[row + 1], edges[col] : edges[col + 1]].sum())

        mass = np.asarray([[cell(row, col) for col in range(grid)] for row in range(grid)], dtype=np.float64)
        return cls(mass / mass.sum())

    def centre_share(self) -> float:
        """The share the central half of the side holds — the number a label's centre share must BEAT."""
        grid = self.shares.shape[-1]
        low, high = grid // 4, grid - grid // 4
        return float(self.shares[low:high, low:high].sum())


@dataclass(frozen=True)
class PatchMaps:
    """A patch sweep's output, read four ways: mass, place, spread, and whether the label matters."""

    maps: Float[NDArray[np.float32], "b o g g"]
    truth: Float[NDArray[np.float32], "b o"] | None = None
    """The holdout labels, for `label_contrast` alone — the other three readings never need them."""

    null: float = CENTRE_SHARE_NULL
    """The centre share to read the labels against.

    Defaults to the flat-field value, which is the WEAK reading and says so in `TissueNull`: pass that
    class's `centre_share` to compare the model against the anatomy rather than against a uniform field.
    """

    def centre_mass(self) -> dict[str, float]:
        """Per label, the share of the map's mass in the central half of the side.

        Absolute value, because a cell the model does BETTER without is still a cell it was using. The
        number is compared against `self.null` and nothing else: an absolute share means little on its
        own, and the null is what makes it a measurement.
        """
        grid = self.maps.shape[-1]
        if grid < FINEST_GRID:
            message = f"a {grid}x{grid} grid has no central half to measure; use {FINEST_GRID} or finer"
            raise ValueError(message)
        low, high = grid // 4, grid - grid // 4
        mass = np.abs(self.maps).sum(axis=0)
        centre = mass[:, low:high, low:high].sum(axis=(1, 2))
        total = mass.sum(axis=(1, 2))
        return {label: float(centre[index] / total[index]) for index, label in enumerate(TARGETS)}

    def centroids(self) -> dict[str, tuple[float, float, float]]:
        """Per label, the mean per-study centroid in cell units, and how much studies disagree about it.

        The centroid is taken per STUDY and then averaged, never off the pooled map: pooling first would
        report a confident centre for a label whose studies each point somewhere different, which is
        exactly the failure this is here to catch. The third number is the standard deviation of those
        per-study centroids — the agreement, and the thing a locator would be fitted against.
        """
        grid = self.maps.shape[-1]
        axis = np.arange(grid, dtype=np.float64)
        mass = np.clip(self.maps, 0.0, None).astype(np.float64)
        weight = mass.sum(axis=(2, 3), keepdims=True)
        share = mass / np.where(weight > 0.0, weight, 1.0)
        rows = (share.sum(axis=3) * axis).sum(axis=2)
        columns = (share.sum(axis=2) * axis).sum(axis=2)
        carried = weight[..., 0, 0] > 0.0
        return {
            label: (
                float(np.nanmean(np.where(carried[:, index], rows[:, index], np.nan))),
                float(np.nanmean(np.where(carried[:, index], columns[:, index], np.nan))),
                float(np.nanstd(np.where(carried[:, index], rows[:, index], np.nan))),
            )
            for index, label in enumerate(TARGETS)
        }

    def label_contrast(self) -> dict[str, float]:
        """Per label, how far the positive studies' centroid sits from the negative studies' centroid.

        THE CONTROL THE CENTRE SHARE NEEDS. A centred map is not evidence of a learned anatomical read:
        blanking a middle cell removes more tissue and more signal than blanking a corner, so a model
        that had learned nothing at all would still produce maps that peak in the middle. That confound
        is a property of the PIXELS, and it is identical for a study with a finding and a study without.
        A displacement between the two is not — it can only come from the label.

        Reported in cells, so it is read against `centroids`' own per-study disagreement: a separation
        smaller than the scatter is not a separation. A label with no positives reports NaN rather than
        a number nothing supports.
        """
        if self.truth is None:
            message = "label_contrast needs the holdout labels; construct PatchMaps with truth"
            raise ValueError(message)
        grid = self.maps.shape[-1]
        axis = np.arange(grid, dtype=np.float64)
        mass = np.clip(self.maps, 0.0, None).astype(np.float64)
        weight = mass.sum(axis=(2, 3), keepdims=True)
        share = mass / np.where(weight > 0.0, weight, 1.0)
        places = np.stack([(share.sum(axis=3) * axis).sum(axis=2), (share.sum(axis=2) * axis).sum(axis=2)], axis=-1)
        separations = {}
        for index, label in enumerate(TARGETS):
            positive = self.truth[:, index] > POSITIVE
            if not positive.any() or positive.all():
                separations[label] = float("nan")
                continue
            moved = places[positive, index].mean(axis=0) - places[~positive, index].mean(axis=0)
            separations[label] = float(np.hypot(*moved))
        return separations

    def lines(self) -> list[str]:
        """The gate as a person reads it — every label, its share, and whether it cleared the null."""
        shares, places = self.centre_mass(), self.centroids()
        moved = self.label_contrast() if self.truth is not None else dict.fromkeys(TARGETS, float("nan"))
        grid = self.maps.shape[-1]
        lines = [
            f"grid {grid}x{grid} over {self.maps.shape[0]} studies",
            f"centre share of the null is {self.null:.3f} — that is what these are read against"
            + ("" if self.null != CENTRE_SHARE_NULL else " (FLAT FIELD, not the tissue null — the weak reading)"),
            "'moved' is positive-minus-negative centroid separation, the reading the centre share cannot do",
            "",
        ]
        for label in TARGETS:
            row, column, spread = places[label]
            verdict = "concentrated" if shares[label] > self.null else "DIFFUSE"
            carries = "label-driven" if moved[label] > spread else "not over the scatter"
            lines.append(
                f"  {label:<18} centre {shares[label]:.3f}  centroid ({row:.2f}, {column:.2f})  "
                f"scatter {spread:.2f}  moved {moved[label]:.2f}  {verdict}, {carries}"
            )
        cleared = sum(1 for label in TARGETS if shares[label] > self.null)
        over = sum(1 for label in TARGETS if moved[label] > places[label][2])
        lines.append("")
        lines.append(f"{cleared} of {len(TARGETS)} labels above the null")
        lines.append(f"{over} of {len(TARGETS)} labels separate positives from negatives by more than their scatter")
        return lines


def main() -> None:
    """Sweep one member's holdout with a patch grid and write the maps beside the gate."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--checkpoint", type=Path, required=True)
    command.parser.add_argument("--labels", type=Path, required=True)
    command.parser.add_argument("--out", type=Path, required=True)
    command.parser.add_argument("--grid", type=int, default=GRID)
    command.parser.add_argument("--studies", type=int, default=STUDIES)
    command.parser.add_argument("--split", choices=("holdout", "train"), default="holdout")
    command.parser.add_argument("--offset", type=int, default=0)
    command.parser.add_argument("--fold", type=int, default=0)
    command.parser.add_argument("--folds", type=int, default=5)
    command.parser.add_argument("--split-seed", type=int, default=2026)
    command.parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    arguments = command.parsed()
    if arguments.grid < FINEST_GRID:
        # Refused HERE rather than in the reading, because the reading runs after the sweep and the
        # sweep is the hours.
        command.parser.error(f"--grid {arguments.grid} has no central half; {FINEST_GRID} is the coarsest gate")

    root = command.rooted(arguments)
    member = Member.from_checkpoint(arguments.checkpoint)
    corpus, identifiers = member.cache(root.processed()).load()
    folds = ReportFolds(arguments.folds, arguments.split_seed)
    train, held = folds.holdout(folds.assign(str(root.train_table()), identifiers), arguments.fold)
    # Blocking bounds the SUBSET COPY below, not the cache load, which is the whole corpus either way:
    # all 3636 train studies of (6, 20, 336, 336) copy 49 GB on top of it and do not fit.
    chosen = train if arguments.split == "train" else held
    holdout = chosen[arguments.offset : arguments.offset + arguments.studies]

    truth = Targets.load(arguments.labels, root.train_table(), identifiers).scores[holdout]
    saved = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    encoder = ENCODERS[str(saved.get("config", {}).get("variant", SMALL))]

    device = torch.device("cpu") if arguments.device == "cpu" else VisibleCards().device()
    predictions = HoldoutPredictions(device)
    model = predictions.load(arguments.checkpoint, encoder.load(), encoder.dim, member.slot_count)
    window = member.window(corpus.slices, overlap=True)
    data = Holdout(corpus.images[holdout], corpus.mask[holdout], truth, window)

    logger.info("device %s  studies %d  grid %d  images %s", device, len(holdout), arguments.grid, data.images.shape)
    # Every cell is a full inference pass over the whole subset, so the run is hours long on a processor
    # and unattended — which is what the idle timer suspends (knee-wyzj).
    with Wakelock():
        maps = Attribution(predictions).patch_importance(model, data, grid=arguments.grid)

    arguments.out.mkdir(parents=True, exist_ok=True)
    block = "" if arguments.split == "holdout" else f"_{arguments.split}{arguments.offset}"
    destination = arguments.out / f"{arguments.checkpoint.stem}_g{arguments.grid}{block}.patches.npz"
    tissue = TissueNull.of(data.images, arguments.grid)
    # `holdout` indexes THIS cache's order; the consumer joins a trainer's corpus by StudyInstanceUID,
    # so the sweep has to record the names rather than the seats it happened to find them in.
    named = np.asarray([identifiers[position] for position in holdout])
    np.savez_compressed(destination, maps=maps, studies=named, truth=truth, tissue=tissue.shares)
    logger.info("wrote %s", destination)
    for line in PatchMaps(maps, truth, tissue.centre_share()).lines():
        logger.info("%s", line)


if __name__ == "__main__":
    main()
