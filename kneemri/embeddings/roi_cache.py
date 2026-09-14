"""The ROI cache: every sagittal stack again, cropped to the joint a tissue locator found, plus the box table.

`python -m kneemri.embeddings.roi_cache --checkpoint locator_sag.pt --slices 20 --image-size 336
--band-low 0.1 --band-high 0.9` reads the canonical cache, runs the locator over each study's three
sagittal slots, and writes `images_<key>_roi.npy` (the six canonical slots copied, then the three crops)
beside `labels/roi_boxes_<key>_roi.csv`, one row per acquired sagittal slot. `--cache-studies N` writes
the truncated twin a smoke reads.

WHY FROM THE CACHE AND NOT FROM DICOM. The crop is taken from the 336 px pixels every member already
reads, at 0.387 mm/px, and upsampled. A re-decode at native resolution would add detail, but it is a
DICOM pass this step is not approved for, and the canonical frame is what the box table is stated in.

THE BOX, DERIVED:
  * extent — the 2nd to 98th percentile of every tissue pixel on slices that draw at least `PRESENT`,
    so one stray blob on one slice cannot stretch it;
  * margin — `MARGIN_MM` on every side, the locator's worst non-shape p90 centroid error on SKM-TEA's
    test split (meniscus, 8.8 mm; `interpretations/kneemri/2026-09-14_roi_locator_audit.md`). Femoral
    cartilage reads 18.1 mm at p90, but that error is its centroid sliding along a curved plate the
    extent already spans, so it is not a placement error the margin has to absorb;
  * floor — never narrower than `DETAIL_MM`, the joint's 70-80 mm span, so a knee where only the
    menisci drew still gets the whole joint;
  * square, then shifted (not shrunk) back inside the frame.

A stack drawing tissue on fewer than `LOCATED_SLICES` slices is BLANK: its ROI slot is absent in the
mask rather than a crop of nothing, which is the one thing the head can already express.
"""

import csv
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Int, UInt8
from torch import Tensor

from core import cli
from core.data.corpus.cache_flags import CacheFlags
from core.data.corpus.crop_centre import CROP_MM
from core.data.corpus.skmtea_slices import LOCATOR
from core.data.corpus.slots import DETAIL_MM, ROI_SCHEME, SLOTS, Plane
from core.data.corpus.study_cache import StudyCache
from kneemri.models.backbone import ENCODERS, SMALL
from kneemri.models.locator import BYTE, PRESENT, Locator

logger = logging.getLogger(__name__)

MARGIN_MM = 9.0
LOCATED_SLICES = 3
"""Slices that must draw tissue before a box is believed: one is a blob, three is a joint."""
LOW, HIGH = 0.02, 0.98
LABELS = "labels"
SAGITTAL = tuple(position for position, slot in enumerate(SLOTS) if slot.plane is Plane.SAGITTAL)


@dataclass(frozen=True)
class RoiBox:
    """One stack's square box in its cache frame, and what it was drawn from."""

    top: int
    left: int
    side: int
    located: int
    tissue: int

    @property
    def blank(self) -> bool:
        return self.located < LOCATED_SLICES

    def row(self, study: str, column: int, size: int) -> dict[str, str | int | float]:
        """The box-table row: the box in this frame, and as a share of it so any other frame is one product."""
        shares = {
            f"{name}_share": round(value / size, 4)
            for name, value in (("top", self.top), ("left", self.left), ("side", self.side))
        }
        slot = SLOTS[column]
        return {
            "study": study,
            "slot": slot.name,
            "plane": str(slot.plane),
            **asdict(self),
            **shares,
            "blank": int(self.blank),
        }


@dataclass(frozen=True)
class BoxRule:
    """Margin and floor in pixels of one cache frame."""

    margin: int
    minimum: int
    size: int

    @staticmethod
    def at(size: int) -> "BoxRule":
        millimetres = CROP_MM / size
        return BoxRule(round(MARGIN_MM / millimetres), round(DETAIL_MM / millimetres), size)

    def box(self, drawn: Int[Tensor, "n h w"]) -> RoiBox:
        tissue = drawn > 0
        located = tissue.sum((1, 2)) >= PRESENT
        count = int(located.sum())
        if count < LOCATED_SLICES:
            return RoiBox(0, 0, self.size, count, int(tissue.sum()))
        _, rows, columns = torch.nonzero(tissue[located], as_tuple=True)
        quantiles = torch.tensor([LOW, HIGH], device=drawn.device)
        top, bottom = torch.quantile(rows.float(), quantiles).tolist()
        left, right = torch.quantile(columns.float(), quantiles).tolist()
        side = min(self.size, max(self.minimum, round(max(bottom - top, right - left)) + 2 * self.margin))
        return RoiBox(
            self.placed((top + bottom) / 2, side), self.placed((left + right) / 2, side), side, count, len(rows)
        )

    def placed(self, centre: float, side: int) -> int:
        return int(min(max(round(centre - side / 2), 0), self.size - side))


@dataclass(frozen=True)
class RoiCache:
    """The canonical cache in, the ROI cache and its box table out."""

    source: StudyCache
    destination: StudyCache
    locator: Locator
    device: torch.device

    @property
    def table(self) -> Path:
        return self.source.directory / LABELS / f"roi_boxes_{self.destination.key}.csv"

    @staticmethod
    def crop(stack: UInt8[Tensor, "n h w"], box: RoiBox) -> UInt8[Tensor, "n h w"]:
        window = stack[:, box.top : box.top + box.side, box.left : box.left + box.side].float()
        resized = F.interpolate(window[:, None], size=stack.shape[1:], mode="bilinear", align_corners=False)
        return resized[:, 0].round().clamp(0, BYTE).to(torch.uint8)

    def located(self, stack: UInt8[Tensor, "n h w"], rule: BoxRule) -> RoiBox:
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            return rule.box(self.locator.predict(stack))

    def build(self) -> list[dict[str, str | int | float]]:
        """Write images, index, box table and — last — mask; refuse a destination already on disk."""
        written = self.destination.images_path
        if written.exists():
            raise FileExistsError(written)
        studies = self.source.recorded()[: self.destination.first or None]
        source = np.load(self.source.images_path, mmap_mode="r")
        present = np.load(self.source.mask_path)[: len(studies)]
        rule = BoxRule.at(self.destination.image_size)
        self.destination.prepare()
        images = np.lib.format.open_memmap(
            self.destination.images_path, mode="w+", dtype=np.uint8, shape=self.destination.shape(len(studies))
        )
        mask = np.concatenate([present, np.zeros((len(studies), len(SAGITTAL)), dtype=present.dtype)], axis=1)
        rows = []
        for position, study in enumerate(studies):
            stack = np.array(source[position])
            images[position, : len(SLOTS)] = stack
            for extra, column in enumerate(SAGITTAL):
                if not present[position, column]:
                    continue
                pixels = torch.from_numpy(stack[column]).to(self.device)
                box = self.located(pixels, rule)
                rows.append(box.row(study, column, rule.size))
                if not box.blank:
                    images[position, len(SLOTS) + extra] = RoiCache.crop(pixels, box).cpu().numpy()
                    mask[position, len(SLOTS) + extra] = 1
        images.flush()
        self.destination.record(studies)
        self.write(rows)
        self.destination.report(mask)
        np.save(self.destination.mask_path, mask)
        return rows

    def write(self, rows: list[dict[str, str | int | float]]) -> None:
        table = self.table
        table.parent.mkdir(parents=True, exist_ok=True)
        with table.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def misses(rows: list[dict[str, str | int | float]], studies: int) -> list[str]:
        """Blank boxes per sagittal slot over the studies that acquired it, and studies left with no ROI."""
        lines = []
        for column in SAGITTAL:
            name = SLOTS[column].name
            mine = [row for row in rows if row["slot"] == name]
            blank = sum(int(row["blank"]) for row in mine)
            sides = [int(row["side"]) for row in mine if not row["blank"]]
            lines.append(
                f"{name}: blank {blank}/{len(mine)} acquired ({100 * blank / max(len(mine), 1):.2f}%), "
                f"median side {np.median(sides) if sides else 0:.0f} px"
            )
        located = {row["study"] for row in rows if not row["blank"]}
        lines.append(f"studies with at least one ROI view: {len(located)}/{studies}")
        return lines


def main() -> None:
    command = cli.Command(__doc__, logger)
    CacheFlags.add_to(command.parser)
    command.parser.add_argument("--checkpoint", required=True)
    arguments = command.parsed()
    processed = command.rooted(arguments).processed()
    named = CacheFlags.cache(arguments, processed)
    source, destination = replace(named, first=0), named.under(ROI_SCHEME)
    device = torch.device("cuda")
    locator = Locator.pretrained(ENCODERS[SMALL])
    locator.load_state_dict(torch.load(processed / LOCATOR / arguments.checkpoint, weights_only=True))
    builder = RoiCache(source, destination, locator.to(device).eval(), device)
    rows = builder.build()
    command.report(
        [str(destination.images_path), str(builder.table), *RoiCache.misses(rows, len(destination.recorded()))]
    )


if __name__ == "__main__":
    main()
