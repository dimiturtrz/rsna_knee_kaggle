"""Read a tissue locator two ways: landmark distance on SKM-TEA's test split, and overlays on RSNA studies.

`python -m kneemri.evaluation.locator_audit --checkpoint locator_sag.pt --slot SAG_T1 --out <dir>` prints
the landmark table and writes `<slot>_<k>.png` grids of ten studies, three slices each, for a person to
read.

WHY NOT DICE ALONE: Dice sat at 0.33-0.40 from epoch 10 to epoch 40 while the drawn regions stayed on
the right tissue. The tissues are two to four pixels thick at 336 px and the trunk sees a 24 x 24 grid,
so a boundary one token wide halves the Dice of a correctly placed cartilage. A locator is asked WHERE,
so it is read by where: the distance between the predicted and true centroid of each tissue on each
slice, in millimetres, plus the slices where one side draws the tissue and the other does not.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor

from core import cli
from core.data.corpus.skmtea_slices import LOCATOR, SIZE
from core.data.corpus.slots import SLOTS
from kneemri.models.backbone import ENCODERS, SMALL
from kneemri.models.locator import CLASSES, EVAL_BATCH, PRESENT, Locator, SliceCorpus

logger = logging.getLogger(__name__)

MILLIMETRES = 0.3125 * 512 / SIZE
"""SKM-TEA's 0.3125 mm pixel on a 512 grid, resized to `SIZE`."""
CACHE = "336px_20sl_0.1-0.9band"
STUDIES = 50
PER_GRID = 10
TILE_SLICES = (5, 10, 15)
OPACITY = 0.5
COLOURS = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 120, 255], [255, 255, 0]], dtype=np.float64)
PERCENTILE = 90


@dataclass(frozen=True)
class Landmarks:
    """Per-tissue centroid errors on slices both sides draw, and the slices only one side draws."""

    errors: dict[int, list[float]]
    missed: dict[int, int]
    extra: dict[int, int]
    present: dict[int, int]

    @staticmethod
    def empty() -> "Landmarks":
        tissues = range(1, len(CLASSES))
        counts = [dict.fromkeys(tissues, 0) for _ in range(3)]
        return Landmarks({t: [] for t in tissues}, *counts)

    @staticmethod
    def centroids(regions: Float[Tensor, "n h w"]) -> Float[Tensor, "n 2"]:
        rows = torch.arange(regions.shape[1], device=regions.device, dtype=regions.dtype)
        columns = torch.arange(regions.shape[2], device=regions.device, dtype=regions.dtype)
        area = regions.sum((1, 2)).clamp(min=1)
        return torch.stack([(regions.sum(2) * rows).sum(1) / area, (regions.sum(1) * columns).sum(1) / area], 1)

    def add(self, drawn: Int[Tensor, "n h w"], labels: Int[Tensor, "n h w"]) -> None:
        for tissue in self.errors:
            predicted, truth = (drawn == tissue).float(), (labels == tissue).float()
            seen, real = predicted.sum((1, 2)) > PRESENT, truth.sum((1, 2)) > PRESENT
            distance = (self.centroids(predicted) - self.centroids(truth)).norm(dim=1) * MILLIMETRES
            self.errors[tissue] += distance[seen & real].tolist()
            self.missed[tissue] += int((real & ~seen).sum())
            self.extra[tissue] += int((seen & ~real).sum())
            self.present[tissue] += int(real.sum())

    def lines(self) -> list[str]:
        return [
            f"{CLASSES[t]}: median {np.median(self.errors[t]):.1f} mm  p{PERCENTILE} "
            f"{np.percentile(self.errors[t], PERCENTILE):.1f} mm  missed {self.missed[t]}/{self.present[t]}  "
            f"extra {self.extra[t]}"
            for t in self.errors
            if self.errors[t]
        ]


@dataclass(frozen=True)
class LocatorAudit:
    """One locator on one device, read on SKM-TEA's test landmarks and on RSNA overlays."""

    locator: Locator
    device: torch.device

    def landmarks(self, corpus: SliceCorpus, split: str = "test") -> Landmarks:
        chosen = corpus.indices(split)
        found = Landmarks.empty()
        for start in range(0, len(chosen), EVAL_BATCH):
            part = chosen[start : start + EVAL_BATCH]
            slices, labels = corpus.batch(part, np.zeros_like(part), self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                found.add(self.locator.predict(slices), labels)
        return found

    @staticmethod
    def overlay(slices: UInt8[np.ndarray, "n h w"], drawn: Int[np.ndarray, "n h w"]) -> UInt8[np.ndarray, "h w_n 3"]:
        """Chosen slices side by side, each tissue tinted over the grey image."""
        tiles = []
        for index in TILE_SLICES:
            tile = np.repeat(slices[index, ..., None].astype(np.float64), 3, axis=-1)
            tinted = drawn[index] > 0
            tile[tinted] = (1 - OPACITY) * tile[tinted] + OPACITY * COLOURS[drawn[index][tinted]]
            tiles.append(tile.astype(np.uint8))
        return np.concatenate(tiles, axis=1)

    def rsna_grids(self, processed: Path, slot: str, out: Path) -> list[Path]:
        """`STUDIES` studies holding `slot`, drawn from the first `STUDIES` of a fixed shuffle, `PER_GRID` a grid."""
        images = np.load(processed / f"images_{CACHE}.npy", mmap_mode="r")
        present = np.load(processed / f"mask_{CACHE}.npy", mmap_mode="r")
        column = [s.name for s in SLOTS].index(slot)
        studies = np.sort(np.random.default_rng(1).choice(np.flatnonzero(present[:, column]), STUDIES, replace=False))
        rows = []
        for study in studies:
            stack = np.ascontiguousarray(images[study, column])
            with torch.autocast(self.device.type, dtype=torch.bfloat16):
                drawn = self.locator.predict(torch.from_numpy(stack).to(self.device)).cpu().numpy()
            rows.append(self.overlay(stack, drawn))
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for start in range(0, STUDIES, PER_GRID):
            path = out / f"{slot}_{start}.png"
            cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows[start : start + PER_GRID]), cv2.COLOR_RGB2BGR))
            written.append(path)
        return written


def main() -> None:
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--checkpoint", required=True)
    command.parser.add_argument("--slot", choices=[s.name for s in SLOTS], default=SLOTS[0].name)
    command.parser.add_argument("--out", type=Path, required=True)
    arguments = command.parsed()
    processed = command.rooted(arguments).processed()
    device = torch.device("cuda")
    locator = Locator.pretrained(ENCODERS[SMALL])
    locator.load_state_dict(torch.load(processed / LOCATOR / arguments.checkpoint, weights_only=True))
    locator = locator.to(device).eval()
    audit = LocatorAudit(locator, device)
    command.report(audit.landmarks(SliceCorpus.read(processed / LOCATOR)).lines())
    command.report(str(path) for path in audit.rsna_grids(processed, arguments.slot, arguments.out))


if __name__ == "__main__":
    main()
