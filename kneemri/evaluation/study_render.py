"""Render NAMED studies out of the pixel cache, so a person can look at the ones a number pointed at.

`python -m kneemri.evaluation.study_render --out <dir> <StudyInstanceUID> …` writes one
`<uid>.png` per study: a row per populated slot, labelled with the slot's name, and `--columns`
slices spaced evenly through each stack.

WHY IT EXISTS SEPARATELY FROM `locator_audit`. That tool also draws grids, but it samples studies at
RANDOM over the presence mask and needs a locator checkpoint to draw its overlays. Neither is usable
when the question is "the member is confidently wrong on THIS study — what is in it?", which is the
question every residual reading ends on. Rendering a named study had been a throwaway script twice.

IT DRAWS THE PIXELS THE MODEL WAS FED, not the DICOM. The array is the cache the trainer reads, so
what a reader sees here is post-routing, post-band, post-crop and resized — which is the point: a
lesion absent from this page was absent from the model's input, and that is a statement about the
reading, not about the acquisition.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from jaxtyping import UInt8
from numpy.typing import NDArray

from core import cli
from core.data.corpus.cache_flags import CacheFlags
from core.data.corpus.study_cache import StudyCache

logger = logging.getLogger(__name__)

TILE = 288
"""Side of one rendered slice. Above the 336 cache it would upsample; below ~256 a meniscus is mush."""
COLUMNS = 4
LABEL_AT = (8, 24)
LABEL_SCALE = 0.7
LABEL_WEIGHT = 2
WHITE = 255


@dataclass(frozen=True)
class StudyRender:
    """One cache, its study order resolved, ready to draw any study named in it."""

    cache: StudyCache
    images: NDArray[np.uint8]
    mask: NDArray[np.float32]
    order: dict[str, int]

    @staticmethod
    def over(cache: StudyCache) -> "StudyRender":
        """Memory-map the cache and index its study order by UID."""
        corpus, recorded = cache.load()
        return StudyRender(cache, corpus.images, corpus.mask, {uid: row for row, uid in enumerate(recorded)})

    def page(self, study: str, columns: int = COLUMNS) -> UInt8[np.ndarray, "h w"]:
        """A row per populated slot, each row `columns` slices spaced evenly through that stack."""
        row = self.order[study]
        names = self.cache.scheme_slots
        rendered = [
            self._strip(self.images[row, slot], names[slot], columns) for slot in np.flatnonzero(self.mask[row] > 0)
        ]
        if not rendered:
            message = f"{study} has no populated slot in {self.cache.key}"
            raise ValueError(message)
        return np.concatenate(rendered, axis=0)

    def write(self, study: str, out: Path, columns: int = COLUMNS) -> Path:
        """Draw `study` and leave it at `<out>/<study>.png`."""
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{study}.png"
        cv2.imwrite(str(path), self.page(study, columns))
        return path

    @staticmethod
    def _strip(stack: UInt8[np.ndarray, "s h w"], name: str, columns: int) -> UInt8[np.ndarray, "h w"]:
        picks = np.linspace(0, stack.shape[0] - 1, columns).round().astype(int)
        strip = np.concatenate([cv2.resize(np.ascontiguousarray(stack[s]), (TILE, TILE)) for s in picks], axis=1)
        cv2.putText(strip, name, LABEL_AT, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, WHITE, LABEL_WEIGHT)
        return strip


def main() -> None:
    """Render named studies out of the pixel cache the trainer reads."""
    command = cli.Command(__doc__, logger)
    CacheFlags.add_to(command.parser)
    command.parser.add_argument("studies", nargs="+", help="StudyInstanceUIDs, as they appear in the cache index")
    command.parser.add_argument("--out", type=Path, required=True, help="directory the PNGs are written to")
    command.parser.add_argument("--columns", type=int, default=COLUMNS, help="slices drawn per slot")
    arguments = command.parsed()
    root = command.rooted(arguments)
    render = StudyRender.over(CacheFlags.cache(arguments, root.processed()))
    command.report([str(render.write(study, arguments.out, arguments.columns)) for study in arguments.studies])


if __name__ == "__main__":
    main()
