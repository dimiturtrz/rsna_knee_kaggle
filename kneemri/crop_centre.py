"""WHERE the physical crop is taken about — the image's middle, or the tissue's.

`SlotReader` cropped 130 mm about `height // 2, width // 2`, which assumes the joint was positioned in
the middle of the acquisition. Measured over 60 studies, the tissue centroid sits a median 7 mm, p90
18 mm and up to 32 mm away from that point, against a 160 mm field of view.

At 130 mm that is survivable — roughly 15 mm of margin, so only the worst tenth clips. At the 80 mm crop
the multi-scale design is gated on (knee-1nn) it is not: a 40 mm half-extent against a joint some
70-80 mm wide is already snug, and an 18 mm error cuts into it. A tighter crop centred on the image
would test "a tight crop that sometimes misses the joint" rather than resolution, and would fail for a
reason that has nothing to do with the hypothesis.

A strategy rather than a fix, because which centre is better is measurable and unmeasured. The image
centre stays the DEFAULT: changing where the crop is taken changes the pixels, so switching silently
would invalidate 99 GB of cached arrays and every member fitted on them, and no fingerprint would catch
it — each is checked against its own reading.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

CROP_MM = 130.0
"""Below the acquired field of view of almost every series here, while still containing the joint.

It lives here rather than with the reader because it is this crop's default extent, and the reader now
takes a crop rather than owning one — putting it beside `SlotReader` made the import cycle that this
split exists to avoid."""

MIN_TISSUE_PIXELS = 100
"""Below this a slice holds too little signal for a centroid to mean anything."""

OTSU_BINS = 256
"""One bin per level of the eight-bit grid the stack is stored on downstream."""


class CropCentre(Protocol):
    """Where the physical crop is taken about. A strategy because the honest answer is measurable."""

    def of(self, stack: NDArray[np.float32]) -> tuple[int, int]:
        """The (row, column) the crop should centre on."""
        ...


@dataclass(frozen=True)
class ImageCentre:
    """The middle of the acquisition — what every cache and every member on disk was built with.

    Kept as the DEFAULT rather than replaced outright: changing where the crop is taken changes the
    pixels, so it would invalidate 99 GB of decoded arrays and every member fitted on them, and no
    fingerprint would catch it because each is checked against its own reading.
    """

    def of(self, stack: NDArray[np.float32]) -> tuple[int, int]:
        """The geometric centre, which assumes the joint was positioned there."""
        return stack.shape[1] // 2, stack.shape[2] // 2


@dataclass(frozen=True)
class TissueCentre:
    """The centre of the imaged tissue, for when the knee is not where the acquisition's middle is.

    MEASURED 2026-08-14 over 60 studies, tissue centroid against the image centre in millimetres: median
    7 mm, p90 18 mm, max 32 mm, against a 160 mm field of view. At the current 130 mm crop that leaves
    about 15 mm of margin and clips only the worst tenth. At the 80 mm crop the multi-scale design is
    gated on, a half-extent of 40 mm against a joint some 70-80 mm wide is already snug, and an 18 mm
    error cuts into the joint — so a tighter crop centred on the IMAGE would test "a tight crop that
    sometimes misses the joint" rather than resolution, and would fail for the wrong reason.

    The knee is the bright mass and the background is near-zero noise, so the split is genuinely bimodal
    and Otsu finds it without a threshold anyone had to choose. The median over slices rather than one
    slice, because a single plane can be an end-of-stack slice that holds almost no tissue.
    """

    def of(self, stack: NDArray[np.float32]) -> tuple[int, int]:
        """The median over slices of each slice's tissue centroid, as (row, column)."""
        centres = [self.centroid(plane) for plane in stack]
        found = [centre for centre in centres if centre is not None]
        if not found:
            return ImageCentre().of(stack)
        rows, columns = zip(*found, strict=True)
        return int(np.median(rows)), int(np.median(columns))

    def centroid(self, plane: NDArray[np.float32]) -> tuple[float, float] | None:
        """One slice's tissue centroid, or None when the slice holds too little signal to locate.

        None rather than the image centre, so a near-empty slice abstains from the median instead of
        dragging it back toward the assumption this class exists to replace.
        """
        mask = plane > self.threshold(plane)
        if mask.sum() < MIN_TISSUE_PIXELS:
            return None
        rows, columns = np.nonzero(mask)
        return float(rows.mean()), float(columns.mean())

    def threshold(self, plane: NDArray[np.float32]) -> float:
        """Otsu's threshold: the split maximising between-class variance, chosen by the data.

        Otsu rather than a percentile because a percentile is a number someone picks and then defends;
        this one is derived from each slice's own histogram, and MRI background against tissue is the
        bimodal case Otsu is exactly right for. How much of the frame the knee fills varies with field
        of view and habitus, which is precisely what a fixed percentile would get wrong.
        """
        finite = plane[np.isfinite(plane)]
        if finite.size == 0 or finite.max() <= finite.min():
            return float("inf")
        counts, edges = np.histogram(finite, bins=OTSU_BINS)
        weights = np.cumsum(counts)
        centres = (edges[:-1] + edges[1:]) / 2.0
        totals = np.cumsum(counts * centres)
        total, mass = weights[-1], totals[-1]
        below, above = weights[:-1], total - weights[:-1]
        usable = (below > 0) & (above > 0)
        if not usable.any():
            return float(finite.mean())
        mean_below = totals[:-1][usable] / below[usable]
        mean_above = (mass - totals[:-1][usable]) / above[usable]
        between = below[usable] * above[usable] * (mean_below - mean_above) ** 2
        return float(centres[:-1][usable][int(np.argmax(between))])


CENTRES: dict[str, type[CropCentre]] = {"image": ImageCentre, "tissue": TissueCentre}
"""Selectable crop centres, by the name a cache and a manifest record.

A declaration-only dispatch table rather than a bucket module: it is the vocabulary a config speaks, and
`SCHEMES` in `slots.py` is the same shape for the same reason. "image" is the historic behaviour and the
name every cache and member on disk means without saying so.
"""

DEFAULT_CENTRE = "image"
"""What every cache built before the crop centre could vary was taken about."""


@dataclass(frozen=True)
class Crop:
    """A physical crop: how much to take, and about what point.

    One object because the two are not free to vary separately in any way that matters. The extent
    decides how much margin there is for a mis-centred window, and the centre decides whether that
    margin is needed — at 130 mm about the image centre the p90 error costs nothing, and at 80 mm it
    cuts into the joint. Passing them one by one is the shape `Window`, `Split` and `Read` were each
    created to fix in this package, and the argument-count gate said so the moment the second one
    arrived.

    Both defaults are the historic behaviour, so a caller that says nothing gets the pixels every cache
    on disk already holds.
    """

    mm: float = CROP_MM
    centre: str = DEFAULT_CENTRE

    def suffix(self) -> str:
        """What this crop adds to a cache's name — empty for the historic one.

        Empty by design: those arrays are memory-mapped by running trainers and every member on disk was
        fitted on them, so the default must keep resolving to the file it always did.
        """
        parts = []
        if self.mm != CROP_MM:
            parts.append(f"{self.mm:g}mm")
        if self.centre != DEFAULT_CENTRE:
            parts.append(self.centre)
        return "".join(f"_{part}" for part in parts)

    def located(self) -> CropCentre:
        """The strategy this crop names, built."""
        return CENTRES[self.centre]()
