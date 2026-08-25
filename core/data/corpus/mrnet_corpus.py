"""The MRNet (Stanford) corpus, read into the arrays this project's trainer already loads.

A THIRD CORPUS on the same three mechanisms that already admit a foreign one — `ReportFolds.assign`
folding an unknown identifier on itself, `Targets.pair` weighting an absent column at 0.0, and
`train.py` never reading the NaN rows it reindexes. `rijeka_corpus` carries that audit; this module
does not repeat it, and is small for the same reason.

WHAT IT ADDS OVER RIJEKA, which is the whole argument for paying for it. A Rijeka exam is one sequence
and fills ONE slot of six, so a flat reading there has two explanations that cannot be told apart:
foreign data does not transfer, or a knee with five zero slots is a degenerate input. MRNet ships
sagittal, coronal and axial per exam and fills THREE, so it separates them. See `slots.MRNET_SLOTS`
for which slot each plane lands in and on whose authority.

IT SUPERVISES ACL AND NOTHING ELSE, and the omissions are deliberate:

  * `meniscus` does NOT distinguish medial from lateral, while our table carries `Medial Meniscus` and
    `Lateral Meniscus` as separate findings. Writing one label into both would teach a medial tear as
    lateral-positive — the same class of error `rijeka_corpus.labels` refuses when it declines to write
    0.5 into eleven columns nobody read.
  * `abnormal` has no column of ours to land in. It is a study-level flag, not one of the twelve.

1,370 exams were published; 1,250 are downloadable (1,130 train + 120 validation) and the 120-exam
test split is withheld. The join below is over what the archive actually holds, never over the paper's
figure — the denominator is the thing this project has got wrong most often.
"""

import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import UInt8
from numpy.typing import NDArray
from torch import Tensor

from core.data.corpus.study_cache import StudyCache
from core.findings import STUDY

logger = logging.getLogger(__name__)

VOLUME_SUFFIX = ".npy"
SPLITS: tuple[str, ...] = ("train", "valid")
"""Both downloadable splits are TRAINING data here. We never score on them — the ruler is our own
fold-0 holdout — so holding 120 exams back would cost data and buy nothing."""

PLANES: tuple[str, ...] = ("sagittal", "axial", "coronal")
"""The archive's directory names, in the order of `slots.MRNET_SLOTS`. The two are one fact and must
stay aligned: position i of this tuple is written into slot i of the cache."""

PREFIX = "mrnet-"
"""Prefixed onto the archive's case number, and not decoration.

The cases are zero-padded numerals, so a merged labels table round-tripped through pandas reads `0000`
back as the integer 0 and stops joining to the index. The prefix also makes `MergedCache._disjoint`'s
guarantee hold by construction rather than by luck.
"""

ACL_LABEL = "acl"
ACL = "ACL"
CONFIDENCE = f"{ACL}__conf"

VOLUME_DEPTH = 3
"""How many path components a volume is identified by: `<split>/<plane>/<case>.npy`. Anything shallower
is a label table or an archive stub, never pixels."""

PROGRESS_EVERY = 50
"""Decoded exams between progress lines — 1,250 of them at three planes each runs unattended."""

MRNET_SPACING = 0.625
"""Millimetres per pixel in plane — DERIVED, because the corpus records no spacing.

Bien et al. resampled every exam to a 256 px grid, and a knee is acquired at a median 160 mm field of
view (`slots.DETAIL_MM` carries our own measurement of that). 160/256 gives 0.625.

IT IS WEAKER EVIDENCE THAN `RIJEKA_SPACING`, and saying so is the point: Rijeka ships ACL bounding
boxes whose median size checks the derivation against a physical length, and MRNet ships no such
quantity at all. What can be said is that the value is load-bearing rather than cosmetic — the
canonical read crops a constant 130 mm, which is 208 px of 256 here, while 0.5 mm/px would want 260 and
`SlotReader.window` would decline to crop at all, handing the encoder a different mm/pixel than every
member was fitted at. The two branches are far apart, so this is a value to re-examine if the corpus
ever reads worse than a random-crop control rather than one to accept quietly.
"""

MRNET_LEVEL: tuple[float, ...] = (0.08, 0.82)
"""What a negative and a positive ACL row are worth on our level scale.

The labels were extracted by hand from the clinical reports, which is our OWN corpus's provenance
rather than Rijeka's double-blind image read — so they belong on the same scale our reader emits onto,
read through `Supervision.MEASURED_AGREEMENT`:

  negative -> 0.08.  The report was read for the ACL and states no tear. That is the confident-absence
                     level, agreed with 95.9% of the time when read as an absence claim.
  positive -> 0.82.  A stated tear, agreed with 68.8% of the time. NOT 0.94: this label is binary and
                     lumps partial with complete, and 0.94 is what our reader reserves for an unhedged
                     severe finding. Taking the middle confident-yes level is what a label carrying no
                     severity can honestly claim.
"""

REPORT_CONFIDENCE = 0.95
"""The weight a populated cell pulls with — our own report-derived cells carry exactly this, and these
labels come from the same kind of text by the same kind of extraction. Rijeka's 1.0 is not transferable:
that corpus is an expert reading the pixels."""


@dataclass(frozen=True)
class MRNetExam:
    """One exam: its identifier, where each plane's pixels are, and what the report said about the ACL."""

    identifier: str
    members: dict[str, str]
    acl: int

    def level(self) -> float:
        """This verdict on our level scale — see `MRNET_LEVEL` for why each value is what it is."""
        return MRNET_LEVEL[self.acl]

    def present(self, slots: int) -> list[float]:
        """The presence flag per slot: 1.0 where this exam's plane ships, 0.0 where it does not.

        A complete download fills all three. A truncated one does not, and the mask is what lets the
        head's softmax renormalise over what is actually there instead of attending to black pixels the
        mask calls acquired.
        """
        return [1.0 if PLANES[position] in self.members else 0.0 for position in range(slots)]


class MRNetCorpus:
    """Reads the MRNet archive into a `StudyCache`-shaped cache and a labels table beside it.

    The archive is read in place rather than extracted: it is 5.7 GB of arrays that are each used once,
    and a second copy on disk buys nothing.
    """

    def __init__(self, archive: Path, cache: StudyCache, spacing: float = MRNET_SPACING) -> None:
        self.archive = archive
        self.cache = cache
        self.spacing = spacing

    def volumes(self) -> list[MRNetExam]:
        """The exams whose PIXELS ship, in split then case order, joined against the archive's members.

        THE JOIN IS THE DENOMINATOR. The label tables name every case in their split; the archive is
        what holds the arrays. Trusting the table's length would report rates over exams that are not
        here — the paper's 1,370 against the 1,250 that download.
        """
        with zipfile.ZipFile(self.archive) as archive:
            names = archive.namelist()
            labels = {split: self._labels_of(archive, names, split) for split in SPLITS}
        members = self._members(names)
        exams = [
            MRNetExam(identifier=f"{PREFIX}{case}", members=members[(split, case)], acl=int(verdict))
            for split in SPLITS
            for case, verdict in labels[split].items()
            if (split, case) in members
        ]
        published = sum(len(table) for table in labels.values())
        logger.info(f"{len(exams)} of {published} labelled exams ship here: {self._share(published, len(exams))}")
        # The truncation is the CACHE's rule and not this module's: `first` is in the cache's name, so a
        # caller that sliced its own list would be free to write 60 exams under the name that promises
        # 1,250. Asking `enrolled` is what makes the file say what it holds.
        kept = set(self.cache.enrolled([exam.identifier for exam in exams]))
        return [exam for exam in exams if exam.identifier in kept]

    @staticmethod
    def _share(published: int, present: int) -> str:
        """The join as a percentage of what the label tables named, so no caller computes it to be honest."""
        return f"{100.0 * present / max(published, 1):.1f}% of the labelled set"

    @staticmethod
    def _members(names: list[str]) -> dict[tuple[str, str], dict[str, str]]:
        """Every volume in the archive, keyed by (split, case) and then by plane.

        Matched on the tail of the path rather than on a fixed root, because the top-level directory
        name differs between the Stanford download and every mirror of it.
        """
        found: dict[tuple[str, str], dict[str, str]] = {}
        for name in names:
            path = Path(name)
            if path.suffix != VOLUME_SUFFIX or len(path.parts) < VOLUME_DEPTH:
                continue
            split, plane = path.parts[-3], path.parts[-2]
            if split in SPLITS and plane in PLANES:
                found.setdefault((split, path.stem), {})[plane] = name
        return found

    @staticmethod
    def _labels_of(archive: zipfile.ZipFile, names: list[str], split: str) -> dict[str, int]:
        """One split's ACL table: case number to 0/1, read from the headerless CSV the corpus ships."""
        wanted = f"{split}-{ACL_LABEL}.csv"
        for name in names:
            if Path(name).name == wanted:
                table = pd.read_csv(io.BytesIO(archive.read(name)), header=None, dtype={0: str})
                return {str(case): int(verdict) for case, verdict in zip(table[0], table[1], strict=True)}
        raise FileNotFoundError(f"{wanted} is not in this archive")

    def stack(self, volume: NDArray[np.float32], slices: int) -> UInt8[Tensor, "s h w"]:
        """One plane's volume as the cache holds it: band-sampled, physically cropped, percentile-scaled.

        Every step is `SlotReader`'s, called on an array instead of on a directory of DICOMs — so this
        corpus arrives at the encoder in one geometry and one intensity space with ours, rather than
        merely in one array shape.
        """
        reader = self.cache.reader()
        sampled = volume[reader.band_indices(len(volume), slices)]
        taken = reader.window(sampled, self.spacing)
        cropped = sampled if taken is None else sampled[:, taken[0], taken[1]]
        return reader.sized(cropped, self.cache.image_size)

    def build(self) -> None:
        """Write `images_<key>.npy`, `index_<key>.json` and — last — `mask_<key>.npy`.

        The mask goes last for the reason every builder here writes it last: its presence is what says
        the arrays beside it are complete, so a build that dies half way leaves a cache that reads as
        unbuilt rather than as a corpus with black exams in it.
        """
        exams = self.volumes()
        self.cache.prepare()
        images = np.lib.format.open_memmap(
            self.cache.images_path, mode="w+", dtype=np.uint8, shape=self.cache.shape(len(exams))
        )
        mask = np.zeros((len(exams), self.cache.slots), dtype=np.float32)
        with zipfile.ZipFile(self.archive) as archive:
            for index, exam in enumerate(exams):
                for position, plane in enumerate(PLANES[: self.cache.slots]):
                    if plane in exam.members:
                        pixels = self._pixels(archive, exam.members[plane])
                        images[index, position] = self.stack(pixels, self.cache.slices).numpy()
                mask[index] = exam.present(self.cache.slots)
                if index % PROGRESS_EVERY == 0:
                    logger.info(f"{index}/{len(exams)} decoded")
        images.flush()
        self.cache.record([exam.identifier for exam in exams])
        self.cache.report(mask)
        np.save(self.cache.mask_path, mask)
        logger.info(f"wrote {self.cache.shape(len(exams))} uint8 -> {self.cache.images_path}")

    @staticmethod
    def _pixels(archive: zipfile.ZipFile, member: str) -> NDArray[np.float32]:
        """One plane's stack as float32. `.npy` inside the zip, so no pickle is ever unpacked."""
        return np.asarray(np.load(io.BytesIO(archive.read(member)), allow_pickle=False), dtype=np.float32)

    def labels(self, destination: Path) -> None:
        """The labels table, carrying the ONE finding this corpus can honestly supervise.

        Absence is the mechanism, not an omission — see this module's own docstring for why `meniscus`
        and `abnormal` are left out rather than mapped onto columns they do not mean.
        """
        exams = self.volumes()
        table = pd.DataFrame(
            {
                STUDY: [exam.identifier for exam in exams],
                ACL: [exam.level() for exam in exams],
                CONFIDENCE: REPORT_CONFIDENCE,
            }
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination, index=False)
        logger.info(f"{len(table)} rows -> {destination}")
        logger.info(f"levels: {table[ACL].value_counts().sort_index().to_dict()}")
