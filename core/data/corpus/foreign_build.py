"""The one decode every foreign corpus shares: array in, `StudyCache`-shaped arrays out.

Rijeka, MRNet, KMAR and fastMRI differ only in where a volume's pixels come from and which slot they
fill. Everything after that — the band, the physical crop, the percentile scaling, the memmap, the mask
and the order the files land in — is one procedure, and a corpus that ran its own copy of it would be a
corpus whose cache could drift from the others without any single file looking wrong.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float, UInt8
from torch import Tensor

from core.data.corpus.study_cache import StudyCache

logger = logging.getLogger(__name__)

PROGRESS_EVERY = 50
"""Decoded studies between progress lines — every foreign build runs unattended, from Rijeka's 736 volumes
at roughly a second each to MRNet's 1,250 exams at three planes each."""


@dataclass(frozen=True)
class SlotVolume:
    """One acquisition's pixels, the slot they fill, and the in-plane spacing they were acquired at."""

    slot: int
    pixels: Float[np.ndarray, "depth rows columns"]
    spacing: float


@dataclass(frozen=True)
class ForeignBuild:
    """Decodes a foreign corpus into the cache it names, one study of `SlotVolume`s at a time."""

    cache: StudyCache

    def stacked(self, pixels: Float[np.ndarray, "depth rows columns"], spacing: float) -> UInt8[Tensor, "s h w"]:
        """One volume as the cache holds it: band-sampled, physically cropped, percentile-scaled.

        Every step is `SlotReader`'s, called on an array instead of on a directory of DICOMs. That is the
        whole point — a corpus normalised its own way would arrive at the encoder in a different
        intensity space, and the transfer this exists for would be measuring the difference between two
        normalisations rather than between two datasets.

        The percentile scaling is also what makes raw magnitudes safe to ignore: fastMRI's reconstructions
        run to a maximum near 8.5e-4 rather than to a DICOM integer range, and a fixed-window scaling would
        return an empty image.
        """
        reader = self.cache.reader()
        sampled = pixels[reader.band_indices(len(pixels), self.cache.slices)]
        taken = reader.window(sampled, spacing)
        cropped = sampled if taken is None else sampled[:, taken[0], taken[1]]
        return reader.sized(cropped, self.cache.image_size)

    def write(self, identifiers: list[str], studies: Iterable[Iterable[SlotVolume]]) -> None:
        """Write `images_<key>.npy`, `index_<key>.json` and — last — `mask_<key>.npy`.

        The mask goes last for the reason `StudyCache.build` writes it last: its presence is what says the
        arrays beside it are complete, so a build that dies half way leaves a cache that reads as unbuilt
        rather than as a corpus with black studies in it. A slot is marked present exactly when a volume
        was written into it, so the mask can never call a black slot acquired.
        """
        count = len(identifiers)
        self.cache.prepare()
        images = np.lib.format.open_memmap(
            self.cache.images_path, mode="w+", dtype=np.uint8, shape=self.cache.shape(count)
        )
        mask = np.zeros((count, self.cache.slots), dtype=np.float32)
        for index, volumes in enumerate(studies):
            for volume in volumes:
                images[index, volume.slot] = self.stacked(volume.pixels, volume.spacing).numpy()
                mask[index, volume.slot] = 1.0
            if index % PROGRESS_EVERY == 0:
                logger.info(f"{index}/{count} decoded")
        images.flush()
        self.cache.record(identifiers)
        self.cache.report(mask)
        np.save(self.cache.mask_path, mask)
        logger.info(f"wrote {self.cache.shape(count)} uint8 -> {self.cache.images_path}")

    @staticmethod
    def write_table(table: pd.DataFrame, destination: Path, columns: Iterable[str]) -> None:
        """Write a labels table and log the level counts of each supervised column it carries."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination, index=False)
        logger.info(f"{len(table)} rows -> {destination}")
        for column in (column for column in columns if column in table):
            logger.info(f"{column}: {table[column].value_counts().sort_index().to_dict()}")
