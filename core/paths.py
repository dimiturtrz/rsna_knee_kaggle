"""Machine-local data locations.

One data root per machine, recorded in the gitignored `paths.yaml`. `KNEEMRI_DATA` overrides it, so a
runner with no dataset can point at an empty directory and the data-gated tests skip instead of erroring.

Only `raw/` and `processed/` are modelled here. Accessors for the competition's own layout (studies,
series tables, reports) belong beside the reader that walks them, and are added when that reader exists
rather than guessed from the file listing.
"""

import os
from pathlib import Path

import yaml

_ENV_OVERRIDE = "KNEEMRI_DATA"

COMPETITION = "rsna_knee_abnormality"
"""Directory name the competition download unzips into, under `raw/` and `processed/`."""


class DataRoot:
    """The data root. Datasets sit under `raw/` untouched; derived artefacts go to `processed/`."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @classmethod
    def from_config(cls, config: Path) -> "DataRoot":
        """Resolve the root: the `KNEEMRI_DATA` env var if set, else the `data:` entry of paths.yaml."""
        override = os.environ.get(_ENV_OVERRIDE)
        if override:
            return cls(Path(override))
        entries: dict[str, str] = yaml.safe_load(config.read_text(encoding="utf-8"))
        return cls(Path(entries["data"]))

    def raw(self, dataset: str = COMPETITION) -> Path:
        """Where an as-downloaded dataset lives. Never written to."""
        return self._root / "raw" / dataset

    def processed(self, dataset: str = COMPETITION) -> Path:
        """Where derived artefacts for a dataset go. Created on demand."""
        destination = self._root / "processed" / dataset
        destination.mkdir(parents=True, exist_ok=True)
        return destination
