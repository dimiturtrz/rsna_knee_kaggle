"""Machine-local data locations, adapted to the OS the process is actually running on.

One data root per machine, recorded in the gitignored `paths.yaml`. `KNEEMRI_DATA` overrides it, so a
runner with no dataset can point at an empty directory and the data-gated tests skip instead of erroring.

A KIND under that root (`raw`, `processed`, and any later one) resolves to `<data>/<kind>` unless
paths.yaml names it as a top-level key, which redirects that kind alone:

    data: D:/data/volumetric/mri
    processed: E:/scratch/processed     # the cache moves; raw/ stays on the slow disk

Accessors for the competition's own layout (studies, series tables, reports) belong beside the reader
that walks them, and are added when that reader exists rather than guessed from the file listing.

Every configured path — the root and any per-kind override — goes through `NativePath.resolve`, so a
path written on one OS either means something here or raises. See `core.native_path` for why.
"""

import os
from pathlib import Path

import yaml

from core.native_path import NativePath

ENV_OVERRIDE = "KNEEMRI_DATA"
"""Environment variable that replaces paths.yaml entirely. Named in the refusal `NativePath` raises."""

COMPETITION = "rsna_knee_abnormality"
"""Directory name the competition download unzips into, under `raw/` and `processed/`."""

RAW = "raw"
"""The as-downloaded datasets. Never written to."""

PROCESSED = "processed"
"""Derived DATA — decoded image caches, feature arrays, label tables, prediction caches. On demand.

Not checkpoints, and not a run's records: those are `MODELS`. A parsed study is still the corpus; a
tensor of weights fitted on it is a different kind of thing that merely happened to be written beside it.
"""

MODELS = "models"
"""Trained weights and the record of the run that produced them. NOT under the data root by default.

A model is not data: nothing trains on a checkpoint, and a corpus does not stop being a corpus when a
member is deleted. They shared `processed/` here for one reason only — 250 checkpoints happened to be
written beside the arrays they were fitted on — and the address `volumetric/mri/processed` is a claim
about MRI volumes that a 1.2 GB tensor of weights does not answer.

A run's `.metrics.jsonl` and `.holdout.npz` live HERE, beside their checkpoint, rather than in a
separate records root. They describe one model, they are named after it, and thirty-two call sites
derive them with `checkpoint.with_name(...)` — a split would break every one of those to relocate ten
megabytes.

Redirected in paths.yaml like any other kind, which is what keeps it off the data disk:

    data: D:/data/volumetric/mri
    models: D:/models
"""

REFERENCE = "reference"
"""Shared material that is neither a dataset nor ours — foreign weights, converted snapshots, papers.

Distinct from `raw/` because it is not a corpus to read and from `processed/` because nothing here
derives it: a converted third-party checkpoint is an INPUT that happens to have been reshaped once."""

CHECKPOINT_SUFFIX = ".pt"
"""What a member is stored as. Named here because the resolver below writes it and the submission layer
reads it, and a literal spelled in two layers is a literal that will one day disagree with itself."""


class DataRoot:
    """The data root. Datasets sit under `raw/` untouched; derived artefacts go to `processed/`."""

    def __init__(self, root: Path, kinds: dict[str, str] | None = None) -> None:
        self._root = root
        self._kinds = kinds or {}

    @classmethod
    def from_config(cls, config: Path) -> "DataRoot":
        """Resolve the root: the `KNEEMRI_DATA` env var if set, else the `data:` entry of paths.yaml.

        The rest of the file is kept rather than dropped: any top-level key is a per-kind override, and
        which kinds exist is a question for `kind`, not for the reader.

        The env override short-circuits the READ, not just the `data:` entry. A runner that sets it has
        no paths.yaml to open, and reading one first would turn the escape hatch into a second thing to
        provide.
        """
        override = os.environ.get(ENV_OVERRIDE)
        if override:
            return cls(Path(NativePath.resolve(override, ENV_OVERRIDE)))
        entries: dict[str, str] = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        return cls(Path(NativePath.resolve(str(entries["data"]), ENV_OVERRIDE)), entries)

    def kind(self, name: str) -> Path:
        """The root for one kind of data — `<data>/<name>`, or the path paths.yaml gave that name.

        An override is resolved exactly like the root is: a redirected kind on a foreign machine has the
        same silently-relative failure available to it, and no reason to be trusted more.
        """
        stated = self._kinds.get(name)
        return Path(NativePath.resolve(stated, ENV_OVERRIDE)) if stated else self._root / name

    def raw(self, dataset: str = COMPETITION) -> Path:
        """Where an as-downloaded dataset lives. Never written to."""
        return self.kind(RAW) / dataset

    def processed(self, dataset: str = COMPETITION) -> Path:
        """Where derived artefacts for a dataset go. Created on demand."""
        destination = self.kind(PROCESSED) / dataset
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    def models(self, dataset: str = COMPETITION) -> Path:
        """Where a dataset's trained members and their run records go. Created on demand."""
        destination = self.kind(MODELS) / dataset
        destination.mkdir(parents=True, exist_ok=True)
        return destination

    def checkpoints(self, names: list[str]) -> list[Path]:
        """Roster lines or bare paths, as checkpoint paths, tolerating a missing suffix and a comment.

        It lives on the root rather than on a roster reader because three layers now name members the
        same way — the submission layer packages them, the evaluation layer prices them, and neither may
        import the other. A resolver both of them speak is shared vocabulary, and the root is where the
        `processed/` half of the answer already lives.

        `comp/members.roster` writes `name.pt   # why this member exists`, so the comment is stripped
        here rather than by every caller — a roster read with its comments attached resolves to nothing
        and reports every member absent, which reads exactly like an empty roster.
        """
        paths = []
        for line in names:
            name = line.split("#")[0].strip()
            if not name:
                continue
            path = Path(name)
            paths.append(path if path.parent != Path() else self.models() / f"{path.stem}{CHECKPOINT_SUFFIX}")
        return paths
