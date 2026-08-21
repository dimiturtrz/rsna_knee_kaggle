"""What a Kaggle dataset actually holds, asked of Kaggle rather than of the directory it was built in."""

import json
import sys
from collections.abc import Callable
from pathlib import Path

from kneemri.spawn import Spawn

DATASET_METADATA = "dataset-metadata.json"
LIVE_TIMEOUT_SECONDS = 120


class LiveDataset:
    """What the members dataset holds ON KAGGLE, which is the only place a kernel can read it from.

    WHY THIS EXISTS, and it cost a kernel run on 2026-08-19. `MountCheck` reads the LOCAL package, so
    it answers "do the kernels name members this machine has packaged" — a question about a directory
    under the processed root. The kernel then ran on Kaggle and died with `WeightsError: ... holds
    [nine members] and was asked for ['band_dense_s2026']`, because the package had been rebuilt
    locally and never pushed. The check was green for the whole of that failure, and it was green
    CORRECTLY: nothing in it had ever addressed the live dataset.

    A stale local package is the harmless direction — a member absent locally cannot be named by a
    kernel that passed. The dangerous direction is the one that happened, where the local package leads
    the live one, and only Kaggle can answer it.

    The listing is INJECTED rather than fetched in `__init__` so the class is testable without a
    network: `hygiene.small` refuses a unit test that touches anything it did not create, and a check
    whose only test needs credentials is a check nobody runs under the gates.
    """

    def __init__(self, identifier: str, listing: Callable[[str], str] | None = None) -> None:
        self.identifier = identifier
        self.listing = listing or self._cli

    @staticmethod
    def entrypoint() -> str:
        """The kaggle CLI, BESIDE THE RUNNING INTERPRETER before anything on PATH.

        `kaggle` is declared in the `submit` extra, so it installs into `.venv/Scripts` and a bare
        `kaggle` resolves only when that directory happens to be on PATH — which it is not under
        `uv run` on Windows, where the first attempt died with `WinError 2` for a binary sitting next
        to the interpreter that raised it.
        """
        beside = Path(sys.executable).parent / "kaggle.exe"
        return str(beside) if beside.exists() else "kaggle"

    def _cli(self, identifier: str) -> str:
        """The CLI's own CSV listing, which is the read path that works without a bearer token."""
        # fixed argv, a dataset id from our own metadata, no shell
        done = Spawn().ran(
            [self.entrypoint(), "datasets", "files", identifier, "--csv"],
            capture_output=True,
            text=True,
            timeout=LIVE_TIMEOUT_SECONDS,
            check=True,
        )
        return done.stdout

    def members(self) -> set[str]:
        """Every member identifier the live dataset holds, from the `.pt` filenames it lists."""
        rows = self.listing(self.identifier).splitlines()
        names = (row.split(",", 1)[0].strip() for row in rows[1:] if row.strip())
        return {name.removesuffix(".pt") for name in names if name.endswith(".pt")}

    @staticmethod
    def identified(root: Path) -> str:
        """The dataset id a package ships under, read from the metadata that is pushed with it."""
        return str(json.loads((root / DATASET_METADATA).read_text(encoding="utf-8"))["id"])
