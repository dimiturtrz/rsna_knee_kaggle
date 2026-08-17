"""The stamp on the dataset Kaggle will MOUNT, as opposed to the kit sitting on this disk.

`kit_check` answered "is the kit I built current?", which is not the question a slot cares about.
Uploading is a manual `kaggle datasets version` step (`comp/build_kit.py`), so a kit can be built,
checked clean, and never pushed — after which the local check passes while Kaggle mounts something
older. On 2026-08-15 that was the live state: the dataset carried `4faac3d` while HEAD was five
shipped-package commits past it, among them the one that makes a member record the router it read.

Only the 46-byte stamp is fetched. It is listed as its own file in the dataset, so asking which code
Kaggle holds costs one small download rather than the weights.
"""

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET = "dimiturnt/kneemri-kit"
"""The dataset a kernel mounts, and the one `build_kit` writes its metadata for."""

STAMP = "kneemri_src/kit.json"
"""Where the stamp sits inside a kit, and the only path this needs to download."""

STAMP_TIMEOUT_SECONDS = 120
"""Generous for 46 bytes, and it is not the download it bounds: the uv fallback resolves an environment
first, which on a cold cache installs the kaggle CLI before a byte of the stamp moves."""


class MountedKit:
    """What Kaggle will run, fetched rather than assumed from the local build."""

    def __init__(self, dataset: str = DATASET) -> None:
        self.dataset = dataset
        self.kaggle = self._cli()

    @staticmethod
    def _cli() -> list[str]:
        """How to invoke the Kaggle CLI here — bare if installed, else through uv as `comp/push.sh` does.

        kaggle is NOT a project dependency: every real push runs `uv run --with kaggle kaggle …`, so
        `shutil.which("kaggle")` is None on this machine and the old `or "kaggle"` fallback raised
        `FileNotFoundError: [WinError 2]` — a check meant to say which code Kaggle holds instead saying
        nothing, right before a submission (2026-08-17).
        """
        installed = shutil.which("kaggle")
        if installed:
            return [installed]
        uv = shutil.which("uv")
        if not uv:
            message = "neither kaggle nor uv is on PATH, so the mounted kit cannot be asked about"
            raise RuntimeError(message)
        return [uv, "run", "--with", "kaggle", "kaggle"]

    def recorded(self) -> dict[str, object] | None:
        """Download the published stamp alone and read it; None when the dataset carries none.

        None rather than a guess, matching a local kit that predates the stamp: a version that cannot
        be named is the state that spent the slot, and it is refused rather than assumed current.

        BOUNDED, because this is the network. 46 bytes over a working connection is a second; without
        a timeout an unreachable Kaggle hangs the LAST check before a submission, and the reflex when a
        pre-submission check stops printing is to skip it. A timeout RAISES rather than returning None:
        None means "the dataset carries no stamp", and "we could not ask" is a different answer.
        """
        with tempfile.TemporaryDirectory() as into:
            subprocess.run(  # noqa: S603 - resolved executable, fixed argv, no shell
                [*self.kaggle, "datasets", "download", "-f", STAMP, self.dataset, "-p", into, "--force"],
                capture_output=True,
                text=True,
                check=True,
                timeout=STAMP_TIMEOUT_SECONDS,
            )
            stamp = Path(into) / Path(STAMP).name
            return dict(json.loads(stamp.read_text(encoding="utf-8"))) if stamp.exists() else None
