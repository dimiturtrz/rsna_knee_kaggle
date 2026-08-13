"""Is the kit we are about to ship the code we are running? `python -m kneemri.kit_check`.

THE FOURTH CHECK, and like the other three it exists for a failure that already happened — twice, and
the second time after the fix was already committed.

2026-08-13: a submission ran on a kit built 2026-08-12 09:46 UTC while HEAD was 54 commits ahead. The
mounted code therefore lacked `9f31b35` (13:11 UTC that day), which bounds the forward pass by IMAGES
rather than studies — so whole-stack members sent four times the activation into a 16 GB T4, no
submission was written, and the fallback's constant table scored 0.500. The members were never measured
and the slot was gone. The same stale kit also predated `Member.precision`, which serves a bf16 member
in the arithmetic it was fitted under.

Members are versioned, kernels are versioned, and the code between them was not. `build_kit` now stamps
the commit into `kit.json` and `kernel_run` logs it, so the drift is VISIBLE. This makes it BLOCKING:
what matters is not that the kit differs from HEAD, but that it differs in the packages a kernel runs.
"""

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

PACKAGES = ("kneemri", "core")
"""What a kit ships. A commit that touches only tests or docs cannot change what a kernel computes."""


class KitCheck:
    """The built kit's commit against the working tree's, in the packages a kernel actually runs."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.git = shutil.which("git") or "git"

    def asked(self, *command: str) -> str:
        """One read-only git question, answered from this repo."""
        return subprocess.run(  # noqa: S603 - resolved executable, fixed argv, no shell, our own repo
            [self.git, "-C", str(self.repo), *command], capture_output=True, text=True, check=True
        ).stdout.strip()

    def stamped(self, kit: Path) -> str | None:
        """The commit a built kit records, or None when it predates the stamp.

        None rather than a guess: a kit whose version cannot be named is the state that spent the slot,
        so it is reported as unknown and refused rather than assumed current.
        """
        stamp = kit / "kneemri_src" / "kit.json"
        return str(json.loads(stamp.read_text(encoding="utf-8"))["commit"]) if stamp.exists() else None

    def drift(self, kit: Path) -> list[str]:
        """Commits touching the shipped packages that the kit does not have, newest first."""
        stamped = self.stamped(kit)
        if stamped is None:
            return ["the kit carries no commit stamp — rebuild it before shipping"]
        span = f"{stamped}..HEAD"
        log = self.asked("log", "--oneline", span, "--", *PACKAGES)
        return log.splitlines() if log else []


def main() -> None:
    """Refuse a kit that is behind the working tree in the packages a kernel runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, help="the built kit; defaults beside the other build artefacts")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from core.paths import DataRoot  # noqa: PLC0415 - CLI-only, and this module is imported as a library

    repo = Path(__file__).resolve().parents[1]
    kit = arguments.kit or DataRoot.from_config(arguments.paths).processed() / "kit"
    check = KitCheck(repo)
    behind = check.drift(kit)
    for line in behind:
        logger.info("MISSING FROM THE KIT: %s", line)
    logger.info("kit %s, %d commit(s) behind in %s", check.stamped(kit) or "UNSTAMPED", len(behind), "/".join(PACKAGES))
    if behind:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
