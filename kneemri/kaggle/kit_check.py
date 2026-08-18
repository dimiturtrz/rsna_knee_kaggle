"""Is the kit we are about to ship the code we are running? `python -m kneemri.kaggle.kit_check`.

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
what matters is not that the kit differs from HEAD, but that it differs in the files a kernel LOADS —
the entrypoint's import closure (`kernel_closure`), which is a fraction of the packages a kit ships.

`--mounted` ASKS THE ONLY QUESTION A SLOT CARES ABOUT, and `MountedKit` fetches the answer. The kernel's
log names the mounted commit, but only after the slot is spent — the same too-late that cost the first two.
"""

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

from kneemri.kaggle.kernel_closure import KernelClosure
from kneemri.kaggle.mounted_kit import DATASET, MountedKit

logger = logging.getLogger(__name__)


class KitCheck:
    """The built kit's commit against the working tree's, in the files a kernel actually loads."""

    def __init__(self, inside: Path) -> None:
        """Anchored at the repository ROOT, whatever path inside it is handed over.

        The pathspec is what makes this a root and not a directory: `git -C <dir> log -- kneemri core`
        resolves those names against <dir>, so anchoring one level down asked about `kneemri/kneemri`
        and `kneemri/core`, matched nothing, and reported EVERY kit as current. The mounted check
        passed that way for 53 shipped-package commits, including the restructure that moved this
        module down a level and caused it. Asking git where the root is cannot drift when a file moves.
        """
        self.git = shutil.which("git") or "git"
        self.repo = inside
        self.repo = Path(self.asked("rev-parse", "--show-toplevel"))
        self.loaded = KernelClosure(self.repo).files()

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
        recorded = self.recorded(kit)
        return str(recorded["commit"]) if recorded else None

    def recorded(self, kit: Path) -> dict[str, object] | None:
        """The whole stamp — the commit AND whether the tree was dirty when it was built."""
        stamp = kit / "kneemri_src" / "kit.json"
        return dict(json.loads(stamp.read_text(encoding="utf-8"))) if stamp.exists() else None

    def drift(self, kit: Path) -> list[str]:
        """Commits touching the files a kernel loads that the built kit does not have, newest first."""
        return self.behind(self.recorded(kit))

    def behind(self, recorded: dict[str, object] | None) -> list[str]:
        """The same judgement for any stamp, wherever it was read from — a build directory or Kaggle.

        The pathspec is the ENTRYPOINT'S IMPORT CLOSURE, not the shipped packages: a kit ships every
        module under `kneemri` and `core`, but a kernel imports a fraction of them, and a commit to a
        measurement module cannot change what the kernel computes any more than a commit to a test can.
        """
        if recorded is None:
            return ["the kit carries no commit stamp — rebuild it before shipping"]
        behind = self.asked("log", "--oneline", f"{recorded['commit']}..HEAD", "--", *self.loaded).splitlines()
        # A DIRTY KIT NAMES A COMMIT IT IS NOT. On 2026-08-14 one was uploaded and run from a tree with
        # uncommitted changes, so its stamp pointed at code that was never what shipped — which is the
        # same unnameable-version state the stamp exists to end, wearing a commit hash.
        if recorded.get("dirty"):
            behind.append(f"built from a DIRTY tree at {recorded['commit']} — the stamp names code that did not ship")
        return behind


def main() -> None:
    """Refuse a kit that is behind the working tree in the packages a kernel runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, help="the built kit; defaults beside the other build artefacts")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    parser.add_argument("--mounted", action="store_true", help="ask Kaggle what it will mount, not the local build")
    parser.add_argument("--dataset", default=DATASET)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from core.paths import DataRoot  # noqa: PLC0415 - CLI-only, and this module is imported as a library

    repo = Path(__file__).resolve().parent
    check = KitCheck(repo)
    if arguments.mounted:
        where, recorded = arguments.dataset, MountedKit(arguments.dataset).recorded()
    else:
        kit = arguments.kit or DataRoot.from_config(arguments.paths).processed() / "kit"
        where, recorded = str(kit), check.recorded(kit)
    behind = check.behind(recorded)
    for line in behind:
        logger.info("MISSING FROM THE KIT: %s", line)
    stamp = str(recorded["commit"]) if recorded else "UNSTAMPED"
    logger.info(
        "%s at %s, %d commit(s) behind in the %d files the kernel loads", where, stamp, len(behind), len(check.loaded)
    )
    if behind:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
