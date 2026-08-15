"""Prove every committed arm's flags before a queue costs GPU: `python -m kneemri.experiments.arm_check`.

This project's most expensive failures are arms that fail SILENTLY or late. Arm U printed `ARM U DONE`
after dying at epoch 7 of 24. A `str.replace` that matched nothing left `dinov3.sh` writing one name
for every read size, so a second probe began overwriting the run it was being compared against. A
package snapshot without a pyproject broke `uv run` and killed two queued arms in under a second. An
arm invoked with no arguments iterated zero times and exited 0 with an empty log.

A flag typo belongs to that family: it costs a queue slot to discover, and it discovers it after the
card has been held. So each arm is sourced against `experiments/stub_lib.sh`, whose `run_train` prints
its arguments instead of training, and the resulting line is handed to the REAL `TrainingOptions.parse`.

Sourcing rather than parsing the file, because the shell is what runs it: an arm that dies on a quoting
error is one this project has already lost two hours to, and only bash finds that.

The accepted flags are then asked a second question: is the cache that reading names actually decoded?
`main` takes the card BEFORE it maps the cache, so an undecoded reading costs the slot in the same way
a typo does — and it is a `FileNotFoundError` on a memory map rather than anything an arm anticipates.
"""

import argparse
import contextlib
import io
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

from core.paths import DataRoot
from kneemri.corpus.study_cache import StudyCache
from kneemri.training.train import TrainingOptions

logger = logging.getLogger(__name__)

ARMS = Path("experiments/arms")
STUB = Path("experiments/stub_lib.sh")
TIMEOUT_SECONDS = 30
"""An arm that has not printed its invocation by now is doing something a lint must not wait for."""

GIT_BASH = Path("C:/Program Files/Git/usr/bin/bash.exe")
"""The interpreter the arms are written for, named explicitly.

A bare `bash` resolves to WSL's on this machine, and WSL here is wedged often enough to have its own
note — so every arm timed out and this lint reported "0 arms checked, 0 refused". A clean pass over
nothing is the exact failure it was built to catch, produced by the tool itself on its first run.

The SECOND time it produced that same pass was subtler: run from PowerShell rather than Git Bash, the
inherited PATH carries no `/usr/bin`, so the stub lib's `dirname` and `mktemp` are not found and every
arm prints nothing. Hence `bin` below — the interpreter's own directory goes on the PATH it is handed,
so the shell is the same shell whatever launched python."""


class ArmCheck:
    """Every committed arm's train invocation, and whether the trainer would accept it."""

    def __init__(self, arms: Path = ARMS, stub: Path = STUB) -> None:
        self.arms = arms
        self.stub = stub

    def _environment(self) -> dict[str, str]:
        """The inherited environment, plus ARM_LIB, plus the shell's own coreutils on PATH."""
        path = os.environ.get("PATH", "")
        if GIT_BASH.exists():
            path = f"{GIT_BASH.parent}{os.pathsep}{path}"
        return os.environ | {"ARM_LIB": str(self.stub.resolve()), "PATH": path}

    def invocations(self) -> dict[str, str]:
        """Each arm's `run_train` flag line, by arm name. Arms that never call it are absent.

        Some arms drive `uv run` directly rather than through `run_train` — the diversity curve, the
        cache build — and have no flags to check. They time out or print nothing, and are skipped
        rather than reported: a lint that cries about files it cannot check gets ignored.
        """
        found: dict[str, str] = {}
        for arm in sorted(self.arms.glob("*.sh")):
            try:
                done = subprocess.run(  # noqa: S603 - fixed argv, our own files, no shell
                    [str(GIT_BASH) if GIT_BASH.exists() else "bash", str(arm)],
                    capture_output=True,
                    text=True,
                    timeout=TIMEOUT_SECONDS,
                    # The INHERITED environment plus ARM_LIB. Replacing it outright leaves Git Bash
                    # without the PATH it needs to find its own coreutils, and every arm then produces
                    # nothing - which this lint reports as "0 arms checked", a clean pass over nothing.
                    env=self._environment(),
                    check=False,
                )
            except subprocess.TimeoutExpired:
                continue
            for line in done.stdout.splitlines():
                name, tab, flags = line.partition("\t")
                if tab and flags.startswith("--"):
                    found[name] = flags
        return found

    def _parsed(self, flags: str) -> tuple[TrainingOptions | None, str]:
        """The options an arm's flags produce, or the parser's own complaint about them.

        One place that drives the real parser, because both halves of the check need it: whether the
        trainer accepts the arm, and what reading it accepted.
        """
        argv = sys.argv
        sys.argv = ["train", *shlex.split(flags)]
        # Bound BEFORE the try, because the parser exits by raising: bound inside the `with`, the
        # handler reads a name the checker cannot prove was ever assigned.
        complaint = io.StringIO()
        try:
            with contextlib.redirect_stderr(complaint):
                return TrainingOptions.parse(), ""
        except SystemExit as exit_code:
            if not exit_code.code:
                return None, ""
            lines = complaint.getvalue().strip().splitlines()
            return None, lines[-1] if lines else "rejected without a message"
        finally:
            sys.argv = argv

    def refused(self, invocations: dict[str, str]) -> dict[str, str]:
        """The arms whose flags `kneemri.training.train` would reject, with the parser's own complaint.

        A pure function of the invocations, so it is testable without a shell — which matters, because
        the unit gate refuses tests that shell out and this is the half that holds the logic.
        """
        complaints = ((name, self._parsed(flags)[1]) for name, flags in invocations.items())
        return {name: complaint for name, complaint in complaints if complaint}

    def uncached(self, invocations: dict[str, str]) -> dict[str, str]:
        """The arms whose reading names a cache that is not on disk, by arm name and cache key.

        THE OTHER LATE FAILURE, and the same price as a flag typo. `--image-size`, `--slices`,
        `--crop-mm`, `--scheme` and `--tie` together pick a cache file, and `main` resolves the CARD
        before it maps that file — so a reading nothing has been decoded for holds the GPU, raises on
        the memory map, and is discovered as a burned queue slot. Multiscale makes it live: the twelve-
        slot scheme is implemented and no twelve-slot cache has ever been built.

        Silent when the data root cannot be resolved, which is CI and any checkout without a
        `paths.yaml` — a lint that fails on the absence of gitignored data gets switched off.
        """
        uncached: dict[str, str] = {}
        for name, flags in invocations.items():
            options = self._parsed(flags)[0]
            if options is None or not options.paths.is_file():
                continue
            cache = StudyCache(
                DataRoot.from_config(options.paths).processed(),
                slices=options.slices,
                image_size=options.image_size,
                routing=options.routing,
                reading=options.reading,
            )
            if not (cache.images_path.is_file() and cache.mask_path.is_file() and cache.index_path.is_file()):
                uncached[name] = cache.key
        return uncached


def main() -> None:
    """Check every committed arm, and exit non-zero if any would be refused."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", type=Path, default=ARMS)
    parser.add_argument("--stub", type=Path, default=STUB)
    parser.add_argument(
        "--require-caches",
        action="store_true",
        help="also refuse an arm whose cache is not decoded yet. Off by default because a queue may "
        "legitimately build one in an earlier slot; on when arming a queue that builds nothing.",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    check = ArmCheck(arguments.arms, arguments.stub)
    invocations = check.invocations()
    refused = check.refused(invocations)
    for name, complaint in refused.items():
        logger.info("REFUSED %s: %s", name, complaint)
    uncached = check.uncached(invocations)
    for name, key in uncached.items():
        logger.info("UNCACHED %s: nothing decoded at %s", name, key)
    logger.info("%d arms checked, %d refused, %d uncached", len(invocations), len(refused), len(uncached))
    # A CLEAN PASS OVER NOTHING IS A FAILURE, and this tool has produced one twice — once from WSL's
    # bash, once from a PowerShell PATH with no coreutils. Both times it printed "0 arms checked, 0
    # refused" and exited 0, which is what a lint says when everything is fine.
    if not invocations:
        logger.info("REFUSED: no arm printed an invocation — the shell, not the arms, is what failed")
        sys.exit(1)
    if refused or (uncached and arguments.require_caches):
        sys.exit(1)


if __name__ == "__main__":
    main()
