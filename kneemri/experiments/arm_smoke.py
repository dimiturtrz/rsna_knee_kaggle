"""Run every accepted arm for one epoch on a few dozen studies: `arm_check --smoke <dir>`.

THE AXIS A FLAG CHECK CANNOT SEE. `ArmCheck` proves the parser accepts an arm and that its cache is on
disk; neither says the recipe those flags select survives contact with the card. `final_fit.sh` — the
deliverable, `--fold -1`, no score to notice a problem by — passed `arm_check` while asking for batch 12
of whole-stack 392 px inside `--vram-gb 22`, six gigabytes below what the member it ships recorded. It
died in the attention backward two minutes into a smoke, and would have died at the same step forty
minutes into the real run.
"""

import logging
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

SMOKE_STUDIES = 60
"""Enough corpus for both halves of the loop, few enough that an epoch costs seconds rather than minutes."""

SMOKE_TIMEOUT_SECONDS = 600
"""A smoke that has not finished one epoch on sixty studies in ten minutes has itself become the problem.

Generous on purpose: the encoder is fetched on a cold HuggingFace cache and `--compile-encoder` pays
inductor once, and a smoke that timed out on its own setup would report a healthy arm as broken."""


class ArmSmoke:
    """Every accepted arm actually RUN, and the ones the card refuses.

    The three appended flags win because argparse keeps the LAST occurrence, so an arm's own `--epochs`,
    `--studies` and `--out` are overridden without rewriting the line the arm printed.
    """

    def __init__(
        self,
        studies: int = SMOKE_STUDIES,
        runner: Callable[[list[str]], tuple[int, str]] | None = None,
    ) -> None:
        self.studies = studies
        self.runner = runner or self._subprocess

    @staticmethod
    def _subprocess(argv: list[str]) -> tuple[int, str]:
        """Run one smoke to completion and return its exit code with whatever it last complained about."""
        done = subprocess.run(  # noqa: S603 - fixed argv built from our own arm files, no shell
            argv, capture_output=True, text=True, timeout=SMOKE_TIMEOUT_SECONDS, check=False
        )
        lines = done.stderr.strip().splitlines()
        return done.returncode, lines[-1] if lines else ""

    def command(self, flags: str, out: Path) -> list[str]:
        """The invocation that runs one arm's recipe as a smoke rather than as an arm."""
        overrides = ["--studies", str(self.studies), "--epochs", "1", "--out", str(out)]
        return [sys.executable, "-m", "kneemri.training.train", *shlex.split(flags), *overrides]

    def failed(self, invocations: dict[str, str], out: Path) -> dict[str, str]:
        """The arms whose recipe does not survive one epoch, by arm name and the last thing it said."""
        broken: dict[str, str] = {}
        for name, flags in invocations.items():
            checkpoint = out / f"{name}.pt"
            code, complaint = self.runner(self.command(flags, checkpoint))
            if code:
                broken[name] = complaint or f"exit {code} with nothing on stderr"
            # DELETED as soon as it has proved the write path, because a smoke checkpoint is a few
            # hundred megabytes and thirty-four of them is a disk problem this tool should not create.
            checkpoint.unlink(missing_ok=True)
        return broken

    def reported(self, invocations: dict[str, str], refused: dict[str, str], out: Path) -> dict[str, str]:
        """Smoke the arms the parser accepted, logging each failure as it is found.

        A REFUSED arm is skipped rather than smoked: its flags never reach a trainer, so running it
        would report the parser's own complaint a second time under a name suggesting a memory failure.
        """
        out.mkdir(parents=True, exist_ok=True)
        runnable = {name: flags for name, flags in invocations.items() if name not in refused}
        logger.info("smoking %d arms at %d studies — minutes, and the card", len(runnable), self.studies)
        broken = self.failed(runnable, out)
        for name, complaint in broken.items():
            logger.info("BROKEN %s: %s", name, complaint)
        return broken
