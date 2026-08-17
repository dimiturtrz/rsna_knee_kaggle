"""Run every accepted arm for one epoch on a few dozen studies: `arm_check --smoke <dir>`.

THE AXIS A FLAG CHECK CANNOT SEE. `ArmCheck` proves the parser accepts an arm and that its cache is on
disk; neither says the recipe those flags select survives contact with the card. `final_fit.sh` — the
deliverable, `--fold -1`, no score to notice a problem by — passed `arm_check` while asking for batch 12
of whole-stack 392 px inside `--vram-gb 22`, six gigabytes below what the member it ships recorded. It
died in the attention backward two minutes into a smoke, and would have died at the same step forty
minutes into the real run.
"""

import json
import logging
import math
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

Generous on purpose: the encoder is fetched on a cold HuggingFace cache and every arm now pays inductor
once (compilation became the default on 2026-08-17), and a smoke that timed out on its own setup would
report a healthy arm as broken. Measured across thirty-six arms: a compiled epoch on sixty studies takes
21-49 s against 5-9 s eager, nearly all of it that one graph build."""


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
        """The invocation that runs one arm's recipe as a smoke rather than as an arm.

        `--cache-studies` is what makes an UNTRIED reading smokeable: it names the truncated cache
        `arm_check` decodes beforehand, so an arm whose full corpus is two hours away still gets the
        one check that sees a recipe fail on the card. `--studies` alone would subsample a full cache
        that, for exactly those arms, does not exist.
        """
        overrides = [
            *("--cache-studies", str(self.studies)),
            *("--studies", str(self.studies)),
            *("--epochs", "1"),
            *("--out", str(out)),
        ]
        return [sys.executable, "-m", "kneemri.training.train", *shlex.split(flags), *overrides]

    def learned(self, checkpoint: Path) -> str:
        """What the arm's OWN metrics say about the epoch it ran — empty when there is nothing wrong.

        AN EXIT CODE IS NOT A RECIPE THAT TRAINS. Everything this module checked until now was whether
        the process died, and a recipe that diverges does not die: the scaler swallows a non-finite
        gradient, the epoch mean comes out NaN, the run writes its checkpoint and returns 0, and the
        sweep prints `ok`. That arm then takes a forty-minute slot to produce the same NaN.

        The per-label AUCs are NOT read, deliberately. At sixty studies a rare finding lands in the
        holdout with one class present and scores NaN by arithmetic rather than by fault — `Fracture`
        does exactly that in every smoke run here. Only the two aggregates the trainer computes over
        the whole epoch can distinguish a broken recipe from a small corpus.

        AND THE AUC IS ONLY READ WHERE THERE IS A HOLDOUT TO SCORE. `--fold -1` fits every study and
        keeps none back, so its mean AUC is NaN by design; on this check's first run it flagged
        `final_fit_e12` — the deliverable arm — as broken on exactly that. A check that cries about a
        recipe working as specified is one nobody reads by the third sweep. The LOSS is required
        finite everywhere, because no fold arithmetic can produce a NaN there.
        """
        records = checkpoint.with_name(checkpoint.stem + ".metrics.jsonl")
        if not records.exists():
            return "exited 0 but recorded no metrics at all"
        lines = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
        recipe: dict[str, str] = next((line["recipe"] for line in lines if "recipe" in line), {})
        epochs = [line for line in lines if "epoch" in line]
        if not epochs:
            return "exited 0 without finishing an epoch"
        last = epochs[-1]
        scored = ("loss", "auc") if str(recipe.get("fold", "0")) != "-1" else ("loss",)
        wrong = [key for key in scored if not math.isfinite(float(last[key]))]
        return f"epoch {last['epoch']} recorded a non-finite {' and '.join(wrong)}" if wrong else ""

    def failed(self, invocations: dict[str, str], out: Path) -> dict[str, str]:
        """The arms whose recipe does not survive one epoch, by arm name and the last thing it said.

        EVERY ARTIFACT THE ARM WROTE IS DELETED, by globbing the name rather than by naming suffixes.
        A smoke checkpoint is a few hundred megabytes and thirty-four of them is a disk problem this
        tool should not create — but the enumerated list has now failed twice, once per new sibling the
        trainer learned to write. It missed `.partial`, the periodic save the trainer drops beside the
        checkpoint whenever the holdout improves, and left 1.9 GB after seventeen arms in the very
        directory the line existed to keep empty; it then missed `.holdout.npz` and `.metrics.jsonl`
        the same way. A cleanup that must be told each filename is one that is wrong again the next
        time an output is added, so it is told the STEM instead — and only for files this run TOUCHED,
        because `--smoke` takes a directory from the caller and one obvious thing to hand it is
        `experiments/arms` itself, where `<name>.*` also matches `<name>.sh`.

        TOUCHED, not new. Skipping every path that existed beforehand protects the arm file, and it
        also protected the previous sweep's leftovers: a second smoke into the same directory REWRITES
        `<name>.partial` under the same name, finds it in the before-set, and keeps it. Measured here
        on 2026-08-17 — 3.5 GB of ninety-megabyte `.partial` files in a smoke directory the cleanup
        exists to leave empty. The modification time separates the two: an arm the smoke did not write
        keeps its stamp, and everything the trainer wrote this run does not.
        """
        broken: dict[str, str] = {}
        for name, flags in invocations.items():
            checkpoint = out / f"{name}.pt"
            untouched = {path: path.stat().st_mtime_ns for path in out.glob(f"{name}.*")}
            code, complaint = self.runner(self.command(flags, checkpoint))
            trouble = (complaint or f"exit {code} with nothing on stderr") if code else self.learned(checkpoint)
            if trouble:
                broken[name] = trouble
            # SAID AS IT IS LEARNED, not collected and reported at the end: thirty-four arms is the
            # better part of an hour, and a sweep that prints nothing until it finishes is one nobody
            # can tell from a hang.
            logger.info("%s %s%s", "BROKEN" if trouble else "ok", name, f": {trouble}" if trouble else "")
            for artifact in out.glob(f"{name}.*"):
                if untouched.get(artifact) != artifact.stat().st_mtime_ns:
                    artifact.unlink(missing_ok=True)
        return broken

    def reported(self, invocations: dict[str, str], refused: dict[str, str], out: Path) -> dict[str, str]:
        """Smoke the arms the parser accepted, logging each failure as it is found.

        A REFUSED arm is skipped rather than smoked: its flags never reach a trainer, so running it
        would report the parser's own complaint a second time under a name suggesting a memory failure.
        """
        out.mkdir(parents=True, exist_ok=True)
        runnable = {name: flags for name, flags in invocations.items() if name not in refused}
        logger.info("smoking %d arms at %d studies — minutes, and the card", len(runnable), self.studies)
        return self.failed(runnable, out)
