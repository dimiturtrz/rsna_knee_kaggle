"""Which runs actually FINISHED: `python -m kneemri.experiments.convergence_check`.

Nine checks in this package guard a SHIPMENT — that a kit builds, that a kernel's members mount, that a
queue is armed exactly once. None of them guards a NUMBER. On 2026-08-24 an audit of every metrics file
on disk found the cost of that asymmetry: three runs whose scores had been written into the ledger as
facts about model architectures were bounded by the harness rather than by the architecture.

- `raddino_dense` asked for 12 epochs and got **7** — `ARM FAILED raddino_dense (trainer exit 1)`. It
  died mid-epoch at 448 img/s with the loss still falling, no traceback and no OOM, killed by the
  environment during a card handoff. 0.8137 went into the ledger as RadDINO's score.
- `vjepa_member` asked for 12 and got **10** — exit 127, command-not-found, so its arm chain broke rather
  than its trainer. That is the third time in this repo a missing binary has read as a result.
- `cnx_dense` ran its full 12 and peaked on the LAST one with its loss still dropping. It was written up
  as "the first non-transformer trunk, weak". Refitting its learning rate returned **+0.0259** and a curve
  that peaks at e6 and then declines — which is what convergence looks like.

WHY TRUNCATED IS NOT MERELY SHORT, and why this needs several verdicts rather than a pass/fail. The recipes
set `one_cycle=True` over the full epoch count. A run that stops early stops in the HIGH-LR portion, so
the anneal — where one-cycle does its fine convergence — never runs at all. A truncated run is a different
experiment, not a shorter one.

The control that proves the distinction is on disk. `v3base_dense` peaks at e6 and then runs its remaining
five epochs and never beats that peak. `raddino_dense` also peaks at e6. Same peak epoch, opposite meaning:
one is a measurement and the other is a floor, and NOTHING IN THE SCORE TELLS THEM APART. Only whether the
run finished does — which is why this reads the recipe's epoch budget and not just the rows.

WHAT A TRUNCATION IS WORTH, which is why the line carries the DEPTH and not only the verdict. Over the 54
converged twelve-epoch runs on disk, the gain a kill at N rows would have missed is: N=7 median +0.0011 /
mean +0.0035 / max +0.0336; N=9 median +0.0000 / mean +0.0006; N=10 median +0.0000 / mean +0.0001. A late
kill costs nothing measurable and an early one can cost five noise floors, so "the score is a floor" is a
statement about the depth rather than about the verdict. The first write-up of this check quoted a single
size at every depth and had to be retracted; read the `best eN of M run` on the line before pricing one.

WHAT IT CANNOT SAY: whether a CONVERGED run's rate was well chosen. `cnx_own_recipe` and `cnx_dense` are
both twelve-epoch runs; only one of them converged, and this check is what separates those two cases, but
a run that converges early at a badly-chosen rate reads CONVERGED here and is still worth refitting.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from core import cli
from core.denominator import Denominator
from kneemri.runs.recorded_run import RecordedRun
from kneemri.runs.run_metrics import RunMetrics

logger = logging.getLogger(__name__)

EPOCHS = "epochs"
"""The recipe key holding the epoch budget the arm ASKED for — the half of the question rows cannot answer."""
STUDIES = "studies"
"""The recipe key a smoke caps. A smoke is short BY CONSTRUCTION and reporting it truncated would teach
the reader to skip this output, which is the failure the check exists to prevent."""
PATIENCE = "patience"
"""The recipe key holding the stopper, and the difference between a kill and a decision.

The first version of this check did not read it and reported TEN truncated runs. Three of those —
`res392_lr8e-6_ws`, `focal_pool`, `calibrated_targets` — carry `patience=3` and stopped at exactly
epoch 10 of 12, which is the stopper working. One of them is the `res392` resolution refutation, so
without this key the check's own first run would have reopened a correctly-closed family. The other
seven carry `patience=0`: no stopper existed, so stopping short is a kill and nothing else."""


class Convergence(StrEnum):
    """What a run's own record says about whether its number is finished.

    A StrEnum rather than a bool because the two failing cases have DIFFERENT fixes: a truncated run is
    re-run (or warm-started), while an undertrained one needs a longer budget or a fitted rate. Collapsing
    them to "not converged" would have hidden that `cnx_dense` wanted a rate and `raddino_dense` wanted
    only to be allowed to finish.
    """

    CONVERGED = "CONVERGED"
    """Ran the whole budget and peaked before the end. Its score is the recipe's number."""

    TRUNCATED = "TRUNCATED"
    """Died before the budget it asked for, with no stopper to explain it. The score is a FLOOR."""

    STOPPED = "STOPPED"
    """Ended short because `--patience` ended it. Its peak is a peak the stopper confirmed, so its score
    is the recipe's number — the same shape on disk as TRUNCATED, and the opposite meaning."""

    UNDERTRAINED = "UNDERTRAINED"
    """Ran the whole budget and was still gaining as the anneal drove the rate to zero.

    Read WITH its base rate, because the first reading of this verdict over the corpus was a
    misinterpretation. Under `one_cycle` a late peak is ordinary — 32% of finished twelve-epoch runs peak at
    e9 or e10, deep in the anneal — so "peaked late" says nothing. Peaking on the very LAST epoch is a
    different matter: 3 of 53 finished twelve-epoch runs do it, against the ~14% a flat prior over the
    observed e5-e11 range would give. It is rarer than chance, and all three carry a recipe defect confirmed
    from somewhere else — `cnx_dense` (+0.0259 once its rate was fitted) and the two `encoder_adapt` runs.
    n=3 makes that a direction, not a rule.
    """

    UNSCORED = "UNSCORED"
    """Held nothing back, so it has no epoch to peak on. Every `--fold -1` member is here by construction."""


@dataclass(frozen=True)
class RunConvergence:
    """One run's verdict, with the three numbers it was read from."""

    name: str
    verdict: Convergence
    best_epoch: int | None
    epochs_done: int
    epochs_asked: int | None

    def settled(self) -> bool:
        """Whether this run's score can be quoted as the recipe's own number."""
        return self.verdict in (Convergence.CONVERGED, Convergence.STOPPED, Convergence.UNSCORED)

    def line(self) -> str:
        """The finding as one line, carrying the evidence rather than only the verdict."""
        asked = self.epochs_asked if self.epochs_asked is not None else "?"
        best = self.best_epoch if self.best_epoch is not None else "-"
        return f"{self.verdict:13s} {self.name:34s} best e{best} of {self.epochs_done} run, {asked} asked"


class ConvergenceCheck:
    """Every run on disk, against whether its own record says the number is finished."""

    def smoke(self, run: RunMetrics) -> bool:
        """Whether the run capped the corpus — short by construction, so no budget verdict applies.

        Read from the RECIPE rather than the name, for the reason `ledger_check` learned the hard way:
        the smoke this project documents takes no `--out` and lands on the default `member_fold0.pt`,
        whose name carries nothing to match on.
        """
        recipe = RecordedRun.of(run).recipe
        return recipe is not None and recipe.get(STUDIES, "0") not in ("0", "")

    def benchmark(self, run: RunMetrics) -> bool:
        """Whether the run asked for one epoch, which makes every convergence verdict vacuous.

        A one-epoch run peaks on its last epoch BY CONSTRUCTION — there is no earlier epoch to peak on —
        so it reads UNDERTRAINED however healthy it was. Six of the fifteen the first corpus reading
        reported were `throughput_*` timing probes that never produced a member, and they diluted the
        three full-budget runs that are the actual finding.
        """
        asked = self.asked(run)
        return asked is not None and asked <= 1

    def asked(self, run: RunMetrics) -> int | None:
        """The epoch budget the recipe requested, or None for a run that recorded no recipe."""
        recipe = RecordedRun.of(run).recipe
        if recipe is None or EPOCHS not in recipe:
            return None
        return int(recipe[EPOCHS])

    def stoppable(self, run: RunMetrics) -> bool:
        """Whether a stopper was armed, which is the only innocent reason to end short.

        `patience=0` is the default and means no stopper, so a run that carries it and stops short was
        ended by something outside the trainer.
        """
        recipe = RecordedRun.of(run).recipe
        return recipe is not None and int(recipe.get(PATIENCE, 0)) > 0

    def verdict(self, run: RunMetrics) -> RunConvergence:
        """Read one run's record and say whether its number is finished.

        The order is deliberate. Ending short is decided FIRST, because a killed run also peaks on its
        last recorded epoch and would otherwise be filed as merely undertrained — the softer finding and
        the wrong fix.

        WHAT THIS CANNOT SEE, and it is on disk today. `RunMetrics.rows()` returns the LAST run's group,
        because a re-run to the same `--out` appends. `slices16_ws_s2026` holds a finished twelve-epoch
        run AND a killed re-run, and its checkpoint is the FINISHED run's — so this reports TRUNCATED for
        a member whose weights are fine. The verdict is about the last thing that ran, not always about
        the member on disk; where they differ, `RunMetrics.begin`'s rotation is the account that settles it.
        """
        rows = run.rows()
        best = run.best()
        peak = None if best is None else int(best["epoch"])
        done = len(rows)
        asked = self.asked(run)
        return RunConvergence(
            name=run.name(),
            verdict=self._judge(peak=peak, done=done, asked=asked, stoppable=self.stoppable(run)),
            best_epoch=peak,
            epochs_done=done,
            epochs_asked=asked,
        )

    def _judge(self, *, peak: int | None, done: int, asked: int | None, stoppable: bool) -> Convergence:
        """The rule, kept apart from the reading so each can be read on its own."""
        if asked is not None and done < asked:
            return Convergence.STOPPED if stoppable else Convergence.TRUNCATED
        if peak is None:
            return Convergence.UNSCORED
        if peak >= done - 1:
            return Convergence.UNDERTRAINED
        return Convergence.CONVERGED

    def audit(self, processed: Path) -> list[RunConvergence]:
        """Every scored run under the processed root, smokes excluded, worst verdict first."""
        found = [
            self.verdict(run)
            for metrics in sorted(processed.glob("*.metrics.jsonl"))
            if (run := RunMetrics(metrics)).rows() and not self.smoke(run) and not self.benchmark(run)
        ]
        return sorted(found, key=lambda seen: (seen.settled(), seen.name))


def main() -> None:
    """Report every run whose number is not finished, and exit non-zero if any is truncated."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--run", type=Path, help="one metrics file or checkpoint, instead of the whole root")
    arguments = command.parsed()

    check = ConvergenceCheck()
    if arguments.run is not None:
        seen = [check.verdict(RunMetrics(arguments.run))]
    else:
        processed = command.rooted(arguments).processed()
        seen = check.audit(processed)
        # The data root is gitignored and machine-local, so an unmounted drive reports "0 runs, all
        # settled" — the sentence a clean tree produces, from a check that read nothing at all.
        Denominator(len(seen), f"no scored run under {processed}").refuse_empty()

    for run in seen:
        if not run.settled():
            logger.info("%s", run.line())
    counted = {verdict: sum(run.verdict is verdict for run in seen) for verdict in Convergence}
    truncated = [run for run in seen if run.verdict is Convergence.TRUNCATED]
    logger.info(
        "%d runs: %s",
        len(seen),
        ", ".join(f"{count} {verdict.lower()}" for verdict, count in counted.items() if count),
    )
    if truncated:
        # Only TRUNCATED blocks. An undertrained run is an honest measurement of a budget and its score
        # is real; a truncated one is a crash whose score is a floor, and quoting it is the defect.
        logger.info("a truncated run's score is a FLOOR - do not quote it as the recipe's number")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
