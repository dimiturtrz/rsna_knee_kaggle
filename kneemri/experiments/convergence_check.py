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
  that peaks at e6 and then declines — which is what convergence looks like. The delta is the point here,
  not the destination: `cnx_own_recipe` landed at 0.8186, seat 53 of 67, and did not move the ensemble
  peak. A rate refit is the largest single delta this repo owns and it did not produce a usable member.

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
from kneemri.runs.run_record import EpochRow, RunRecord

logger = logging.getLogger(__name__)

EPOCHS = "epochs"
STOP_AFTER_EPOCH = "stop_after_epoch"
"""The recipe key holding the epoch budget the arm ASKED for — the half of the question rows cannot answer."""
STUDIES = "studies"
"""The recipe key a smoke caps. A smoke is short BY CONSTRUCTION and reporting it truncated would teach
the reader to skip this output, which is the failure the check exists to prevent."""
AUC = "auc"
"""The row key holding an epoch's holdout score — the series a budget verdict is a statement about."""
TAIL = 3
"""How many epochs back the final gain is measured over.

One step is a single noisy difference and cannot separate a plateau from a wobble; three is the shortest
window over which `one_cycle`'s final descent is a trend rather than a pair of points. It is also the
window every four-epoch run on disk can still answer."""
RISING_MARGIN = 0.0062
"""The gain over `TAIL` epochs below which a run was not still gaining, only ending high.

The project's paired holdout floor, legislated elsewhere and reused here with the same argument: it is the
smallest difference this 909-study holdout resolves, so a tail gain under it is not a measured gain at all."""
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

    AND PEAKING LAST IS NOT ENOUGH, since 2026-08-30. The argmax alone reported eleven runs, seven of them
    plateaus that had gained 0.0004-0.0020 over their final three epochs. `rising` now requires the tail gain
    to clear the holdout floor, so this verdict means the score was still MOVING when the budget ended.
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

    def smoke(self, run: RunRecord) -> bool:
        """Whether the run capped the corpus — short by construction, so no budget verdict applies.

        Read from the RECIPE rather than the name, for the reason `ledger_check` learned the hard way:
        the smoke this project documents takes no `--out` and lands on the default `member_fold0.pt`,
        whose name carries nothing to match on.
        """
        recipe = RecordedRun.of(run).recipe
        return recipe is not None and recipe.get(STUDIES, "0") not in ("0", "")

    def benchmark(self, run: RunRecord) -> bool:
        """Whether the run asked for one epoch, which makes every convergence verdict vacuous.

        A one-epoch run peaks on its last epoch BY CONSTRUCTION — there is no earlier epoch to peak on —
        so it reads UNDERTRAINED however healthy it was. Six of the fifteen the first corpus reading
        reported were `throughput_*` timing probes that never produced a member, and they diluted the
        three full-budget runs that are the actual finding.
        """
        asked = self.asked(run)
        return asked is not None and asked <= 1

    def asked(self, run: RunRecord) -> int | None:
        """The epochs the recipe's loop was asked to run, or None for a run that recorded no recipe.

        `stop_after_epoch` ends the loop early while keeping the full anneal, so a run that stops there
        finished what it was asked for — the budget `Fitting` itself runs (knee-o52mm).
        """
        recipe = RecordedRun.of(run).recipe
        if recipe is None or EPOCHS not in recipe:
            return None
        epochs = int(recipe[EPOCHS])
        stop = int(recipe.get(STOP_AFTER_EPOCH, 0))
        return min(epochs, stop) if stop else epochs

    def stoppable(self, run: RunRecord) -> bool:
        """Whether a stopper was armed, which is the only innocent reason to end short.

        `patience=0` is the default and means no stopper, so a run that carries it and stops short was
        ended by something outside the trainer.
        """
        recipe = RecordedRun.of(run).recipe
        return recipe is not None and int(recipe.get(PATIENCE, 0)) > 0

    def rising(self, run: RunRecord) -> bool:
        """Whether the run was still GAINING at the end, rather than merely peaking there.

        Peaking on the last epoch is an ARGMAX; being bound by a budget is a SLOPE. Seven of the eleven runs
        the argmax rule called UNDERTRAINED had gained 0.0004-0.0020 across their last three epochs —
        `mixup_dense` 0.8548, 0.8541, 0.8546, 0.8552 is the shape — which is a plateau whose highest point
        happened to be the final one. Reading those as budget-bound argues for a longer schedule, and a
        longer schedule is the one fix this repo has already measured as HARMFUL: an oversized budget
        stretches `one_cycle`'s anneal and leaves the rate far from annealed where the model converges.

        A run with no more epochs than the window cannot be asked, and stays undertrained: `lora_r8_small_336`
        and `adapt_null` ran four and gained 0.078, which is what a budget binding actually looks like.
        """
        scored = [row[AUC] for row in run.rows() if row.get(AUC) is not None]
        if len(scored) <= TAIL:
            return True
        return scored[-1] - scored[-1 - TAIL] > RISING_MARGIN

    def reached(self, rows: list[EpochRow]) -> int:
        """How many epochs the run got through, which is NOT how many rows the live file holds.

        A `--resume` round is not a re-run: `RunMetrics.begin` rotates the earlier round's epochs aside
        and the new round's `epoch` keeps counting from where the last one stopped, so `dml_a_s2026`'s
        completed 24 epochs sit on disk as a final group of EIGHT rows numbered 16 to 23. Counting rows
        reported it as `best e16 of 1 run, 24 asked` — TRUNCATED, whose stated meaning is that the
        anneal never ran and the score is a FLOOR. That is the precise opposite of the truth for a run
        that finished its budget, and it was about to be recorded against a verdict that closes a family.

        Read from the last row's own index rather than by reading the rotated files, because the rotated
        checkpoints are a different member: `begin` moves the weights and the predictions aside with the
        epochs, so a peak unioned across rounds could name an epoch whose `.pt` is no longer the one this
        run's name resolves to. The index answers the budget question on its own and leaves that
        provenance rule intact. Identical to the row count for every run that starts at epoch 0, which is
        every run that was not resumed.
        """
        return rows[-1]["epoch"] + 1 if rows else 0

    def verdict(self, run: RunRecord) -> RunConvergence:
        """Read one run's record and say whether its number is finished.

        The order is deliberate. Ending short is decided FIRST, because a killed run also peaks on its
        last recorded epoch and would otherwise be filed as merely undertrained — the softer finding and
        the wrong fix.

        WHAT THIS CANNOT SEE, and it is on disk today. `RunRecord.rows()` returns the LAST run's group,
        because a re-run to the same `--out` appends. `slices16_ws_s2026` holds a finished twelve-epoch
        run AND a killed re-run, and its checkpoint is the FINISHED run's — so this reports TRUNCATED for
        a member whose weights are fine. The verdict is about the last thing that ran, not always about
        the member on disk; where they differ, `RunMetrics.begin`'s rotation is the account that settles it.
        """
        rows = run.rows()
        best = run.best()
        peak = None if best is None else int(best["epoch"])
        done = self.reached(rows)
        asked = self.asked(run)
        return RunConvergence(
            name=run.name(),
            verdict=self._judge(
                peak=peak, done=done, asked=asked, stoppable=self.stoppable(run), rising=self.rising(run)
            ),
            best_epoch=peak,
            epochs_done=done,
            epochs_asked=asked,
        )

    def _judge(self, *, peak: int | None, done: int, asked: int | None, stoppable: bool, rising: bool) -> Convergence:
        """The rule, kept apart from the reading so each can be read on its own."""
        if asked is not None and done < asked:
            return Convergence.STOPPED if stoppable else Convergence.TRUNCATED
        if peak is None:
            return Convergence.UNSCORED
        if peak >= done - 1 and rising:
            return Convergence.UNDERTRAINED
        return Convergence.CONVERGED

    def audit(self, models: Path) -> list[RunConvergence]:
        """Every scored run under the models root, smokes excluded, worst verdict first."""
        found = [
            self.verdict(run)
            for metrics in sorted(models.glob("*.metrics.jsonl"))
            if (run := RunRecord(metrics)).rows() and not self.smoke(run) and not self.benchmark(run)
        ]
        return sorted(found, key=lambda seen: (seen.settled(), seen.name))

    def single(self, path: Path) -> list[RunConvergence]:
        """The one run at `path`, refusing a path that holds no file.

        A PATH, and a name is not one. `--run swin_supervised_s2026` resolves against the working
        directory, finds nothing, and reports UNSCORED — a verdict that describes a real state and
        hides a typo, which is how a converged run was read as having no score at all. The whole point
        of this check is that a number is not quoted from a run that cannot support it, so a reading
        taken of no file is the one thing it may never return quietly.
        """
        record = RunRecord(path)
        if not record.exists():
            message = f"{record.path} is not on disk - --run takes a PATH to a metrics file, not a run name"
            raise SystemExit(message)
        return [self.verdict(record)]

    @staticmethod
    def summary(seen: list[RunConvergence]) -> str:
        """How many runs were read, and how many landed on each verdict that occurred."""
        counted = {verdict: sum(run.verdict is verdict for run in seen) for verdict in Convergence}
        return f"{len(seen)} runs: " + ", ".join(
            f"{count} {verdict.lower()}" for verdict, count in counted.items() if count
        )


def main() -> None:
    """Report every run whose number is not finished, and exit non-zero if any is truncated."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--run", type=Path, help="one metrics file or checkpoint, instead of the whole root")
    arguments = command.parsed()

    check = ConvergenceCheck()
    if arguments.run is not None:
        seen = check.single(arguments.run)
    else:
        models = command.rooted(arguments).models()
        seen = check.audit(models)
        # The data root is gitignored and machine-local, so an unmounted drive reports "0 runs, all
        # settled" — the sentence a clean tree produces, from a check that read nothing at all.
        Denominator(len(seen), f"no scored run under {models}").refuse_empty()

    for run in seen:
        if not run.settled():
            logger.info("%s", run.line())
    truncated = [run for run in seen if run.verdict is Convergence.TRUNCATED]
    logger.info("%s", check.summary(seen))
    if truncated:
        # Only TRUNCATED blocks. An undertrained run is an honest measurement of a budget and its score
        # is real; a truncated one is a crash whose score is a floor, and quoting it is the defect.
        logger.info("a truncated run's score is a FLOOR - do not quote it as the recipe's number")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
