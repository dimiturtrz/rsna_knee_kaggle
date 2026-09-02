"""Can a running arm still reach its branch: `python -m kneemri.experiments.trajectory_check`.

`convergence_check` guards a number AFTER a run ends. This guards the CARD while it is still running, and
it is the only check here that can end a job. A twelve-epoch `one_cycle` arm costs about three hours; the
question it exists to answer is whether the remaining two of them can change which branch the arm lands in.

WHAT LICENSES IT. Across every run under the models root that recorded at least `FLOOR_SHARE` of its own
budget and asked for the same epoch BUDGET as the arm under test, the gain from the gate epoch to the peak is a measured
distribution rather than a guess. On 2026-08-30, matched to a twelve-epoch schedule, it read median +0.0208,
p90 +0.0368, max +0.0529 over n=102 — the maximum being `encoder_adapt_gentle`, a run that started very low
and so had the most room to climb. So an arm sitting at the gate more than the MAXIMUM
such gain below its own SEAT threshold cannot reach that threshold by any trajectory this project has ever
recorded, and the epochs left decide nothing.

FIRST USE, and the reason the numbers above are quoted from a real reading rather than from a design: on
2026-08-30 `vit_in21k` sat at 0.7847 at epoch 4 against a pre-registered seat branch of 0.8393. It needed
+0.0546 — larger than the largest gain in the distribution, and that largest gain was itself measured from
an EARLIER epoch of a longer run, so the comparison was generous in the arm's favour twice over. It was
truncated and the card went to `swin_wholetop`. The second anchor is a matched one: `swin_supervised` sat
at 0.8192 at epoch 3 and was still declined as too weak, so the band a seat-worthy arm occupies at the gate
is known from the other side too.

WHAT IT MAY NOT DO, and these are the failure modes it would otherwise invite.

- It may only TRUNCATE. It must never promote an arm, never rank two arms, and never close a family: a high
  value at the gate says the run is not yet refuted, which is not the same as a result.
- It may only truncate against a branch that was PRE-REGISTERED before the run started. Reading a target
  off the curve is fitting the gate to the arm, and then every arm passes.
- A truncated `one_cycle` score is a FLOOR — the anneal never ran — so the record must carry it as one.
  `convergence_check` reports exactly that verdict, and the ledger row has to say it in prose too.
- The distribution is over THIS harness's runs, most of them DINOv2 at twelve epochs. An arm on a trunk
  with a genuinely different shape is outside it, and the honest move there is to widen the corpus rather
  than to trust the ceiling. A different SCHEDULE is handled — the corpus is matched on the epoch budget —
  but that only narrows the sample, so a budget with two runs on it is not an instrument. `MIN_CORPUS`
  now enforces that sentence rather than only stating it.

AND THE EPOCH FLOOR WAS BLIND TO SHORT BUDGETS, found 2026-08-30 on `cnx_warm_anneal` (knee-e8sm). The floor
was an absolute ten recorded epochs, which no six-epoch run can ever meet, so every budget under ten was
structurally unreadable no matter how large its corpus grew. That constant's own docstring said it was there
to "keep the corpus at the twelve-epoch schedule" — a job the explicit budget match already does, so it was
selecting the schedule twice and excluding every other one on the second pass. It matters because the warm
lane is now a commitment: `warm_calib_lowlr` measured warm starting as a TIME mechanism, which makes the arms
it recommends SHORT, and the only check that can end a bad job was blind to exactly them. `FLOOR_SHARE`
replaces it and is calibrated to change nothing at twelve — the twelve-epoch corpus is 102 runs under both
rules on the same tree, verified rather than argued, and the n=96 this file used to quote was the same
corpus six runs earlier that morning.

The fix does NOT make `cnx_warm_anneal` readable, and that is the honest outcome: its budget now MATCHES,
and the answer is that one run on this tree has ever asked for six epochs. The refusal changed from a
misleading "no run reached epoch 3 with 10 recorded" — which described a floor no six-epoch run could ever
clear — to "only 1 run matches a 6-epoch budget, and 8 are needed", which names the thing to go build.

AND THE CORPUS ITSELF NEEDED A GUARD, which is the first thing this tool found that the hand reading missed.
Before runs recording more epochs than they asked for were refused, the twelve-epoch maximum read +0.1350
rather than +0.0529: `lora_r8_v3small_224` holds 16 rows against a 12-epoch recipe, so its gate value and its
peak belong to different runs. One stitched series, and the ceiling is two and a half times the largest real
gain — which for a gate decided by the MAXIMUM means it can barely fire at all.

WHY THE CEILING IS THE MAXIMUM AND NOT THE p90. A gate that fires at p90 is wrong one time in ten, and the
thing it costs when it is wrong is a member — the asymmetry runs entirely one way, since the thing it saves
when it is right is two hours of a card that is not otherwise scarce today. Both numbers are printed, and
only the maximum decides.
"""

import logging
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from core import cli
from core.denominator import Denominator
from kneemri.experiments.convergence_check import AUC, ConvergenceCheck
from kneemri.runs.run_metrics import RunMetrics

logger = logging.getLogger(__name__)

GATE_EPOCH = 3
"""The epoch the distribution is measured FROM, zero-indexed.

Early enough to save most of a run and late enough to be past the noise of the first epochs: over the
reference curve (`swin_supervised`) epochs 0-2 climb 0.7441 -> 0.8110 and are dominated by the head
learning to read a frozen trunk at all, while the ordering between arms only settles once the backbone has
moved. A gate at e1 would refute arms that go on to win."""

FLOOR_SHARE = 10 / 12
"""What share of its OWN budget a run must have recorded to enter the distribution.

A run that died early has not shown where its peak is, so its gate-to-peak gain understates a completed
run's — and this ceiling is only sound if it is generous. What makes a run short is therefore how much of
its own schedule it missed, not an epoch count: ten of twelve and five of six have each seen the same share
of their anneal, while an absolute floor of ten calls the second one incomplete and the corpus it belongs to
empty.

Ten twelfths rather than a rounder number because it is exactly the previous absolute floor at the budget
the gate was calibrated on — `ceil(FLOOR_SHARE * 12) == 10` — so the twelve-epoch reading this tool's
numbers are quoted from is unchanged by the fix: both floors admit exactly 102 runs on the same tree. A
threshold that moves the existing corpus is a different instrument wearing the same name."""

MIN_CORPUS = 8
"""How many runs a budget needs before its spread may decide anything.

The module docstring has said since it was written that "a budget with two runs on it is not an instrument";
this is that sentence as a refusal. The ceiling is the MAXIMUM of the distribution, and a maximum over three
draws is a statement about which three runs happened to use that schedule, not about what a trajectory on it
can do — and being wrong here kills a member. Eight is the point at which the maximum stops moving on every
addition; it is a legislated bound, not a count read off the current tree."""

HIGH = 0.9
"""The reported quantile. Printed for the reader, never used to decide — see the module docstring."""


@dataclass(frozen=True)
class Spread:
    """The measured gate-to-peak gains, as the three numbers a ceiling is read from."""

    runs: int
    median: float
    high: float
    largest: float

    def line(self) -> str:
        """The distribution as one line, so a truncation decision can be quoted with its evidence."""
        return (
            f"gain from e{GATE_EPOCH} to peak over {self.runs} runs: "
            f"median +{self.median:.4f}, p{int(HIGH * 100)} +{self.high:.4f}, max +{self.largest:.4f}"
        )


@dataclass(frozen=True)
class RunTrajectory:
    """One arm read against a target it was given before it started."""

    name: str
    epoch: int
    auc: float
    target: float
    ceiling: float

    def truncatable(self) -> bool:
        """Whether no trajectory in the corpus takes this arm from where it is to where it must be."""
        return self.ceiling < self.target

    def line(self) -> str:
        """The finding with the arithmetic in it, rather than only the verdict."""
        verdict = "CANNOT REACH" if self.truncatable() else "still open"
        return (
            f"{verdict:13s} {self.name:34s} e{self.epoch} {self.auc:.4f}, "
            f"needs +{self.target - self.auc:.4f} for {self.target:.4f}, generous ceiling {self.ceiling:.4f}"
        )


class TrajectoryGate:
    """The gate-to-peak distribution, and one arm read against it."""

    def __init__(self) -> None:
        self._runs = ConvergenceCheck()

    def scored(self, run: RunMetrics) -> list[float]:
        """The run's holdout series, NaN rows dropped.

        A `--fold -1` member holds nothing back and records NaN rather than omitting the key, so a filter
        on `is not None` keeps them — and one NaN in the list makes the median and the maximum NaN too.
        The first run of this check reported `median +nan, max +nan` over 119 runs for exactly that."""
        return [value for row in run.rows() if isinstance(value := row.get(AUC), float) and math.isfinite(value)]

    def gain(self, run: RunMetrics, budget: int | None) -> float | None:
        """This run's gain from the gate epoch to its peak, or None if it does not belong in the corpus.

        `budget` matches on the epoch count the recipe ASKED for, because the gain is a statement about a
        SCHEDULE and not about a model: a 24-epoch `one_cycle` run has twice the anneal left after the gate
        and reads median +0.0339 against a 12-epoch run's +0.0210. Comparing across the two would let a
        long schedule's headroom excuse a short arm.

        A run recording MORE epochs than it asked for is refused outright, and it stays refused now that
        `RunMetrics.rows()` splits on a restarting epoch as well as on a recipe line (knee-v5lb). That fix
        is what makes `lora_r8_v3small_224` readable at all — 16 rows against a 12-epoch recipe, its e3 in
        one run and its peak in another, contributing +0.1350 as one series against a corpus whose largest
        real gain is +0.0529, and the maximum is what decides here. This guard caught it BEFORE the reader
        was fixed, so it is kept: a series stitched by some future shape the reader does not know about
        would arrive the same way, and a gate that can barely fire fails silently. See knee-gnzx.
        """
        asked = self._runs.asked(run)
        if asked is None or (budget is not None and asked != budget):
            return None
        scored = self.scored(run)
        if len(scored) > asked or len(scored) < math.ceil(FLOOR_SHARE * asked) or len(scored) <= GATE_EPOCH:
            return None
        return max(scored) - scored[GATE_EPOCH]

    def spread(self, models: Path, budget: int | None = None) -> Spread:
        """The distribution over every non-smoke run under the root that ran long enough to have one."""
        gains = [
            gain
            for metrics in sorted(models.glob("*.metrics.jsonl"))
            if (run := RunMetrics(metrics)).rows() and not self._runs.smoke(run) and not self._runs.benchmark(run)
            if (gain := self.gain(run, budget)) is not None
        ]
        # An unmounted data root, or a budget too thinly used to have a distribution, would otherwise make
        # every arm truncatable against a corpus that cannot answer — the one direction this gate must
        # never fail in. A thin corpus is reported with its real count, because "3 runs" and "none at all"
        # send the reader to different fixes.
        schedule = "any schedule" if budget is None else f"a {budget}-epoch budget"
        Denominator(
            len(gains) if len(gains) >= MIN_CORPUS else 0,
            f"only {len(gains)} run(s) under {models} match {schedule} past epoch {GATE_EPOCH}, "
            f"and {MIN_CORPUS} are needed before a spread may decide anything",
        ).refuse_empty()
        ranked = sorted(gains)
        return Spread(
            runs=len(ranked),
            median=statistics.median(ranked),
            high=ranked[min(int(HIGH * len(ranked)), len(ranked) - 1)],
            largest=ranked[-1],
        )

    def reading(self, run: RunMetrics, target: float, spread: Spread) -> RunTrajectory:
        """Where the arm stands now, and the best it could still do by the corpus's own maximum gain.

        Read from the LAST recorded epoch rather than from the gate, which is generous in the arm's favour:
        the corpus's gains are measured from the gate, so an arm already past it is credited with a climb
        it has partly spent. That is deliberate — the gate may only truncate, so every approximation in it
        leans towards letting the run continue.
        """
        scored = self.scored(run)
        return RunTrajectory(
            name=run.name(),
            epoch=len(scored) - 1,
            auc=scored[-1],
            target=target,
            ceiling=scored[-1] + spread.largest,
        )


def main() -> None:
    """Report the gate distribution, and whether one named arm can still reach a pre-registered target."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--run", type=Path, help="a metrics file or checkpoint for the arm under test")
    command.parser.add_argument(
        "--target", type=float, help="the PRE-REGISTERED score the arm must reach; read off no curve"
    )
    arguments = command.parsed()

    gate = TrajectoryGate()
    models = command.rooted(arguments).models()
    if arguments.run is None:
        logger.info("%s", gate.spread(models).line())
        return

    record = RunMetrics(arguments.run)
    if not record.exists():
        message = f"{record.path} is not on disk - --run takes a PATH to a metrics file, not a run name"
        raise SystemExit(message)
    if arguments.target is None:
        message = "--run needs --target: this gate compares against a branch registered BEFORE the run started"
        raise SystemExit(message)
    # MATCHED on the arm's own epoch budget, so the ceiling it is measured against is one a run on this
    # schedule actually achieved. Falls back to the whole corpus only for an arm that recorded no recipe.
    spread = gate.spread(models, ConvergenceCheck().asked(record))
    logger.info("%s", spread.line())
    scored = gate.scored(record)
    if len(scored) <= GATE_EPOCH:
        message = f"{record.name()} has {len(scored)} scored epochs - nothing to read before e{GATE_EPOCH}"
        raise SystemExit(message)

    reading = gate.reading(record, arguments.target, spread)
    logger.info("%s", reading.line())
    if reading.truncatable():
        # Non-zero so a watcher can act on it, and the sentence is the record: what it licenses is ending
        # THIS run, never a statement about the recipe, whose score becomes a floor the moment it is killed.
        logger.info("no trajectory in the corpus reaches the target - truncating leaves a FLOOR, not a number")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
