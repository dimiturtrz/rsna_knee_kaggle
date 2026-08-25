"""What the VRAM cost model DERIVES against what the arms MEASURED: `python -m
kneemri.experiments.budget_residuals`.

`DeviceBudget` is the piece of this repo that has been wrong most often — five arms have died to a
term of it being fitted on one configuration and handed to another, and every one of those fixes was
argued from a single incident quoted in a bead. Meanwhile every run since 2026-08-11 has been writing
`vram_peak_gb` into its own epoch rows, so the model has had a held-out test set of seventy-four points
sitting on disk the whole time and nothing that read it. This is that reader.

IT GRADES THE MODEL, NOT A SHIPMENT. The other checks here refuse to let something bad LEAVE — a
mis-armed queue, an unmounted kit, a truncated score. This one asks whether a NUMBER the repo derives
matches the world, which is a question no amount of care at the call site can answer.

THE DENOMINATOR IS THE WHOLE DIFFICULTY. Rows written before 2026-08-22 recorded
`torch.cuda.max_memory_allocated` under the name `vram_peak_gb` (see `RunMetrics.record`), and the
model predicts what the ALLOCATOR RESERVES, which is the larger quantity. Grading against the old
spelling would score the model against a number it is not trying to predict and flatter it by the
fragmentation — so a run is priced only when it carries `vram_alloc_gb`, the field whose presence marks
the corrected reading. That is 74 of the 104 metrics files on disk, and the 30 excluded are COUNTED in the header rather
than silently averaged in.

WHY IT IS NOT IN `nox` YET: it fails today. Three recorded runs measured MORE than the model would
have allowed them, which is the one failure `DeviceBudget` calls fatal, and they are filed rather than
suppressed (knee-bj90 and the two beads it spawned). It graduates to a gate when `--assert` passes
clean.
"""

import logging
from dataclasses import dataclass, replace

from core import cli
from kneemri.models.backbone import ENCODERS
from kneemri.runs.run_metrics import METRICS_RECORD, RunMetrics
from kneemri.training.device_budget import DeviceBudget
from kneemri.training.options import TrainingOptions
from kneemri.training.run_setup import RunSetup

logger = logging.getLogger(__name__)

RESERVED = "vram_peak_gb"
"""What the allocator held at its peak — the quantity `DeviceBudget.needed` is a model of."""

ALLOCATED = "vram_alloc_gb"
"""What the run's own tensors held at their peak. Read here only as the MARK of a corrected row: it
appeared in the same commit that made `vram_peak_gb` mean reserved, so its presence dates the file."""


@dataclass(frozen=True)
class PricedRun:
    """One finished run, as the cost model would have priced it and as it actually measured."""

    name: str
    budget: DeviceBudget
    measured: float
    """The largest reserved peak any epoch of the run recorded, in GB."""

    @classmethod
    def of(cls, metrics: RunMetrics) -> "PricedRun | None":
        """Price a recorded run, or None when it cannot be priced against the corrected reading.

        Three ways a run declines to be priced, and all three are silence rather than an error: it
        recorded no recipe (14 members predate options entirely), it recorded no peak under the
        corrected spelling, or its recipe names an encoder this tree no longer has. A reader over the
        whole ledger meets all three and must not stop at the first.
        """
        recipe = metrics.recipe()
        peaks = [row[RESERVED] for row in metrics.rows() if RESERVED in row and ALLOCATED in row]
        if recipe is None or not peaks:
            return None
        options = TrainingOptions.of_recipe(recipe)
        if options.variant not in ENCODERS:
            return None
        return cls(metrics.name(), RunSetup(options).budget(), max(peaks))

    @property
    def derived(self) -> float:
        """What the model says this recipe reserves, WITHOUT the slack `needed` adds on top.

        The slack is a slice of margin against the granularity at which the quantity moves; grading the
        model against it would credit the model for its own safety factor and hide a term that is a
        slice too small. `needed` minus one slice is the model's actual claim about the world.
        """
        return self.budget.needed() - self.slack

    @property
    def slack(self) -> float:
        """The one slice of margin `needed` carries — what a residual may reach before an arm dies.

        Asked of the model rather than reassembled from its constants: `needed` at this recipe minus
        `needed` at one slice fewer IS the per-slice term, whatever the model's shape becomes. A reader
        that spelled out `GB_PER_SLICE * step` would be a second copy of the formula it is grading.
        """
        return self.budget.needed() - replace(self.budget, slices=self.budget.slices - 1).needed()

    @property
    def residual(self) -> float:
        """Measured minus derived, in GB. POSITIVE is the dangerous sign: the model is under the truth."""
        return self.measured - self.derived

    @property
    def refused(self) -> bool:
        """Whether this run would have been capped BELOW what it went on to reserve.

        The one failure `DeviceBudget` calls fatal rather than expensive: a cap under a measured demand
        prevents nothing and guarantees the OutOfMemoryError. An overstated cap only costs polling.
        """
        return self.measured > self.budget.needed()

    def line(self) -> str:
        """One run on one line, in the order a reader scans: how wrong, which way, and on what."""
        share = 100.0 * self.residual / self.derived if self.derived else 0.0
        mark = "  CAP UNDER DEMAND" if self.refused else ""
        return (
            f"  {self.name:<32} measured {self.measured:6.2f}  derived {self.derived:6.2f}  "
            f"residual {self.residual:+6.2f} GB ({share:+5.1f}%){mark}"
        )


def main() -> None:
    """Report every priceable run against the model, worst under-derivation last."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument(
        "--assert",
        dest="asserted",
        action="store_true",
        help="exit non-zero when any recorded run measured more than the model would have allowed it",
    )
    arguments = command.parsed()

    processed = command.rooted(arguments).processed()
    found = sorted(processed.glob(f"*{METRICS_RECORD}"))
    priced = [run for run in (PricedRun.of(RunMetrics(path)) for path in found) if run is not None]
    logger.info("%d recorded runs, %d priceable against the corrected reading", len(found), len(priced))
    command.report(run.line() for run in sorted(priced, key=lambda run: run.residual))

    refused = [run for run in priced if run.refused]
    if not refused:
        logger.info("no recorded run measured more than the model would have allowed it")
        return
    logger.warning("%d runs would have been CAPPED BELOW their own measured demand:", len(refused))
    command.report(
        f"  {run.name}: measured {run.measured:.2f} GB, allowed {run.budget.needed():.2f}" for run in refused
    )
    if arguments.asserted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
