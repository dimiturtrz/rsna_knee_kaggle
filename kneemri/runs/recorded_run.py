"""What a finished run recorded about ITSELF, from whichever of its two records can state each fact.

A run leaves two accounts and neither is guaranteed: the scores JSONL beside its checkpoint, and the
options block stamped inside the checkpoint. Of the 69 members on disk 50 wrote no metrics file at all,
and 36 of those stamp the whole recipe — so a reader that knows only the first calls three quarters of
the ledger unrecorded for want of a FILE rather than a fact. `wsfold0_e6` is the case that costs
something: it is the control knee-bot's argument rests on, it has no metrics file, and `run_comparison`
printed "recipe unrecorded, so nothing here can say the two runs are matched" while the recipe sat in
the checkpoint it had already been handed.

ONE HOME FOR THE ORDER, because the order is a judgement rather than a preference. The metrics file
wins where it exists: it names the epoch that SCORED best directly, while the stamp can only say what
the saver believed, and a run that wrote its own account has already answered. `holdout_backfill` had
this rule inline and `run_comparison` had half of it, which is how the two came to disagree.

ONE READ, because the stamp costs a `torch.load` of the whole payload. The two facts are asked for
together — a comparison wants the recipe, a back-fill wants both — so they are answered together.
"""

from dataclasses import dataclass

from kneemri.runs.member_record import StampedRun
from kneemri.runs.run_metrics import RunMetrics


@dataclass(frozen=True)
class RecordedRun:
    """One run's account of itself: the options it ran under, and which epoch's weights were kept."""

    recipe: dict[str, str] | None
    """Every option the run recorded, or None when neither record carries one. 14 members predate
    options entirely, and for those "unrecorded" is the honest answer rather than an empty recipe that
    would compare as matched against anything."""

    kept_epoch: int | None
    """The epoch whose weights are on disk — the BEST one, never the last. Under patience a later epoch
    exists in the metrics file and its weights were thrown away, so a reader that took the last one
    would describe a model that was never saved. None when neither record can say; it is not derived
    from `epochs_done`, which is what stamped `wsfold0_e6` as epoch 11 for its epoch-6 best."""

    @classmethod
    def of(cls, metrics: RunMetrics) -> "RecordedRun":
        """Read both records once and answer each fact from the one that can state it.

        Takes the metrics reader rather than a path because that is what its callers hold: the
        comparison names runs by their metrics files, the back-fill by their checkpoints, and
        `RunMetrics` already accepts either spelling and resolves the pair.
        """
        stamped = StampedRun.of(metrics.checkpoint())
        recorded, best = metrics.recipe(), metrics.best()
        return cls(
            recipe=recorded if recorded is not None else (stamped.options if stamped else None),
            kept_epoch=int(best["epoch"]) if best is not None else (stamped.kept_epoch if stamped else None),
        )
