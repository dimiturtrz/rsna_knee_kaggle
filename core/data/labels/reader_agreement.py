"""Does the label reader agree with the radiologist equally in every language?

Run as `python -m core.data.labels.reader_agreement`.

Every score this project reports is agreement with LLM-read report labels, and a third of the cells the
reader calls positive are not positive to the radiologist. The obvious suspect is language: 59.9% of the
reports are not English (`report_language`), and "the rest needs a multilingual encoder" is the premise
branch (a) of `knee-mxj` rests on. It had never been measured.

IT IS ALSO THE ACCEPTANCE TEST FOR ANY FUTURE READER. A replacement reader has to agree with the
radiologist more than this one does, and a reader built to fix a language has to close a gap in that
language. Both questions are this module, run before and after.

THE DENOMINATOR IS 58 STUDIES — 1.32% of the corpus, all-or-nothing annotated, and the only ground truth
that is not itself the reader's output. Cells within one study share a report and a radiologist, so they
are NOT independent draws: the interval here is bootstrapped over STUDIES, not cells, and it is about
twice the width a cell-wise interval would report.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from core import cli
from core.data.labels.report_language import Language, ReportLanguages
from core.findings import STUDY, TARGETS
from core.paths import LABEL_TABLE, DataRoot

logger = logging.getLogger(__name__)

RESAMPLES = 5000
SEED = 2026
INTERVAL = (2.5, 97.5)
ENGLISH = Language.EN
POSITIVE = "YES"
SILENT = "UNK"
"""The verdict that is an abstention rather than a read — the whole subject of `silence`."""

VERDICT_SUFFIX = "__verdict"
BASE = "__base"
OTHER = "__other"
BLIND = 3
"""Exit code for a table this tool cannot distinguish from the default — see `blind`."""

MIDPOINT = 0.5
"""Where a graded annotation becomes a positive, matching `LabelScores`."""

GRADED = "graded"
HEDGED = "hedged"
"""The two report-text flags, named where the split that reads them lives — see `signal_report`."""


class ReaderAgreement:
    """How often the reader's verdict matches the radiologist, split by the report's language."""

    def __init__(self, languages: ReportLanguages | None = None) -> None:
        self.languages = languages or ReportLanguages()

    def blind(self, read: pd.DataFrame, default: pd.DataFrame) -> str:
        """Why this table cannot be READ here, or an empty string when it can.

        MEASURED 2026-08-26. Every one of the nine tables in `processed/labels/` printed byte-identical
        output through `--labels` — the same 311 positive calls, the same 140 wrong of 684, the same
        per-language rates. The flag lands; the tool is blind. `cells` merges the `__verdict` columns,
        and `table_blend` blends the NUMERIC column while carrying verdicts and `__conf` over from the
        base untouched, so every blend IS its base to everything downstream of that merge.

        THE TABLES ARE NOT THE SAME, which is what makes the silence a defect rather than a fact. Over
        the same 57 annotated studies the blends flip 7 to 33 of 684 cells across the midpoint, and macro
        AUC against the radiologist moves 0.8700 to 0.8978. The printed number was never wrong — it is a
        true statement about verdicts — but the flag invites a question this tool cannot answer, and the
        answer it prints instead reads as "every blend agrees exactly as well as v2 does".

        So a table whose verdicts are the default's while its numbers are not is REFUSED by name, and the
        refusal names the tool that scores the numeric column instead.
        """
        merged = default.merge(read, on=STUDY, suffixes=(BASE, OTHER))
        verdicts = [f"{target}{VERDICT_SUFFIX}" for target in TARGETS if f"{target}{VERDICT_SUFFIX}" in read.columns]
        spoken = all(merged[f"{name}{BASE}"].equals(merged[f"{name}{OTHER}"]) for name in verdicts)
        moved = [
            target
            for target in TARGETS
            if not np.allclose(
                merged[f"{target}{BASE}"].to_numpy(float), merged[f"{target}{OTHER}"].to_numpy(float), equal_nan=True
            )
        ]
        if not (spoken and moved):
            return ""
        return (
            f"this table's verdicts are {LABEL_TABLE}'s while its numbers differ on {len(moved)} of "
            f"{len(TARGETS)} findings ({', '.join(moved[:3])}...), so every line below would be about the "
            "DEFAULT table and say so in the language of a reading — score the numeric column with "
            "`kneemri.evaluation.table_separation` or `supervision_gap` instead"
        )

    @staticmethod
    def annotated(root: DataRoot) -> pd.DataFrame:
        """The studies a radiologist graded, in the columns `cells` joins on.

        Lives beside `cells` rather than in each caller because it is the JOIN KEY of every reading of
        the reader — a second copy is a second definition of which studies count.
        """
        truth = pd.read_csv(root.train_table())
        return truth.dropna(subset=TARGETS, how="all")[[STUDY, "Report", *TARGETS]]

    def cells(self, annotated: pd.DataFrame, read: pd.DataFrame) -> pd.DataFrame:
        """One row per (study, finding) the radiologist annotated AND the reader gave a verdict on.

        Both sides are required. A cell the reader could not read is not a cell it got wrong, and
        counting it as one would measure coverage while claiming to measure accuracy.
        """
        verdicts = [column for column in read.columns if column.endswith("__verdict")]
        merged = annotated.merge(read[[STUDY, *verdicts]], on=STUDY)
        rows = []
        for _, study in merged.iterrows():
            report = str(study["Report"])
            language = self.languages.language(report)
            # Read once per STUDY rather than per cell: both flags are properties of the report, and the
            # inner loop runs twelve times over the same text.
            flags = (self.languages.graded(report, language), self.languages.hedged(report, language))
            for target in TARGETS:
                verdict, truth = study.get(f"{target}__verdict"), study.get(target)
                if pd.isna(verdict) or pd.isna(truth):
                    continue
                rows.append((study[STUDY], language, *flags, target, str(verdict).upper(), float(truth)))
        return pd.DataFrame(rows, columns=["study", "language", GRADED, HEDGED, "target", "verdict", "truth"])

    def errors(self, cells: pd.DataFrame) -> pd.DataFrame:
        """Every cell the reader got wrong, split by DIRECTION and by what it said.

        WHY THE DIRECTION IS THE WHOLE POINT. "A third of positive calls are wrong" prices the reader
        but does not say what a replacement must do differently, and the two directions ask for opposite
        things: over-calling wants precision on YES, under-calling wants recall. Measured over the 58
        annotated studies they are not balanced — 108 false positives against 32 misses — so a
        replacement tuned for recall would make the dominant failure worse.

        SPLIT BY VERDICT TOO, because a confident error and an abstention are different defects. All
        108 false positives are confident YES, while 24 of the 32 misses are UNK: the reader is
        confidently wrong far more often than it is uncertain, which is why its constant per-verdict
        `__conf` cannot be used to downweight them.
        """
        wrong = cells[(cells["verdict"] == POSITIVE) != (cells["truth"] > MIDPOINT)].copy()
        wrong["kind"] = np.where(wrong["verdict"] == POSITIVE, "false positive", "missed")
        return wrong

    def error_report(self, cells: pd.DataFrame) -> list[str]:
        """The error census as lines: the two directions, then the findings that drive each.

        Sorted by how LOPSIDED a finding is rather than by its error count, because a finding that only
        over-calls and one that only under-calls need different work, and a ranking by volume hides
        exactly that. Synovitis is the reason: it is the sole inversion, missing 14 and over-calling 3,
        and it is also the finding whose labels predict the radiologist worst
        (`kneemri.evaluation.supervision_gap`, which read it at 0.6870 against report_labels_v2 on
        2026-08-25; quote that tool rather than this line, which has no table beside its number).
        """
        wrong = self.errors(cells)
        lines = [f"{len(wrong)} wrong of {len(cells)} cells over {cells['study'].nunique()} studies"]
        for kind, group in sorted(wrong.groupby("kind"), key=lambda item: -len(item[1])):
            said = ", ".join(f"{verdict} {count}" for verdict, count in group["verdict"].value_counts().items())
            lines.append(f"  {kind}: {len(group)}   (the reader said {said})")
        counts = wrong.groupby(["target", "kind"]).size().unstack(fill_value=0)
        for column in ("false positive", "missed"):
            if column not in counts:
                counts[column] = 0
        counts["net"] = counts["false positive"] - counts["missed"]
        lines.append(f"  {'finding':18s} {'over':>5s} {'under':>6s}   net (over minus under)")
        for target, row in counts.sort_values("net", ascending=False).iterrows():
            lines.append(f"  {target:18s} {row['false positive']:5d} {row['missed']:6d}   {row['net']:+d}")
        return lines

    def gap(self, cells: pd.DataFrame) -> dict[str, float]:
        """English minus non-English agreement on the reader's POSITIVE calls, with a study-clustered interval.

        Reported as an interval rather than a verdict because with 58 annotated studies the honest answer
        is a bound, not a number: what this can establish is the size of gap it would have SEEN.
        """
        positives = cells[cells["verdict"] == POSITIVE]
        english = positives[positives["language"] == ENGLISH]
        other = positives[positives["language"] != ENGLISH]
        deltas = self._resample(positives, english["study"].unique(), other["study"].unique())
        low, high = np.percentile(deltas, INTERVAL)
        return {
            "english": float(english["truth"].mean()),
            "other": float(other["truth"].mean()),
            "delta": float(english["truth"].mean() - other["truth"].mean()),
            "low": float(low),
            "high": float(high),
        }

    def split(self, cells: pd.DataFrame, signal: str) -> dict[str, float]:
        """Agreement on the reader's positive calls where `signal` fires, minus where it does not.

        Same shape and same study-clustered interval as `gap`, because it is the same question asked of a
        different column: is there a population of cells the reader is measurably worse on?

        A flag that fires on every study or on none returns NaNs rather than raising. That is a real
        input — `hedged` fires nowhere outside English for want of vocabulary — and the bootstrap draws
        from an empty study array, so the alternative is a crash on a corpus that is merely homogeneous.
        """
        positives = cells[cells["verdict"] == POSITIVE]
        fired, plain = positives[positives[signal]], positives[~positives[signal]]
        if fired.empty or plain.empty:
            return dict.fromkeys(("fired", "plain", "cells", "delta", "low", "high"), float("nan"))
        deltas = self._resample(positives, fired["study"].unique(), plain["study"].unique())
        low, high = np.percentile(deltas, INTERVAL)
        return {
            "fired": float(fired["truth"].mean()),
            "plain": float(plain["truth"].mean()),
            "cells": float(len(fired)),
            "delta": float(fired["truth"].mean() - plain["truth"].mean()),
            "low": float(low),
            "high": float(high),
        }

    def signal_report(self, cells: pd.DataFrame) -> list[str]:
        """Whether either report-text flag finds cells the reader is worse on — the weight-column question.

        `Targets` multiplies every cell's loss by a weight, and the reader's `__conf` column takes exactly
        one value per verdict level (std 0.0 over all 52,872 cells), so that weight is today a
        deterministic function of the target it multiplies and carries no information of its own. The
        obvious way to fill it is to read the report again: weight down the cells whose report hedges, or
        weight up the cells whose report grades. This is the measurement that prices both.

        IT CAME BACK BACKWARDS ON BOTH, which is why the flags live here as an acceptance test rather than
        in the loss. A regex cannot supply an INDEPENDENT weight, because the reader has already read the
        same sentence — its five levels ARE its confidence in the text — so where a report hedges the cell
        has already been downgraded and the residual assertion is more reliable, not less. A replacement
        reader that genuinely separates grade from certainty has to move these two numbers.

        HEDGED IS SCORED ON ENGLISH REPORTS ONLY. `HEDGES` holds no other language, so elsewhere the flag
        is False for want of vocabulary rather than for want of hedging, and pooling would compare English
        against everything.
        """
        english = cells[cells["language"] == ENGLISH]
        lines = []
        for signal, population in ((GRADED, cells), (HEDGED, english)):
            split = self.split(population, signal)
            lines.append(
                f"{signal:>7} {split['cells']:>5.0f} positive calls   agrees {split['fired']:.1%} against "
                f"{split['plain']:.1%} plain: {split['delta']:+.3f} [{split['low']:+.3f}, {split['high']:+.3f}]"
            )
        return lines

    def _resample(
        self, positives: pd.DataFrame, english: NDArray[np.object_], other: NDArray[np.object_]
    ) -> NDArray[np.float64]:
        by_study = {study: group["truth"].to_numpy() for study, group in positives.groupby("study")}
        generator = np.random.default_rng(SEED)

        def drawn(studies: NDArray[np.object_]) -> float:
            picked = generator.choice(studies, len(studies))
            return float(np.concatenate([by_study[study] for study in picked]).mean())

        return np.array([drawn(english) - drawn(other) for _ in range(RESAMPLES)])

    def by_label(self, cells: pd.DataFrame) -> dict[str, tuple[int, float]]:
        """Agreement on the reader's positive calls for each finding, as (cells, rate).

        THE RULER IS NOT EQUALLY GOOD EVERYWHERE, and that decides where local optimisation is safe.
        A label the reader is right about half the time is a label whose AUC measures agreement with a
        coin as much as it measures the model — so a change that improves the true finding there may not
        move the number, and a change that moves the number may not have improved the finding.

        Counts are returned beside every rate because they are small: twelve findings over 58 annotated
        studies leaves 16-50 positive calls each, so these separate a bad label from a good one and
        cannot rank two adjacent ones.
        """
        positives = cells[cells["verdict"] == POSITIVE]
        return {str(label): (len(group), float(group["truth"].mean())) for label, group in positives.groupby("target")}

    def silence(self, cells: pd.DataFrame) -> dict[str, float]:
        """What the reader's SILENCE actually costs, measured against the radiologist rather than assumed.

        `knee-pju4` rests on one number: 27.3% of all 52,872 (study, finding) pairs carry an UNK verdict
        and are filled with a base-rate prior instead of a read, and filling them with a local reader is
        proposed as the largest remaining lever. That is a coverage statistic. It says how much the reader
        did not say; it does not say how much of what it did not say was POSITIVE, and only the second
        number bounds what filling the silence can buy.

        This is that bound. An UNK cell the radiologist also calls negative is already right at the prior
        (0.28 sits below the midpoint), so a perfect silence-filler changes nothing there. The cells a
        perfect filler could fix are exactly the UNK cells the radiologist calls POSITIVE, and `hidden`
        counts them.

        STATE THE DENOMINATOR: 684 cells over 57 annotated studies, 1.3% of the corpus, and the only
        ground truth here that is not the reader's own output. `hidden` is a small integer and carries no
        interval — read it as a ratio against the reader's other error mode, never as a rate to project
        onto 14,445 cells.
        """
        spoken = cells[cells["verdict"] != SILENT]
        silent = cells[cells["verdict"] == SILENT]
        positive = silent["truth"] > MIDPOINT
        return {
            "cells": float(len(silent)),
            "share": float(len(silent) / len(cells)) if len(cells) else float("nan"),
            "hidden": float(positive.sum()),
            "positive_silent": float(positive.mean()) if len(silent) else float("nan"),
            "positive_spoken": float((spoken["truth"] > MIDPOINT).mean()) if len(spoken) else float("nan"),
        }

    def silence_report(self, cells: pd.DataFrame) -> list[str]:
        """The silence bound beside the over-calling it competes with, so the two are priced together.

        Printed as a comparison rather than alone because that is the decision it informs: a reader built
        to fill silence and a reader built to stop over-calling are different readers, and the annotated
        studies say which failure is larger before either is built.
        """
        census = self.silence(cells)
        wrong = self.errors(cells)
        over = int((wrong["kind"] == "false positive").sum())
        lines = [
            f"{census['cells']:.0f} silent cells of {len(cells)} ({census['share']:.1%}), the radiologist "
            f"calls {census['positive_silent']:.1%} of them positive against {census['positive_spoken']:.1%} "
            "where the reader spoke",
            f"  a perfect silence-filler fixes at most {census['hidden']:.0f} cells; over-calling costs {over}",
        ]
        silent = cells[cells["verdict"] == SILENT]
        counts = silent.groupby("target")["truth"].agg(["size", lambda column: int((column > MIDPOINT).sum())])
        counts.columns = ["silent", "hidden"]
        lines.append(f"  {'finding':18s} {'silent':>6s} {'hidden':>7s}")
        for target, row in counts.sort_values("hidden", ascending=False).iterrows():
            lines.append(f"  {target:18s} {row['silent']:6d} {row['hidden']:7d}")
        return lines

    def report(self, cells: pd.DataFrame) -> list[str]:
        """Per-language agreement on the reader's positive calls, then the English/non-English gap."""
        positives = cells[cells["verdict"] == POSITIVE]
        lines = [f"{len(cells)} annotated cells, {len(positives)} of them called positive by the reader"]
        for language, group in positives.groupby("language"):
            lines.append(f"  {language:>8} {len(group):>4} cells   radiologist agrees {group['truth'].mean():>6.1%}")
        for label, (count, rate) in sorted(self.by_label(cells).items(), key=lambda item: item[1][1]):
            lines.append(f"  {label:18s} {count:>4} calls   radiologist agrees {rate:>6.1%}")
        gap = self.gap(cells)
        lines.append(
            f"English {gap['english']:.1%} against non-English {gap['other']:.1%}: {gap['delta']:+.3f} "
            f"[{gap['low']:+.3f}, {gap['high']:+.3f}] over studies"
        )
        return lines


def main() -> None:
    """Report reader-radiologist agreement per language on the annotated studies."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--labels", type=Path, help="the reader's table; defaults to report_labels_v2.csv")
    arguments = command.parsed()

    root = command.rooted(arguments)
    fallback = root.labels()
    read = pd.read_csv(arguments.labels or fallback)

    agreement = ReaderAgreement()
    annotated = agreement.annotated(root)
    if arguments.labels and (refusal := agreement.blind(read, pd.read_csv(fallback))):
        logger.error(refusal)
        raise SystemExit(BLIND)
    cells = agreement.cells(annotated, read)
    command.report(agreement.report(cells))
    # WHAT a replacement reader must do differently, beside how well this one agrees. The two are
    # separate questions and the second is the actionable one.
    command.report(agreement.error_report(cells))
    # And what the reader's SILENCE is worth, which is the premise of every proposal to replace it.
    command.report(agreement.silence_report(cells))
    # And whether the report text holds a signal the reader's own verdict does not already carry, which
    # is the only thing that could fill the empty weight column (knee-mxj).
    command.report(agreement.signal_report(cells))


if __name__ == "__main__":
    main()
