"""Read the reports with a model instead of a lexicon: `python -m core.data.labels.report_reader`.

WHAT IS BROKEN. 27.3% of this project's supervision is a CONSTANT, not a read. 14,445 of 52,872
(study, finding) pairs in `report_labels_v2.csv` carry the value 0.28 with confidence 0.05 and the
verdict `UNK` — the rung that asserts nothing. That is not spread evenly: Synovitis is 84.2% UNK,
Fracture 56.4%, Baker's 45.9%. The whole file holds eight distinct confidence values, which is the
fingerprint of a deterministic lexicon rather than of a reader, and a lexicon's failure mode is
SILENCE. It propagates all the way to the board: `merged_labels.csv` carries 14,445 cells at exactly
0.28, and the champion blend's second most common value (9,049 cells) is a prior-derived average.

THE OBVIOUS DIAGNOSIS IS WRONG, AND IT WAS MEASURED BEFORE A CARD HOUR WAS SPENT — 2026-08-27. The
standing story was multilingual: a lexicon built on English patterns goes silent on Greek, Bulgarian,
Turkish, Croatian, Dutch, German, Spanish and French reports that DO state the finding. Two things
refute it. First the UNK rate is 20.7% in ENGLISH, the language the lexicon was built for — 4,384
cells — against 42.3% Spanish, 33.6% Turkish, 22.2% Croatian. Second, and decisively: on the UNK cells
themselves, the report does not NAME the structure at all. English 0.7% of 3,550 UNK cells over nine
findings mention it; Spanish 1.2%, Turkish 3.2%, Dutch 1.6%, Greek 1.5%, each measured with that
language's own words. The silence is the REPORT's, not the reader's. A knee report discusses what was
asked about, and nothing in the text can be mined for what it never discussed.

WHAT SURVIVES IT, because the refutation is finding-shaped rather than global. Broadening the English
patterns moves two findings and only two: Baker's names a cyst in 236 of its 765 UNK cells (30.8%) and
Contusion in 55 of 342 (16.1%), while Synovitis manages 25 of 1,424 (1.8%) and Effusion 0 of 84. So
there IS text a lexicon left on the floor — roughly 5-10% of the UNK cells, concentrated in two
findings — and a global normality statement ("otherwise unremarkable") that could license an absent
call covers only 84 of 1,768 English studies. That is one to three percent of the label matrix moving
one rung. It cannot plausibly deliver the 1.5% the house bar asks for, so the FULL SWEEP is not armed.

WHICH LEAVES THIS AS A PROBE, NOT A PIPELINE. The question a sweep cannot answer cheaply is whether the
model reads the 72.7% of cells the lexicon DID read any better than the lexicon did — where
`finding_calibration` has already measured the lexicon over-calling Contusion (44.4% agreement on its
positives) and missing Synovitis. That is answerable on the 58 image-annotated studies for minutes of
card rather than hours: `--annotated`. See `LocalReader.annotated`.

THE READER INVENTS NO NUMBER. It picks a rung on the ladder `Supervision.MEASURED_AGREEMENT` already
prices, where a level MEANS the rate an RSNA radiologist agrees the finding is present: 0.08 -> 4.1%,
0.28 -> 13.6%, 0.68 -> 37.8%, 0.82 -> 68.8%, 0.94 -> 95.8%. So the question put to the model is never
"how sure are you" but `ReportGrade`'s question — what did the report SAY — and the mapping from what
it said to what that is worth stays where it was fitted. Everything downstream (`Targets`,
`ConfidentLabels`, the per-finding offsets) reads this table unchanged.

WHAT IT IS NOT ALLOWED TO DO. Local open weights only, no hosted API and no external label source: the
competition permits a model run against the reports we were given, and permits nothing that phones out.

THE GATE, BEFORE ANY SEAT IS REFITTED. `kneemri.evaluation.annotated_readings` scores a table against
the 58 image-read annotations. The new table has to beat the lexicon ON THE PAIRS THE LEXICON LEFT
BLANK — that is the only comparison that reads on this mechanism, since on the pairs it did read, the
two tables largely agree by construction. A win averaged over all pairs would be a win over a
denominator this file exists to correct.
"""

import json
import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pandas as pd

from core import cli
from core.data.labels.targets import ASSERTS_PRESENCE, UNCERTAIN_LEVEL
from core.findings import CRITERIA, SPECIFICITY, STUDY, TARGETS

logger = logging.getLogger(__name__)

REPORT = "Report"
"""The column of `train.csv` this reads. The only free text the competition gives us."""

UNSURE = "UNK"
PRESENT = "YES"
ABSENT = "NO"
"""The three verdicts `report_labels_v2.csv` carries, so the two tables are interchangeable."""


class ReportGrade(StrEnum):
    """WHAT THE REPORT SAID about one finding — never how sure the model feels about it.

    A grade is a claim about the text, which is the only thing a report reader can witness. What that
    claim is WORTH is `LEVEL`, and those numbers were fitted against image annotations elsewhere.
    """

    ABSENT = "absent"
    """The report states the structure is intact, normal, or unremarkable."""
    UNCLEAR = "unclear"
    """The report does not mention it, or mentions it without asserting either side."""
    POSSIBLE = "possible"
    """Hedged: suspicious for, cannot exclude, possible, question of."""
    PRESENT = "present"
    """Stated plainly as a finding."""
    SEVERE = "severe"
    """Stated with emphasis a reader could not disagree with: complete tear, large effusion, displaced."""


LEVEL: dict[ReportGrade, float] = {
    ReportGrade.ABSENT: 0.08,
    ReportGrade.UNCLEAR: UNCERTAIN_LEVEL,
    ReportGrade.POSSIBLE: 0.68,
    ReportGrade.PRESENT: 0.82,
    ReportGrade.SEVERE: 0.94,
}
"""Each grade on `Supervision.MEASURED_AGREEMENT`'s scale, read off the measured rate and not the wording.

  absent   -> 0.08.  An explicit statement of a normal structure. 4.1% is the most confident absence
                     the label set has, and an explicit negative is what earns it.
  unclear  -> 0.28.  The rung that asserts nothing, which is where the lexicon put everything it could
                     not read. This reader should reach it far less often, and every cell it MOVES off
                     0.28 is the deliverable.
  possible -> 0.68.  A hedge is the classic between-reader disagreement, and 37.8% describes it.
  present  -> 0.82.  A stated finding read from text rather than pixels; 68.8% is where a plain
                     report-mined positive sits, matching MRNET_LEVEL's own positive.
  severe   -> 0.94.  Reserved for a finding stated so it cannot be argued with. `rijeka_corpus` gives
                     this rung to a complete rupture read off the images; an unhedged severe call in
                     the text is the nearest a report gets, and nothing weaker may take it.
"""

CONFIDENCE: dict[str, float] = {PRESENT: 0.95, ABSENT: 0.85, UNSURE: 0.05}
"""The `__conf` column, kept at the lexicon's three values so the schema is unchanged.

It carries no information here and carried little there — it is constant per verdict in both tables,
and nothing downstream reads it (`ConfidentLabels` reads the verdict, `Targets` reads the value). It is
written so a consumer of the old table is not broken by the new one, not because it means anything.
"""

MAX_NEW_TOKENS = 320
"""Twelve grades as JSON is ~90 tokens; this leaves room for a preamble without paying for an essay."""

DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
"""A 35B mixture-of-experts with 3B active, int4, already in the local HF cache.

Chosen for the axis that is failing: it is multilingual, and the blanks are Greek, Bulgarian, Turkish,
Croatian, Dutch, German, Spanish and French. 3B active weights is what makes 4,406 reports affordable
on one card in a window measured in hours rather than days.
"""

JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
"""The model is asked for bare JSON and mostly gives it; this survives the times it adds a sentence."""


@dataclass(frozen=True)
class ReadingPrompt:
    """The instruction put to the reader, and the only place the task is stated in words.

    Separate from the model so it can be read, diffed and tested without loading 20 GB. A prompt is a
    parameter of this experiment, not an implementation detail of it.
    """

    findings: tuple[str, ...] = TARGETS

    def criteria(self) -> str:
        """The host's published criterion for each finding, in the contract order, one per line.

        Rendered from `core.findings.CRITERIA` rather than restated here so the reader is told the same
        words `data.labels.rubric_gap` measures it against. A prompt that paraphrased them would be
        scored by a ruler reading the originals.
        """
        return "\n".join(f"- {finding}: {CRITERIA[finding]}" for finding in self.findings)

    def system(self) -> str:
        """What the reader is, what it may claim, and the vocabulary it must answer in.

        THE RUBRIC IS IN HERE BECAUSE THE TARGET IS NOT THE REPORT. The labels are an IMAGE review by two
        MSK radiologists under stated criteria, with a third adjudicating; the report is a separate
        document written for a different purpose. A reader told only to transcribe therefore grades
        "intrasubstance degeneration without a definite tear" above absent, and the competition's label
        for it is 0 — 448 such cells are measured in `rubric_gap`, across four of the twelve columns.

        What does NOT change is that the reader may only use the REPORT: it still may not diagnose,
        infer across findings, or imagine an image. The criteria are a THRESHOLD applied to what the
        report said, not a licence to decide what the images would have shown.
        """
        grades = ", ".join(f"{grade.value} ({grade.__doc__})" for grade in ReportGrade)
        return (
            "You read knee MRI radiology reports. Reports come in English, Spanish, Turkish, Croatian, "
            "Greek, German, Bulgarian, Dutch and French. Read the report in the language it is written "
            "in; never translate before deciding.\n\n"
            "For each of the twelve findings, say what THE REPORT SAID about it. Do not diagnose, do not "
            "infer from other findings, and do not guess what an image would show. If the report is "
            "silent about a finding, that is 'unclear' — silence is not a normal result.\n\n"
            "The findings are scored against an image review by radiologists working to the criteria "
            "below. Apply these criteria to what the report says. A finding the report describes but "
            f"that falls UNDER its criterion is 'absent', not 'possible'. {SPECIFICITY}\n\n"
            f"{self.criteria()}\n\n"
            f"Grades: {grades}\n\n"
            "Answer with a single JSON object mapping each finding name to one grade word. No prose."
        )

    def user(self, report: str) -> str:
        """The report itself, beside the exact finding names the answer must be keyed by."""
        names = ", ".join(self.findings)
        return f"Findings: {names}\n\nReport:\n{report}"

    def messages(self, report: str) -> list[dict[str, str]]:
        """The chat turns a causal reader is templated with."""
        return [
            {"role": "system", "content": self.system()},
            {"role": "user", "content": self.user(report)},
        ]


@dataclass(frozen=True)
class ReportReading:
    """One report's twelve grades, and the label-table row they become."""

    study: str
    grades: dict[str, ReportGrade]

    @classmethod
    def parse(cls, study: str, answer: str) -> "ReportReading":
        """A model answer read into grades, tolerantly — an unreadable finding falls to `UNCLEAR`.

        Falling back rather than raising is deliberate: `UNCLEAR` is the value the lexicon already
        assigned to everything it could not read, so a parse failure leaves that cell no worse than it
        is today, while a raise would cost the other eleven findings in the same report.
        """
        match = JSON_OBJECT.search(answer)
        spoken: dict[str, str] = {}
        if match is not None:
            try:
                loaded = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning("study %s: answer was not JSON", study)
            else:
                spoken = {str(key): str(value) for key, value in loaded.items()}
        return cls(study, {target: cls.grade_of(spoken.get(target, "")) for target in TARGETS})

    @staticmethod
    def verdict_of(level: float) -> str:
        """Which of the three verdict strings a level asserts."""
        if level == UNCERTAIN_LEVEL:
            return UNSURE
        return PRESENT if level > ASSERTS_PRESENCE else ABSENT

    @staticmethod
    def grade_of(word: str) -> ReportGrade:
        """One spoken word as a grade, or `UNCLEAR` if it is not one of the five."""
        try:
            return ReportGrade(word.strip().lower())
        except ValueError:
            return ReportGrade.UNCLEAR

    def row(self) -> dict[str, str | float]:
        """This reading in `report_labels_v2.csv`'s schema — value, confidence and verdict per finding."""
        row: dict[str, str | float] = {STUDY: self.study}
        for target in TARGETS:
            level = LEVEL[self.grades[target]]
            verdict = self.verdict_of(level)
            row[target] = level
            row[f"{target}__conf"] = CONFIDENCE[verdict]
            row[f"{target}__verdict"] = verdict
        return row

    def unread(self) -> int:
        """How many of the twelve stayed on the rung that asserts nothing — the number this must lower."""
        return sum(grade is ReportGrade.UNCLEAR for grade in self.grades.values())


class LocalReader:
    """The open-weights reader itself: reports in, gradings out, nothing leaving the machine."""

    def __init__(self, model_id: str = DEFAULT_MODEL, prompt: ReadingPrompt | None = None) -> None:
        self.model_id = model_id
        self.prompt = prompt or ReadingPrompt()
        self._loaded: tuple[object, object] | None = None

    @staticmethod
    def annotated(train_table: Path) -> pd.DataFrame:
        """The 58 studies a radiologist annotated on EVERY finding, with their reports.

        THE PROBE THIS EXISTS FOR, and why it comes before the sweep. Reading all 4,406 reports costs
        hours of card; reading these 58 costs minutes, and they are the ONLY studies where the reader's
        answer can be checked against pixels rather than against another text reader. So the sweep is
        gated on this: grade the annotated studies, compare both tables to the annotations on the same
        696 cells, and spend the hours only if the model is the better reader. All-or-nothing rows, as
        `SupervisionGap.annotated` takes them — train.csv annotates a study on all twelve or on none.
        """
        table = pd.read_csv(train_table)
        return table.dropna(subset=list(TARGETS), how="any")[[STUDY, REPORT]]

    def load(self) -> tuple[object, object]:  # devtools-ignore: test-mirror
        """Tokenizer and model, once. Imported here because a unit test may never pay for transformers."""
        if self._loaded is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

            logger.info("loading reader %s", self.model_id)
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto")
            self._loaded = (tokenizer, model)
        return self._loaded

    def answer(self, report: str) -> str:  # devtools-ignore: test-mirror
        """The model's raw text for one report."""
        tokenizer, model = self.load()
        text = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
            self.prompt.messages(report), tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)  # type: ignore[operator, attr-defined]
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)  # type: ignore[attr-defined]
        fresh = generated[0][inputs["input_ids"].shape[-1] :]
        return str(tokenizer.decode(fresh, skip_special_tokens=True))  # type: ignore[attr-defined]

    def read(self, study: str, report: str) -> ReportReading:  # devtools-ignore: test-mirror
        """One report, graded."""
        return ReportReading.parse(study, self.answer(report))

    def table(self, reports: pd.DataFrame) -> pd.DataFrame:  # devtools-ignore: test-mirror
        """Every report graded into a label table with `report_labels_v2.csv`'s columns."""
        rows: list[dict[str, str | float]] = []
        pairs = zip(reports[STUDY], reports[REPORT], strict=True)
        for position, (study, report) in enumerate(pairs):
            reading = self.read(str(study), str(report))
            rows.append(reading.row())
            if position % LOG_EVERY == 0:
                logger.info("read %d of %d, %d unclear", position, len(reports), reading.unread())
        return pd.DataFrame(rows)


LOG_EVERY = 100
"""How often the sweep says where it is. 4,406 reports is long enough to want a pulse, short enough
that a line per report would be the majority of the log."""


def main() -> None:
    """Grade every report in `train.csv` and write the table beside the lexicon's."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--reports", type=Path, required=True, help="train.csv, with its Report column")
    command.parser.add_argument("--out", type=Path, required=True, help="where the label table is written")
    command.parser.add_argument("--model", default=DEFAULT_MODEL, help="local open-weights reader")
    command.parser.add_argument("--limit", type=int, default=0, help="read only the first N reports")
    command.parser.add_argument(
        "--annotated", action="store_true", help="read ONLY the image-annotated studies — the gate before the sweep"
    )
    options = command.parsed()

    if options.annotated:
        reports = LocalReader.annotated(options.reports)
    else:
        reports = pd.read_csv(options.reports, usecols=[STUDY, REPORT])
    if options.limit:
        reports = reports.head(options.limit)
    table = LocalReader(options.model).table(reports)
    table.to_csv(options.out, index=False)
    unclear = float((table[[f"{target}__verdict" for target in TARGETS]] == UNSURE).to_numpy().mean())
    logger.info("wrote %s: %d studies, %.1f%% of pairs unread", options.out, len(table), 100 * unclear)


if __name__ == "__main__":
    main()
