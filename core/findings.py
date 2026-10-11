"""The twelve findings, in the ONE order everything downstream agrees on: `core.findings`.

At the package root because this is vocabulary rather than behaviour — thirteen modules speak it and
none of them trains anything. It lived in `models/model.py`, which meant a label CLI, a scorer and the
submission writer each imported a 637-line torch module holding `StudyModel` to read a tuple of
strings; `models.model` was the graph's top bottleneck (fan-in 25) largely on that traffic.

THE ORDER IS A CONTRACT, not a listing. It is the column order of the submission CSV, of every trained
member's head, and of the score matrices in between — a member fitted under one order and served under
another is silently wrong on every finding rather than loudly wrong on one.
"""

from enum import StrEnum


class Finding(StrEnum):
    """The twelve finding names. DECLARATION ORDER IS THE CONTRACT — `TARGETS` is read off it."""

    ACL = "ACL"
    MCL = "MCL"
    MEDIAL_MENISCUS = "Medial Meniscus"
    LATERAL_MENISCUS = "Lateral Meniscus"
    MEDIAL_OA = "Medial OA"
    LATERAL_OA = "Lateral OA"
    PF_OA = "PF OA"
    EFFUSION = "Effusion"
    SYNOVITIS = "Synovitis"
    BAKERS = "Baker's"
    CONTUSION = "Contusion"
    FRACTURE = "Fracture"


TARGETS: tuple[str, ...] = tuple(finding.value for finding in Finding)
"""The twelve findings, in the column order the submission and the trained members both use — plain strings."""

STUDY = "StudyInstanceUID"
"""The column every labels table is keyed by, beside the findings those tables carry.

Vocabulary, so it sits with TARGETS rather than in whichever module needed it first — which was a leaf
analysis CLI, and a second labels PRODUCER would otherwise have had to import an agreement report to
learn the name of its own index column.
"""

COMPARTMENT_PAIRS: tuple[tuple[str, str], ...] = (
    (Finding.MEDIAL_MENISCUS.value, Finding.LATERAL_MENISCUS.value),
    (Finding.MEDIAL_OA.value, Finding.LATERAL_OA.value),
)
"""The findings this corpus annotates ONCE PER COMPARTMENT: the same lesion, located to one side.

Structure in the label set rather than a fact about any model, which is why it sits beside TARGETS and
not in the two places that read it. A pair is the only case where two of the twelve columns describe
the SAME finding and differ only in where it is, so it is the only case where a study can be positive
on exactly one of them and the difference between the two columns carries information.

Named as pairs rather than derived by a prefix match on "Medial"/"Lateral". That match would also catch
MCL — the medial COLLATERAL ligament, which has no lateral counterpart annotated here — and pair it
with nothing or, worse, with whichever lateral label happened to sort beside it.
"""

SPECIFICITY = (
    "Ambiguous or borderline findings (\u201con the fence\u201d) were graded as negative to favor specificity."
)
"""The one rule that governs all twelve, stated by the host in the Overview thread (topic 733343).

At the package root with the criteria it qualifies, because it is the sentence that turns each of them
from a description into a THRESHOLD: every criterion below names what counts, and this names what to do
when the report does not clearly say.
"""

CRITERIA: dict[str, str] = {
    Finding.ACL: "A high-grade partial or full-thickness tear of the anterior cruciate ligament, meaning complete "
    "discontinuity of the ligament, or more than 50 percent of fibers disrupted, with or without secondary "
    "signs such as characteristic pivot-shift bone contusions. Mild signal change, degeneration, or "
    "thickening without discontinuity is graded negative.",
    Finding.MCL: "A high-grade partial or complete acute tear of the medial collateral ligament, with disrupted "
    "fibers and edema within and adjacent to the ligament. Low-grade sprains and chronic or remote stress "
    "changes are graded negative.",
    Finding.MEDIAL_MENISCUS: "Abnormal signal that definitely contacts the meniscal surface on at least two images, "
    "or a morphologic abnormality such as a truncated, diminutive, or displaced fragment, involving the "
    "medial meniscus. Intrasubstance degeneration that does not reach the surface is negative.",
    Finding.LATERAL_MENISCUS: "The same criteria as the medial meniscus, applied to the lateral meniscus.",
    Finding.MEDIAL_OA: "A moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss, defined as "
    "greater than 50 percent of cartilage thickness, in the medial compartment, with or without underlying "
    "subchondral marrow changes.",
    Finding.LATERAL_OA: "The same criteria as the medial compartment, applied to the lateral compartment.",
    Finding.PF_OA: "The same criteria as the medial compartment, applied to the patellofemoral compartment.",
    Finding.EFFUSION: "A moderate or large amount of fluid distending the joint.",
    Finding.SYNOVITIS: "Inflammation and thickening of the synovial lining of the joint.",
    Finding.BAKERS: "A moderate or large fluid collection in the characteristic location behind the knee.",
    Finding.CONTUSION: "A bone contusion, seen as bone marrow edema-like signal from impact, without a discrete "
    "fracture line.",
    Finding.FRACTURE: "An acute cortical break or fracture line.",
}
"""What the two annotating radiologists were told each label MEANS, quoted from the host's Overview thread.

Vocabulary at the package root because three producers now speak it and none of them owns it: the reader
that must be told the rule (`report_reader.ReadingPrompt`), the ruler that measures how far our derived
targets sit from it (`data.labels.rubric_gap`), and the severity ruler that reads the size thresholds
(`data.labels.severity_boundary`). Quoted rather than paraphrased — a paraphrase of a threshold is a
different threshold, and this project has twice quoted a constant wrong from prose.

The labels are assigned at the level of the WHOLE examination for a single knee, by two subspecialty MSK
radiologists with a third adjudicating. They are an IMAGE review. They are not a reading of the report,
which is why a report can be worded against its own label without either being wrong.
"""
