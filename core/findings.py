"""The twelve findings, in the ONE order everything downstream agrees on: `core.findings`.

At the package root because this is vocabulary rather than behaviour — thirteen modules speak it and
none of them trains anything. It lived in `models/model.py`, which meant a label CLI, a scorer and the
submission writer each imported a 637-line torch module holding `StudyModel` to read a tuple of
strings; `models.model` was the graph's top bottleneck (fan-in 25) largely on that traffic.

THE ORDER IS A CONTRACT, not a listing. It is the column order of the submission CSV, of every trained
member's head, and of the score matrices in between — a member fitted under one order and served under
another is silently wrong on every finding rather than loudly wrong on one.
"""

TARGETS: tuple[str, ...] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
"""The twelve findings, in the column order the submission and the trained members both use."""

STUDY = "StudyInstanceUID"
"""The column every labels table is keyed by, beside the findings those tables carry.

Vocabulary, so it sits with TARGETS rather than in whichever module needed it first — which was a leaf
analysis CLI, and a second labels PRODUCER would otherwise have had to import an agreement report to
learn the name of its own index column.
"""

COMPARTMENT_PAIRS: tuple[tuple[str, str], ...] = (
    ("Medial Meniscus", "Lateral Meniscus"),
    ("Medial OA", "Lateral OA"),
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
