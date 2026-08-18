"""The twelve findings, in the ONE order everything downstream agrees on: `kneemri.findings`.

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
