"""An arm's PRECONDITION, as a line something can evaluate rather than a sentence in its header.

WHY THIS EXISTS. `final_fit_six.sh` opened with its own stop condition — "Run this ONLY if
`kneemri-finalfit` reads 0.936 or above on the board. At 0.934-0.935 keep the one member and stop."
The board read 0.934. The arm was launched anyway and ran four members to completion plus a fifth to
epoch 10 of 12, roughly five hours of card, and was stopped only because a peer project asked for the
machine — not because anything re-read the header.

The arm was not wrong: the gate was written BEFORE the reading, priced the decision honestly, and the
cost-tiered design worked exactly as intended. The defect is that its verdict had no consumer. Nine
pre-flight checks guard shipments and processes and not one of them asked whether an arm's own
precondition still holds, because the precondition lived in a bash comment, which is the one place in
this repository where nothing executes.

So it becomes data: `# gate: kneemri-finalfit >= 0.936`, evaluated against `comp/board.jsonl`, which
already holds every scored entry in the form something can read.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

GATE_LINE = re.compile(r"^#\s*gate:\s*([A-Za-z0-9_-]+)\s*>=\s*([0-9]*\.?[0-9]+)\s*$", re.MULTILINE)
"""How an arm declares what must already be true before it may spend a card.

One entry and one floor, because a gate a launcher cannot evaluate is the comment this replaces. The
prose stays — it says what to do at each band, which no threshold can — but the line is what refuses.
"""

BOARD = Path("comp/board.jsonl")
"""Every scored entry, one JSON object a line. `id` names the entry, `kernel` the slug it was pushed as."""


@dataclass(frozen=True)
class ArmGate:
    """The board reading an arm waits on, and the score it needs before it is allowed to run."""

    entry: str
    threshold: float

    @classmethod
    def declared(cls, header: str) -> "ArmGate | None":
        """The gate this arm's header declares, or None where it declares none.

        Silence is not a fault: most arms are ungated, and an arm that answers its own question has
        nothing to wait for. Only a declared gate can be unmet.
        """
        match = GATE_LINE.search(header)
        return None if match is None else cls(match.group(1), float(match.group(2)))

    def unmet(self, board: Path = BOARD) -> str:
        """Why this arm may not run, or the empty string when the board says it may.

        AN UNREAD GATE REFUSES TOO, and that is the whole point of naming the entry rather than the
        score: an arm gated on a submission nobody has pushed yet is exactly the arm most likely to be
        launched on a hunch, and "the board has never scored it" is the sentence that stops that. A
        missing board file is the same answer for the same reason — a check that passes because its
        evidence is absent is the failure mode this repository keeps paying for.
        """
        if not board.is_file():
            return f"gated on {self.entry} >= {self.threshold:.3f} and {board} is not on disk"
        read = self.reading(board)
        if read is None:
            return f"gated on {self.entry} >= {self.threshold:.3f}, which the board has never scored"
        if read < self.threshold:
            return f"gated on {self.entry} >= {self.threshold:.3f}; the board reads {read:.3f}"
        return ""

    def reading(self, board: Path = BOARD) -> float | None:
        """The best board score recorded for this gate's entry, or None where nothing names it.

        BEST rather than latest, because a gate asks whether a method has ever cleared a bar, and a
        re-push of the same slug at a worse score does not retire the reading that licensed the arm.
        Both `id` and `kernel` are matched: fourteen of the thirty-four rows predate the `kernel` field,
        so a gate naming a slug would silently be unread over exactly the older half of the record.
        """
        rows = [json.loads(line) for line in board.read_text(encoding="utf-8").splitlines() if line.strip()]
        scores = [float(row["board"]) for row in rows if self.entry in {row.get("id"), row.get("kernel")}]
        return max(scores) if scores else None
