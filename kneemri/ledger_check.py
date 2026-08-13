"""Every run that finished should appear in the ledger: `python -m kneemri.ledger_check`.

The gates catch defects in code. NOTHING CHECKS A DOCUMENT, and on 2026-08-13 that turned out to be the
larger exposure: five files were carrying conclusions that later measurement had overturned, and the
ledger itself had five arms sitting as "queued" that had run two days earlier — `slice_attention` at
0.8473, `extended_slots` at 0.8459, `intensity_augment`, `dinov3`, `calibrated_targets`. An arm that
runs and is never written down is an arm paid for and thrown away, and the next planning pass reads its
row as unstarted work.

WHAT IT CHECKS, and deliberately only this: every `*.metrics.jsonl` on disk has its best score quoted
somewhere in the ledger. It matches on the NUMBER rather than the arm name, because the table's names
are prose (`whole_stack`, `resolution`) while the files are outputs (`wholestack_s2026`,
`res392_ws_s2026`) and a mapping between them is one more thing to get wrong.

WHAT IT CANNOT CHECK: whether the surrounding sentence is true. A number can be present and its
interpretation stale — which is the other half of what went wrong that day, and needs a reader.
"""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

LEDGER = Path("experiments/README.md")
THROUGHPUT = "throughput"
"""Speed probes write metrics too, and they have no score worth quoting in a results ledger."""


class LedgerCheck:
    """Runs on disk, against the scores the ledger quotes."""

    def __init__(self, ledger: Path = LEDGER) -> None:
        self.text = ledger.read_text(encoding="utf-8") if ledger.exists() else ""

    def best(self, metrics: Path) -> float | None:
        """The best holdout mean AUC a run recorded, or None if it never scored an epoch."""
        rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
        scored = [row["auc"] for row in rows if row.get("auc") is not None]
        return max(scored) if scored else None

    def unrecorded(self, runs: dict[str, float]) -> dict[str, float]:
        """The runs whose best score appears nowhere in the ledger.

        Rounded to four places because that is how the ledger quotes them, and a run whose score is
        written to more digits elsewhere is still recorded.
        """
        return {name: score for name, score in runs.items() if f"{score:.4f}" not in self.text}


def main() -> None:
    """Report finished runs the ledger does not mention, and exit non-zero if any."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from core.paths import DataRoot  # noqa: PLC0415 - CLI-only, and this module is imported as a library

    check = LedgerCheck(arguments.ledger)
    processed = DataRoot.from_config(arguments.paths).processed()
    runs: dict[str, float] = {}
    for metrics in sorted(processed.glob("*.metrics.jsonl")):
        name = metrics.name.removesuffix(".metrics.jsonl")
        if THROUGHPUT in name:
            continue
        score = check.best(metrics)
        if score is not None:
            runs[name] = score

    missing = check.unrecorded(runs)
    for name, score in missing.items():
        logger.info("UNRECORDED %s: best %.4f appears nowhere in %s", name, score, arguments.ledger)
    logger.info("%d finished runs, %d unrecorded", len(runs), len(missing))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
