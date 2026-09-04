"""colnull — every finding column a constant. The metric's FORM, read directly off the board.

**WHAT IS BEING MEASURED.** Not a method, not a member, not a seating: the shape of the scoring function.
`comp/README.md` has recorded since the first week that the formula is unread — the evaluation page is
JavaScript-only and the API does not serve it — and every per-label and per-column number this project has
priced rests on the assumption that the score is an unweighted mean of twelve per-column AUCs.

**THE MECHANISM, and why this entry needs no model at all.** The metric is per-column and rank-based:
`kneemri-monotone` shipped a monotone transform of the champion's table and read the champion's exact 0.944.
A column holding one constant therefore has an AUC of exactly 0.5, whatever the constant is — no seed, no
variance, no calibration. Fill EVERY column with a constant and a mean-of-per-column-AUCs metric must return
exactly 0.500. Nothing else in this project can produce a number that is known in advance to three decimals,
which is what makes this a measurement of the formula rather than of us.

It is the companion to `kneemri-colmcl`, which damages ONE column and reads that column's share. This one
reads the form; that one reads the weight. Neither is readable without the other: a single-column drop can
be explained by a weight or by a different board-side AUC, and this entry removes the second explanation by
pinning the intercept.

PRE-REGISTRATION, written before the number.

- **exactly 0.500** — the score is a mean of per-column AUCs with no offset and no non-AUC term. Every
  per-column reading in `interpretations/` is priced on the right formula, and `colmcl`'s drop is then a
  clean read of MCL's weight.
- **0.495-0.505 but not 0.500** — a mean of per-column AUCs with a tie-handling or a per-column
  prevalence weighting that a constant column does not resolve to exactly a half. Small, but it means the
  formula carries a term nothing here has modelled.
- **anything else** — the score is NOT a mean of per-column AUCs. Every per-label plan this project holds is
  priced on a formula that does not hold, and the ledger's per-column arithmetic has to be re-derived before
  another arm runs on the label axis. This is the branch worth a slot.

**COST, STATED HONESTLY.** This entry scores about 0.5 — the worst row this project will ever have put on
the board, on purpose. It buys the metric's form for one slot, no card time, and no members: it does not
load a checkpoint, which is also why `run_check`'s member census does not apply to it. It is an INSTRUMENT.

The refutation this cannot make: it says nothing about which columns are weighted, only whether the total is
a plain mean of per-column AUCs. That is `colmcl`'s question.
"""

# >>> KERNEL-BOOTSTRAP
import logging
import os
import sys
import zipfile
from pathlib import Path

SERIES = {"train_series", "test_series", "train_images", "test_images"}
"""Competition mounts holding one directory per study. `rglob` walks every one of them."""

KIT_DEPTH = 5
"""How deep under /kaggle/input a mounted kit has ever sat, plus room."""


def find_kit() -> Path:
    """The directory to put on `sys.path`, found WITHOUT walking the competition data."""
    root = Path("/kaggle/input")
    for parent, directories, _ in os.walk(root):
        here = Path(parent)
        depth = len(here.relative_to(root).parts)
        directories[:] = [] if depth >= KIT_DEPTH else [name for name in directories if name not in SERIES]
        if here.name == "kneemri" and (here / "__init__.py").exists():
            return here.parent
        archive = here / "kneemri_src.zip"
        if archive.exists():
            zipfile.ZipFile(archive).extractall("/tmp/kit")  # our own kit, built by build_kit.py
            return next(iter(sorted(Path("/tmp/kit").rglob("kneemri/__init__.py")))).parent.parent
    raise FileNotFoundError("no kneemri package under /kaggle/input")


# <<< KERNEL-BOOTSTRAP

CONSTANT = 0.5
"""What every finding column is filled with. Any single value gives a column an AUC of exactly 0.5 under a
rank metric; a half is chosen so a reader of the file can see at a glance that it is not a score."""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.path.insert(0, str(find_kit()))

    import pandas as pd  # noqa: PLC0415 - the kit is on sys.path only after find_kit

    from core.findings import TARGETS  # noqa: PLC0415 - same
    from core.mounts import Mounts  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import WORKING  # noqa: PLC0415 - same

    # The sample submission, not our own test frame: this entry deliberately loads no model, so the ONLY
    # thing it can be sure of is the row set the competition itself asked for.
    entry = pd.read_csv(Mounts().one("sample_submission.csv", "competition data"))
    for target in TARGETS:
        entry[target] = CONSTANT
    logging.getLogger(__name__).info(f"{len(entry)} studies, {len(TARGETS)} columns, every one at {CONSTANT}")
    entry.to_csv(WORKING / "submission.csv", index=False)


if __name__ == "__main__":
    main()
