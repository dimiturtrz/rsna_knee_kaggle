"""SIX HONEST FOLD-0 MEMBERS, SEATED -- the (offline, board) pair the record does not have, and a
champion attempt in the same entry.

**THE HOLE THIS FILLS.** D372 (researcher, 2026-09-11) interrogated `board_transfer` and found the
record cannot price the leader gap at all. Every one of the 21 ensemble rows at board >=0.940 -- the
entire band the champion lives in -- carries `offline: None`, honestly recorded, because their `ff`
and `cf` seats were fitted at `--fold -1` and have seen all 909 holdout studies. The highest rows
holding BOTH numbers are 0.935-0.937, seven of them, spanning 0.0046 of offline. Any transfer slope
quoted at 0.944 is an extrapolation past the last resolved pair by the whole width of the band.
D372 also retired the +1.55 this project has priced arms against all week: `board_transfer.py:24,34`
says in its own docstring that every number in it is history, and the live whole-record figure is
+1.04 (45 entries, 828 pairs), saturating to +0.99 [+0.94,+1.05] above 0.8668 offline and +0.82
[+0.71,+0.93] above 0.8695 (4 pairs, tagged DIRECTION ONLY by the instrument itself).

This entry is built ONLY from members that hold out all 909, so it has an honest offline number AND
will have a board score. That is the pair the 0.940+ band has never produced.

**THE SEATING, AND WHY IT INVOLVED NO SEARCH.** The six are the top six of the 133 PACKAGED members
that pass all three honesty filters (`epoch >= 0`; identical `studies` vector to holdout909;
identical binarised `truth` cell-for-cell), ranked by SOLO macro AUC, with `oof_teacher_s2026`
excluded. No combination was searched: ranking on solos and taking the first six is the whole rule,
which is what keeps the offline number from carrying the combination-selection bias I measured
tonight at ~0.011 on per-column maxima over this same pool.

    clean_calib_s2026          0.8649
    chainfour_s2026            0.8648
    stack_increments_s2026     0.8642
    stack_increments_base      0.8635
    calib_dense_hold_s2026     0.8634
    seat_clean_distill_s2026   0.8634

**OFFLINE, COMPUTED BEFORE THE PUSH: 0.8737 macro AUC @holdout909 (cut-free, LogitAverage of the six
logits).** For scale, the best single honest packaged solo is 0.8685 and the best excluding the
leaked one is 0.8649, so the seating is +0.0088 over its own best member. Sensitivity: the same rule
at 5 seats reads 0.8738 and at 7 reads 0.8735 -- the number is flat in the seat count, which is what
says it is not a lucky cut.

**WHY oof_teacher IS EXCLUDED.** `comp/board.jsonl:118` (`oofsolo_v1`, ref 56065873) records its
0.8685 as LEAKED: it was distilled from `oof_fold1..4`, each fitted at `--fold k`, and
`ReportFolds.holdout` puts fold 0 in every one of those training sets. Its own arm header later
measured the leak at ~0.0006 against a `--exclude-fold 0` twin and downgraded the void to a
retraction of the ARGUMENT. Both documents are in the tree and they disagree; an entry whose whole
purpose is to carry an honest offline number does not get to depend on which one is right.

**PRE-REGISTRATION, written before the score.** Anchor: `stackfit_s2026`, 0.8631 offline -> 0.933
board, the cleanest resolved pair on this side of the record. This entry is +0.0106 of offline above
it, so the live regimes predict:

  - **>= 0.945** -- the champion is beaten by an all-honest seating and the offline ruler orders at
    the operating point after all. Every arm can be priced against `holdout909` again.
  - **0.940-0.944** -- at the champion. Transfer is near the mid regime (+0.99 predicts 0.9435) and,
    win or not, the record finally gains a resolved (offline, board) pair inside the top band, which
    is the whole reason this entry exists.
  - **0.933-0.939** -- the saturation D372 measured is confirmed on the ensemble side: +0.0106 of
    honest offline buys at most half a board point per offline point up here.
  - **<= 0.932** -- at or below `stackfit_s2026`'s solo. Then the `--fold -1` seats in the champion
    are carrying real board value rather than only holdout leakage, and "honest members only" is a
    measurably worse seating rather than a cleaner one. That would be the most useful branch of the
    four and it is named here so it cannot be re-read as a disappointment.

**WHAT IT IS NOT.** It is not a reseating of the champion; it shares no member with it. It is not
chosen on the gold or corrected rulers (`cleanseat`/`goldseat`/`tablepool` occupy that thread). And
it claims no attribution: six members change at once against every prior entry, deliberately.
"""

# >>> KERNEL-BOOTSTRAP
import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

SERIES = {"train_series", "test_series", "train_images", "test_images"}
"""Competition mounts holding one directory per study. `rglob` walks every one of them."""

KIT_DEPTH = 5
"""How deep under /kaggle/input a mounted kit has ever sat, plus room. A bound rather than a guess: it
turns an unbounded walk into a bounded one even if a mount arrives with a name this does not know."""


def find_kit() -> Path:
    """The directory to put on `sys.path`, found WITHOUT walking the competition data.

    The obvious spelling — `Path("/kaggle/input").rglob("kneemri/__init__.py")` — cost **248 seconds** on
    2026-08-14, four minutes before a single line of ours ran. It descends into the competition mount,
    which holds one directory per study and one per series beneath that, and stats every DICOM on the
    way. The efficiency board scores total runtime, so this was our largest single expense and it was
    spent looking for a file we already knew the neighbourhood of.

    Pruning by name is what removes the cost; the depth bound is what keeps a differently-named mount
    from restoring it silently.
    """
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


def install_wheels(kit: Path) -> None:
    """Install the kit's wheels offline, before anything imports our package.

    The runtime probe checked the libraries the PIPELINE needs (pydicom, timm, transformers…) and found
    them all present — but not the ones our own modules import. `jaxtyping` is not in Kaggle's image and
    `kneemri.models.model` imports it at module top, so the kit needs wheels after all. Probe what you import,
    not only what you plan to use.

    Names what it installs, because a kit shipped without the wheel it needed fails much later, inside
    an import, in a log nobody reads until the slot is spent.
    """
    wheels = sorted(kit.parent.glob("*.whl")) + sorted(kit.glob("*.whl"))
    if not wheels:
        logging.getLogger(__name__).info("no wheels in the kit — assuming every import is already in the image")
        return
    logging.getLogger(__name__).info(f"installing {[wheel.name for wheel in wheels]}")
    subprocess.run(  # noqa: S603 - pip installing wheels we mounted ourselves, fixed argv, no shell
        [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *[str(w) for w in wheels]],
        check=True,
    )


def boot() -> Path:
    """Everything a kernel does before its first line of ours can run, as one call.

    Fifty-three kernels opened `main` with these same four statements in this same order, because the
    order is forced: nothing may import `kneemri` until the wheels are installed, and nothing may find
    the wheels until the kit is found. Four lines that MUST NOT be reordered are a step, not a preamble,
    and a step copied fifty-three times is fifty-three places for the order to be got wrong once.

    Returns the kit so a kernel that needs a path into it has one; most only needed it for `sys.path`.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    kit = find_kit()
    install_wheels(kit)
    sys.path.insert(0, str(kit))
    return kit


# <<< KERNEL-BOOTSTRAP


def main() -> None:
    boot()

    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - the kit is on sys.path only after boot

    chosen = (
        "clean_calib_s2026",
        "chainfour_s2026",
        "stack_increments_s2026",
        "stack_increments_base_s2026",
        "calib_dense_hold_s2026",
        "seat_clean_distill_s2026",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
