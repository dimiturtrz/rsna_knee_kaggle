"""distinct14 — honest12 plus the only two members that are DISTINCT and STRONG at once.

WHAT THIS ASKS, and it is the question every seating entry so far has failed to isolate. Five board
rows now read flat on the seating axis (raptorweight +0.001, vjepaseat -0.001, holdswap 0.000, dark7
flat, spanseat 0.944), and `knee-uco3`'s inert-seat reading is the economical explanation of all of
them. But `vjepaseat` seated a member that was DISTANT AND WEAK, and its own write-up priced what
would have been needed instead: a member at that correlation needs roughly +0.017 to +0.020 solo over
the lane's 0.8331 — that is, >= 0.8521 — to buy a seat rather than sit inert in one. No entry has ever
seated a member that satisfies both halves, because until this sweep none was known to exist.

WHERE THE TWO CAME FROM. `seat_value --against honest12` over the ten strongest unshipped members
returns eight TWINs and one SCALED against the roster's own boundaries (rank 0.8956, js 0.0150). Only
two fall on the distinct side of the rank boundary while clearing the size bar:

    DISTINCT  attend_dense_s2026  rank 0.8916  js 0.0128  maha 6.123   solo 0.8569
    DISTINCT  aprior_dense_s2026  rank 0.8811  js 0.0139  maha 6.133   solo 0.8557

Both clear 0.8521. `stack_increments_s2026` — the strongest solo the project owns at 0.8642 — reads
SCALED (rank 0.9062) and is NOT seated here, which is the point: solo strength and distinctness have
been anti-correlated across this pool, and that is the most economical account of why every seating
operation went flat.

THE CONTROL IS A LANDED ROW, NOT A PROXY. honest12 scored 0.937 on the board with exactly these twelve
under the same LogitAverage vote. This entry adds two and changes nothing else, so the delta is the two
members. That matters because the fold-0 seat proxy is KNOWN NOT to order blind seats — ref 55900461
priced a swap at +0.0022 and the board returned 0.000 — so a matched landed control is the only ruler
here that has earned trust.

WHY THE FOLDS ARE NOT MIXED, which is what disqualified the obvious entry. The champion's six are
`--fold -1` and have seen every holdout study; these two are honest-fold. Seating them beside the
champion would pay the all-data dose (measured at -0.004 to -0.005) and a flat would be unreadable —
it could be the dose or the members. Against honest12 every member is matched-fold and the dose is
absent from the comparison entirely.

PRE-REGISTERED, BEFORE THE NUMBER. Bands on the BOARD score, delta against honest12's landed 0.937.
The board quantises to 0.001 and this entry's BOARD_RESOLVABLE is 0.0015, so:

  >= 0.939   DISTINCT-AND-STRONG BUYS A SEAT. The member axis is alive and the five flats were a
             statement about WEAK distant members, not about distance. Licenses adopting the maha
             column under knee-iyad's gate, and a sweep of the remaining 149 unshipped scoreables for
             the same pair of properties.
  0.936-0.938  FLAT. Distinctness WITH strength buys nothing either, and the inert-seat reading
             survives its strongest available challenge. This closes the member axis, not just the
             seating one, and what is left is what a member IS (trunk) or what it SEES (corpus) —
             noting that `merged3_corpus_s2026` has just read TWIN, so corpus-as-implemented does not
             produce distance.
  <= 0.935   THE TWO COST. The honest greedy's twelve were already saturated and adding past the curve
             peak dilutes. Would also mean the rank boundary is not the seat criterion it is read as.

A FLAT HERE IS WORTH ITS SLOT, which is why it is being spent on this rather than held. Four of five
seating flats were consistent with "the pool has no distinct-and-strong member to seat"; this is the
first entry that can separate that from "seats are inert". No further slot goes to the member axis on
a flat.

BUDGET. Fourteen members against honest12's twelve, so the deadline projection matters more here than
it did there; the Budget is passed explicitly rather than defaulted.
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

    from kneemri.inference.budget import Budget  # noqa: PLC0415 - the kit is on sys.path only after boot
    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - same
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same

    chosen = (
        "attend_dense_s2026",
        "aprior_dense_s2026",
        "calib_dense_hold_s2026",
        "merged_blend_cf",
        "finding_calib_dense_s2026",
        "blendhold_s2026",
        "margin_seat_s2026",
        "calib_span_s2026",
        "through_dense_s2026",
        "small_batch6_s2026",
        "mae_matched_s2026",
        "base224_s2026",
        "calib_span_hold_s2026",
        "merged_dense_cf",
    )
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget())).run()


if __name__ == "__main__":
    main()
