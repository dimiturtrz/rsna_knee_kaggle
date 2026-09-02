"""The five fold-0 twins, shipped as themselves — an INSTRUMENT, not a candidate.

**WHAT IS BROKEN.** `board_transfer` prices every arm in this project by fitting board points against
offline holdout points. Asked for the compression at or above 0.8700 it answers **`0 pairs, median
+0.00`**: the record holds 37 paired offline-and-board entries and they top out at offline 0.8751 /
board 0.937. The champion `raptorweight_v1` ships at **0.944 with no offline number at all** — it
carries a foreign seat whose holdout is 98.6% in-sample, so it has none to record. Every entry above
0.937 on this board was built on a slope extrapolated past every point that fitted it.

**WHAT THIS ENTRY BUYS.** One paired point at the top of the curve. The five members below are all
`--fold 0`: their holdout studies were withheld from their own training, so the pool has an HONEST
offline number, and shipping exactly that pool — same names, same vote rule, same serving — makes the
first offline-to-board pair the record has ever had above 0.870.

**THE SECOND THING IT SETTLES.** `aucvote_v1` (0.940) and `stackvote_v1` (0.943) both fit their vote
rules on the fold-0 twins and then SHIPPED the `_ff` / `_cf` members, because those are the ones the
champion seats. So every combiner reading this project has taken is confounded: a rule fitted on one
set of weights and served by another. This entry ships the pool the rules were fitted on. Its board
number minus the champion's is the twin-to-shipped gap, measured once, and it prices that confound
for every combiner reading already on the board.

**WHY IT IS NOT A CANDIDATE, and is not being sold as one.** These are the fold-0 twins, each trained
on five sixths of the corpus; the shipped `_ff` / `_cf` members see all of it. The honest expectation
is BELOW the champion. Under CLAUDE.md's pacing rule this slot may go to an instrument or to something
mechanism-backed; nothing mechanism-backed is ready, both halves of the ensemble axis read flat across
nine board entries, and an unspent slot is destroyed at 00:00 UTC. This is the instrument.

PRE-REGISTRATION, written before the board number, and against the OFFLINE number rather than the
champion — because a compression reading is what the slot is being spent on.

**THE OFFLINE NUMBER, MEASURED BEFORE THE PUSH.** `vote_ceiling` over these exact five names, 909
shared studies, weights fitted on one half and scored on the other: RankAverage 0.8700, LogitAverage
**0.8706** (+0.0006, overfit gap +0.0000), MedianLogit 0.8676. `LogitAverage` is the rule shipped
below and is the family best, as it was at every other pool size. Prose carried 0.8745 for this pool
from `knee-xsfc`'s close note; that number is a SIX-seat shape, not these five, and measuring rather
than quoting it is the difference between an anchor and a fiction.

- **0.936-0.941** — the twin pool lands where the existing 37 pairs' slope says a 0.8706 offline pool
  should. `board_transfer` extends cleanly to the top of the record, the champion's 0.944 stops being
  an extrapolation, and every arm priced against that slope keeps its price.
- **0.942 or above** — the slope is STEEPER at the top than the fitted median, i.e. the board rewards
  the last offline point more than the record says. Every arm this project killed for a sub-0.0015
  offline delta was killed on an under-stated exchange rate, and the marginal-seat pricing that closed
  the ensemble axis has to be re-read at the champion's operating point.
- **0.930-0.935** — the slope FLATTENS at the top: offline gains stop transferring where it matters.
  That is the reading that says the offline holdout has saturated as a ruler, and the next lever must
  be chosen by mechanism rather than by holdout delta.
- **below 0.930** — the twin-to-shipped gap is the dominant term, larger than everything the combiner
  entries were reading as a combiner effect. `aucvote_v1` and `stackvote_v1` then have to be re-read
  as measurements of that gap, and no vote rule can be fitted on twins again.

COST: one slot. No card — every member already exists and is already on `comp/members.roster`.
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

    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - the kit is on sys.path only after boot
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same

    # The five fold-0 members whose pooled holdout is the offline number this entry exists to pair.
    # No foreign seat: raptor's holdout is 98.6% in-sample, and one in-sample seat would void the
    # pair the same way it voided `raptor_solo`'s.
    chosen = (
        "calib_span_s2026",
        "grade_ladder_s2026",
        "merged_blend_cf",
        "lr4e-5_ws_s2026",
        "through_dense_s2026",
    )
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage())).run()


if __name__ == "__main__":
    main()
