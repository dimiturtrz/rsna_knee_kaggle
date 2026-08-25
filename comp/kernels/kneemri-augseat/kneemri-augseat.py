"""The seventh seat filled by a REGULARISED member, against a same-type control measured hours earlier.

THIS IS THE TYPE-OF-DISTANCE TEST, and it is only worth a slot because `lastseat` ran first. That entry
held everything and exchanged the one fold-0 seat for a member of the SAME kind — `band_dense_long_s2026`
out, `small_batch6_s2026` (the best solo member this project has, +0.0011) in — and read **0.937, the
champion's own number to the tick**. So at seven members a same-type exchange moves nothing, even when it
raises solo quality. That is the control this entry needed and never had: any movement here cannot be
seat scatter and cannot be solo quality, because both were just measured at zero.

**THE ONE VARIABLE IS THE KIND OF DISTANCE THE SEVENTH MEMBER CARRIES.** `aug_long_s2026` is the
long-augmentation schedule — the REGULARISATION axis, the only axis in the roster whose mechanism is to
trade in-distribution fit for robustness to shift. Everything else is held: same six `--fold -1` members,
count 7, `RankAverage`, overlap False, same kit.

**WHY A MEMBER ITS OWN ARM KILLED.** `aug_long` read **-0.0076 offline**, outside the 0.0062 noise floor,
and was filed as a kill. It has never been mounted, not once. The reason to mount it now is that the ruler
which killed it is structurally unable to price it: the 909-study holdout is drawn from the SAME
distribution as the training side, so a mechanism whose entire purpose is to give up in-distribution fit
reads flat or negative on it BY CONSTRUCTION. That ruler is separately measured to under-price DIVERSITY
by 2.4x, and the ledger holds three members whose offline number said no and whose board said yes —
`calib_span` (-0.0020), `base_dense` (-0.0066, at 2.5x the card cost), `through_dense` (flat). Its solo
0.8511 is level with `base_dense`'s 0.8512, and `base_dense` carried the 0.931 entry.

**WHAT COSTS WHAT.** One slot, ZERO card — the member exists, is packaged, and the card is training
`distill_pairs` meanwhile. P(this clears the owner's 1.5% bar) is about 0.00; nothing on this board has
ever moved that far. What it buys is a reading on the only axis at the top of this board that still has a
live mechanism, now that roster search (`greedy7`) and the seventh seat as an identity (`lastseat`) are
both retired.

**NO OFFLINE NUMBER IS REPORTED AND NONE CAN BE.** Six members have seen the entire 909-study holdout.

PRE-REGISTRATION, written before the number, against `allfit` 0.937 and `lastseat` 0.937.

- **0.940 or above** — the regularisation axis buys a seat the ruler cannot see, on a member the ruler
  scored 0.0076 BELOW the one it replaces. Against the same-type control at 0.937 this is attributable to
  the kind of distance and to nothing else. It licenses re-reading every arm killed on a small negative
  offline delta, which is a large backlog, and the next slot searches the regularisation axis rather than
  the pool.
- **0.936-0.939** — neutral, and the honest reading is that TYPE of distance does not carry at seven
  members either. Note what it does NOT settle: a member 0.0076 worse offline holding the board flat is
  itself evidence the ruler over-prices in-distribution fit — the blindness argument survives a tie here,
  the seat argument does not.
- **0.933-0.935** — the swap costs about what the all-data dose bought. The offline number was right about
  this member specifically, and the blindness argument does not extend to the regularisation axis. This is
  the band that would retire knee-avzz's use as a licence to mount killed members.
- **0.932 or below** — a drop of 0.005+ from one seat exchange, which nothing in 16 board entries has
  produced. It would say the seventh seat is load-bearing in a direction `lastseat` could not see: six
  all-data members agreeing with each other, and the single dissenting seat carrying more than its share,
  so a WEAKER dissenter breaks it where an equal one does not. The next entry would re-seat more fold-0
  members, not fewer.

Three of the four bands change something written down, which is the test for whether an instrument earns a
slot.
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
        "final_ws_e12",
        "calib_span_ff",
        "grade_ladder_ff",
        "aug_long_s2026",
        "base_dense_ff",
        "lr4e-5_ws_ff",
        "through_dense_ff",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
