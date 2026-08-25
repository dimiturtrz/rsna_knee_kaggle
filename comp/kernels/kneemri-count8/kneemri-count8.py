"""EIGHT members at the SAME all-data dose — the count axis, isolated for the first time.

**WHAT IS ACTUALLY UNMEASURED HERE.** Four entries have now read 0.936-0.937 (`allfit`, `lastseat`,
`augseat`, and the champion) and every one of them is the same shape: SIX `--fold -1` members plus ONE
fold-0 seat. Three of the four exchanged that seat — on identity, on solo quality, on regularisation —
and moved at most a tick. The seat is closed as a variable. What those entries HOLD FIXED, and what no
entry has ever varied, is the COUNT.

**WHY `peak8` DOES NOT ALREADY ANSWER IT.** `peak8` mounted eight members and scored 0.932, five ticks
under `allfit`. It is not a count reading: all eight of its seats are fold-0, an all-data dose of 0/8
against `allfit`'s 6/7, on the one axis this board has ever measured worth +0.003. Count and dose move
together there and the -0.005 is the dose gap's own size with the sign it already had. Reading `peak8`
as "eight is worse than seven" prices the wrong denominator.

**THE ONE VARIABLE IS THE COUNT.** This entry is `allfit` with one member ADDED, not exchanged. The same
six dark members, the same seventh seat (`band_dense_long_s2026`), plus `small_batch6_s2026` as an
eighth. Dose stays at six dark. Vote stays `RankAverage`, overlap stays False, the kit is unchanged.
Count moves 7 -> 8 and nothing else moves.

**WHY THESE TWO AS THE FOLD-0 PAIR.** Both have already held that seat on this board: `band_dense_long_s2026`
in `allfit` at 0.937 and `small_batch6_s2026` in `lastseat` at 0.937. Neither is a new quantity. The
three flat exchanges are the control for their identity, so whatever this entry moves cannot be
attributed to which fold-0 member is present — both of them separately produced 0.937 in this exact
ensemble.

**WHAT COSTS WHAT.** One slot, ZERO card — every member is already trained and packaged. P(this clears
the owner's 1.5% bar) is about 0.02; nothing on this board has moved that far. P(it clears the 0.0015
board-resolvable) is about 0.35. What it buys is the last structural variable at the top of this board
that has never been moved, now that roster search (`greedy7`, `peak8`), the seventh seat as an identity
(`lastseat`), and the seventh seat as a type (`augseat`) are all retired.

**NO OFFLINE NUMBER IS REPORTED AND NONE CAN BE.** Six members have seen the entire 909-study holdout,
so this entry has no offline score of any kind and enters no pair in `board_transfer`.

PRE-REGISTRATION, written before the number, against `allfit` 0.937 and `lastseat` 0.937 — the same
ensemble at count 7, measured twice.

- **0.940 or above** — count is a live lever and the pool has been under-mounted the whole time. It
  licenses a count ladder rather than a roster search: the next slot mounts nine, and the card goes to
  producing members rather than to better members.
- **0.938-0.939** — directional, below the owner's bar, and consistent with a real but small count
  effect. Keep it and mount nine on the next slot to see whether it accumulates; do NOT call it a
  finding on one tick.
- **0.935-0.937** — flat. Count is closed alongside the seat, and the honest reading is that this
  ensemble is saturated against every variable reachable by rearranging members that already exist.
  That retires the entire mount-side axis and sends the remaining card to members of a kind the pool
  does not contain. This is the band four prior entries predict.
- **0.934 or below** — an eighth member COSTS, which would mean `peak8`'s -0.005 was partly count after
  all and not purely dose. Suspect the harness FIRST: confirm the kernel log names eight members and
  that all eight loaded, before reading it as a result.

The bands are one board tick wide against a board that has returned 0.936-0.937 four times. That is
deliberate: this entry is not trying to clear the 1.5% bar alone. It converts the last open structural
axis into a measured one, in either direction, for a slot that cannot be banked.

bead: knee-rndp
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
        "band_dense_long_s2026",
        "base_dense_ff",
        "lr4e-5_ws_ff",
        "through_dense_ff",
        "small_batch6_s2026",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
