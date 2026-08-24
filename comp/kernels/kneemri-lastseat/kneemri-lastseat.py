"""One seat exchanged on the champion — the first REPEAT this project has ever bought at the top.

THIS IS AN INSTRUMENT AND IT IS NOT SOLD AS AN IMPROVEMENT. Expected board movement is about zero. It is
worth a slot because of what it calibrates: since `allfit`, the board is the ONLY ruler this project has
on its own champion pool, and nobody has ever measured what that ruler's noise is at 0.937.

**WHAT COSTS WHAT.** One slot of three remaining today, ZERO card — every member already exists and is
already packaged, and the card is busy with `ema_dense` (knee-56w5) meanwhile. P(this clears the owner's
1.5% bar) is about 0.00. The return is a repeat measurement and a seat-identity reading, both of which
every future board-only decision is priced against.

**THE SINGLE VARIABLE.** `kneemri-allfit` (0.937, champion) is six `--fold -1` members plus the one
fold-0 seat `band_dense_long_s2026` (0.8587 solo). This is the same six plus `small_batch6_s2026` (0.8598
solo, the best single member this project has, and the roster's best recorded EIGHTH seat at +0.0005).
Count (7), vote (`RankAverage`), kit and serving config are held fixed, and the all-data DOSE is held
fixed at six seats. Only which fold-0 member occupies the seventh seat moves.

That is the point. `allfit` moved the board by +0.003 and the whole reading attributed it to the dose. The
dose is held here, so if the number moves at all, part of what was read as dose is seat identity.

**NO OFFLINE NUMBER IS REPORTED AND NONE CAN BE.** Six members have seen the entire 909-study holdout.

PRE-REGISTRATION, written before the number, against `allfit`'s 0.937.

- **0.940 or above** — the seventh seat was UNDERPERFORMING and seat identity matters at the all-data
  operating point. This refutes *"the seven-member vote is flat to single-member exchange"* — which has
  held across four entries — specifically in the all-data pool, and it says the remaining fold-0 seat is
  worth searching rather than assumed settled. New champion, and the next slot searches that seat.
- **0.936-0.939** — THE REPEAT HOLDS, and this is the expected outcome. Two entries differing by one seat
  read the same band, so board scatter at this level is at most one tick across a single-seat exchange,
  `allfit`'s +0.003 replicates as a dose effect rather than a seat accident, and the flat-to-exchange
  clause extends to the all-data pool. Every board-only decision from here inherits that error bar. Note
  the honest limit: two entries reading the same band bound the scatter, they do not measure it.
- **0.933-0.935** — the exchange COST about what the dose bought, which is the reading that hurts most.
  `small_batch6` is the better member offline (+0.0011 solo) and the worse seat here, so solo score does
  not predict seat value in the all-data pool — a thing already true in the fold-0 pool (knee ledger:
  "seat value in a four-member vote and solo score are ordered oppositely") and never checked in this one.
  It would also mean part of `allfit`'s +0.003 was `band_dense_long_s2026` specifically, not the dose, and
  the dose table in `interpretations/kneemri/2026-08-24_the_all_data_dose_response.md` needs a caveat.
- **0.932 or below** — a drop of 0.005+ from ONE seat exchange, which nothing in 15 board entries has ever
  produced. That would say the all-data pool is FRAGILE where the fold-0 pool is robust: six members that
  all saw the same 4,406 studies agree with each other, so the single dissenting seat is carrying more of
  the ensemble than its share, and replacing it is not a swap but a load-bearing change. Consequence: the
  champion is a narrower structure than it looks, and the next entry re-seats MORE fold-0 members rather
  than fewer.

Three of the four bands change something written down. That is the test for whether an instrument earns a
slot, and it is the reason this one is being spent while the card answers a different question.
"""

import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# >>> KERNEL-BOOTSTRAP
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


# <<< KERNEL-BOOTSTRAP


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    kit = find_kit()
    install_wheels(kit)
    sys.path.insert(0, str(kit))

    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - kit is on sys.path only after install_wheels

    chosen = (
        "final_ws_e12",
        "calib_span_ff",
        "grade_ladder_ff",
        "small_batch6_s2026",
        "base_dense_ff",
        "lr4e-5_ws_ff",
        "through_dense_ff",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
