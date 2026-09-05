"""ceiling5 — the best blend this project can PRICE, shipped to test the ruler that prices it.

THIS IS AN INSTRUMENT. It is predicted to score BELOW the champion (0.944) and cannot improve board
position. It is spent on the ruler because every remaining arm decision now runs through a compression
that rests on four resolved pairs, and a wrong slope points the whole schedule at the wrong axis.

WHAT IT IS. Forward selection over the 195 members that recorded a holdout, scored on the corpus label
table under the vote rule this kernel serves (`python -m kneemri.kaggle.priceable_blend`). Five seats,
offline 0.8757 under `rank` and 0.8758 under `LogitAverage` — the same number, which is why serving the
rule every other entry serves does not change what was priced. Two further seats the search accepted are
dropped: `warm_calib_lowlr_s2026` is DECLINED on `comp/members.declines`, and `disagreed_ramp` bought
+0.0002. Together they moved the blend +0.0006 and the prediction not at all.

WHY IT CAN BE PREDICTED AT ALL, and it could not have been this morning. `EntryOffline` resolved holdout
records against `root.processed()` while every record lives under `root.models()`, so every entry that
survived the in-sample test was falsely declined `no-record`. With that fixed, `distinct14` (board 0.937)
carries an offline number, 0.8749, and is the first anchor the compression has ever had in the regime
where entries are actually shipped. Four of this entry's five members are seats in it.

THE PREDICTION. Board 0.937 + 0.82 x (0.8757 - 0.8749) = **0.938**, with the top-regime band
[+0.71, +0.93] that `board_transfer` calls DIRECTION ONLY.

PRE-REGISTERED, BEFORE THE NUMBER. Bands on the BOARD score:

  0.936-0.940  COMPRESSION VALIDATED in the regime that matters. Pricing an arm at roughly 1:1 offline
               to board is licensed, and the arithmetic that says the 0.953 target needs offline ~0.891
               — +0.015 over the best blend assemblable from every member this project owns — stands as
               the reason no selection, weighting or seating operation can reach it.
  0.941-0.944  THE COMPRESSION UNDER-PREDICTS. The priceable pool is better than its offline says, the
               `_ff` dose effect is smaller than the 0.007 knee-97ib measured, and holding out a split
               stops costing board position — which would reopen honest-fold arms as shippable.
  above 0.944  THE RULER INVERTS at the top. Every offline-based decision this project holds becomes
               suspect and the honest next move is to stop pricing arms offline at all.
  below 0.936  SELECTION OPTIMISM EXCEEDS THE GAIN. An offline-selected blend does not transfer, and
               greedy search over the holdout is not a tool this project may use to choose an entry.

NO FOREIGN SEAT, on purpose. The blend was priced without one, and a `RaptorSeat` would make the board
number a fact about a different ensemble than the one the offline number describes.

The full argument is `interpretations/kneemri/2026-09-05_priceable-ceiling.md`.
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
        "stack_increments_s2026",
        "labelwise_dense_s2026",
        "margin_seat_s2026",
        "finding_calib_dense_s2026",
        "blendhold_s2026",
    )
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget())).run()


if __name__ == "__main__":
    main()
