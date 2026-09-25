"""fullseat — tripleseat_v2 with ONE seat refit on the WHOLE corpus: the DATA axis, not a reseat.

MECHANISM entry (knee-sexix). Serving is the champion's, untouched: `LogitAverage` over seven,
`overlap=False`, raptor at 1/3, kit `c3effa61`. One variable: `labelwise_ext_drop40_long_s2026` (fold 0,
trained on ~3500 studies, 0.9045 ledger75, the best honest solo owned) out, and its byte-identical recipe
refit at `--fold -1 --ema`, `labelwise_ext_drop40_long_ff` (all ~4400 studies, 24 of 24 epochs), in.
Same recipe, same seat, same count: the moved term is the ~fifth of the corpus the seat never saw, plus
the EMA the champion's other full-fit seats already carry.

Not a seat-identity move. Two single-seat swaps in a row (tripleseat -> doseseat, warmseat) read the
seat axis saturated at board resolution; this holds the seat's recipe and moves its DATA, the lever the
honest harness could never ask for because a `--fold -1` member has no holdout row.

UNPRICEABLE OFFLINE BY CONSTRUCTION: `--fold -1` holds nothing back. Its twin is the seat it replaces.
Candidates it was chosen over, priced on 896 A'-OOF rows, twin n=6, ledger75, 200 bootstraps
(`<scratchpad>/price.py`): raptor weight 1/3 -> 0 reads +0.0016 [-0.0003, +0.0038] against the only
honest foreign proxy (A', solo 0.8787, weaker than the shipped raptor) while `raptor_v4_swa` is in-sample
and reads monotone in weight, so no honest offline read of the weight exists; the board pair
honestfar/honestsix already says raptor at 1/3 carries +0.003. RankAverage -0.0003 [-0.0007, +0.0002],
MedianLogit +0.0002 [-0.0008, +0.0011]; the combiner closed on the board (stackvote_v1).

PRE-REGISTRATION, against tripleseat_v2's 0.946 and `BOARD_RESOLVABLE = 0.0015`:

- **0.948+** — full-corpus data moves the entry at ONE seat of seven. Re-pin; refit the other two drop40
  seats (`labelwise_depth24`, `labelwise_dense`) at `--fold -1 --ema` and ship all three.
- **0.947** — directional, one tick. Do not re-pin on it alone; the three-seat refit is the confirm.
- **0.946** — FLAT: one seat's extra fifth of data is under board resolution at a 2/3 share over seven.
  Does NOT close full-fit of the pool; a three-seat refit is the dose that could resolve it.
- **0.945** — one tick down: noise-level; the fold-0 member's diversity against the full-fit seats was
  worth a tick. Check `run_check` census first.
- **Below 0.945** — the full-fit refit hurts (EMA or fit-all overshoot on this recipe); stop full-fitting
  drop40 recipes and keep fold-0 bases.
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

    from kneemri.kaggle.seated_entry import SeatedEntry  # noqa: PLC0415 - the kit is on sys.path only after boot

    chosen = (
        "final_ws_cf",
        "calib_span_ff",
        "stack_increments_ff",
        "through_dense_cf",
        # The three drop40 bases live on the SECOND shelf (`dimiturnt/kneemri-members-2`).
        "labelwise_depth24_drop40_long_s2026",
        "labelwise_ext_drop40_long_ff",
        "labelwise_dense_drop40_long_s2026",
    )
    # `labelwise_ext_drop40_long_s2026` out, its `--fold -1 --ema` refit in: same recipe, whole corpus.
    SeatedEntry(
        names=chosen,
        # A quality-weighted share of two SYSTEMS, not `raptorseat_v1`'s 1/7 seat-count convention.
        weight=1.0 / 3.0,
    ).run()


if __name__ == "__main__":
    main()
