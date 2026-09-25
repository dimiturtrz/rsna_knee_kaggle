"""doseseat — tripleseat_v2 with one more `drop40_long` seat: the DOSE of the mechanism that landed.

INSTRUMENT, not a mechanism bet: its own proxy predicts a move under the board's 0.001 resolution.
Serving is the champion's, untouched: `LogitAverage` over seven, `overlap=False`, raptor at 1/3.
One variable: `through_dense_cf` out, `labelwise_drop_drop40_long_s2026` in (second shelf, uploaded).
Count held at seven, so drop40 seats go 3 -> 4 and nothing else moves.

Priced two ways, ledger75, 909 fold-0 rows, neither able to see the dark pin (four `--fold -1` seats):
- fold-0 twin proxy (`<scratchpad>/dose.py`, twin n=6 — `final_ws_cf` has no fold-0 twin, so it is
  neither priced nor moved): the swap reads **+0.0011** [+0.0003, +0.0021] (200 bootstraps), best of all
  twelve +1 swaps; every +2 dose reads +0.0000..+0.0007 with intervals spanning zero, so +1 it is.
- `seat_value --against cleanseat_v1 --ruler ledger75`: **+0.0021** [+0.0014, +0.0029] added,
  +0.0027 for `blendhold`; under the +0.0040 floor this ruler has ever been shown to resolve.
tripleseat's proxy said +0.0084 and +0.002 landed (about a quarter), so the expected board move here is
~+0.0003: null. What the slot buys is the dose-response reading itself.

PRE-REGISTRATION, against tripleseat_v2's 0.946 and `BOARD_RESOLVABLE = 0.0015`:

- **0.948+** — the dose is NOT saturated at three and the proxy under-reads it. Re-pin; the next slot
  doses +2 (`calib_span_ff` or `stack_increments_ff` out for a fifth drop40 base).
- **0.947** — directional, one tick. Do not re-pin on it alone; dose +2 next to see if the sign holds.
- **0.9455–0.9465 (reads 0.946)** — FLAT: the drop40 dose is saturated at three, or `through_dense_cf`'s
  dark refit carries what its twin cannot show. Check `run_check`'s census FIRST. Pin stays.
- **0.945** — one tick down: the dark seat was worth more than its twin; stop dosing by removal and
  dose by ADDITION (n=8) instead.
- **Below 0.945** — `through_dense_cf` is load-bearing, and the twin proxy's SIGN is refuted for dark
  seats — every removal priced through a twin is suspect from here.
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
        # The four drop40 bases live on the SECOND shelf (`dimiturnt/kneemri-members-2`).
        "labelwise_depth24_drop40_long_s2026",
        "labelwise_ext_drop40_long_s2026",
        "labelwise_dense_drop40_long_s2026",
        "labelwise_drop_drop40_long_s2026",
    )
    # `through_dense_cf` out, priced on its fold-0 twin `through_dense_s2026`. The twin is NOT the seat
    # (the seat is a `--fold -1 --calibrated-targets` refit), so a flat or a drop reads on that exposure.
    SeatedEntry(
        names=chosen,
        # A quality-weighted share of two SYSTEMS, not `raptorseat_v1`'s 1/7 seat-count convention.
        weight=1.0 / 3.0,
    ).run()


if __name__ == "__main__":
    main()
