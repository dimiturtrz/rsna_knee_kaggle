"""fullseat2 — tripleseat_v2 with TWO seats refit on the WHOLE corpus: the DATA axis at a second dose.

MECHANISM entry (knee-sexix). Serving is the champion's, untouched: `LogitAverage` over seven,
`overlap=False`, raptor at 1/3, kit `c3effa61`. Two variables, both the same move:
`labelwise_ext_drop40_long_s2026` -> `labelwise_ext_drop40_long_ff` (as fullseat) AND
`labelwise_depth24_drop40_long_s2026` (fold 0, 0.9053 ledger75) -> `labelwise_depth24_drop40_long_ff`
(its byte-identical recipe at `--fold -1 --ema`, all ~4400 studies, 24 of 24 epochs 2026-09-15). Same
recipes, same seats, same count: the moved term is the fifth of the corpus two seats never saw, plus EMA.

Issued while fullseat_v1 (ref 56246750, the one-seat dose) is still PENDING, so the two readings form a
dose series: 0 seats (0.946), 1 seat, 2 seats.

UNPRICEABLE OFFLINE BY CONSTRUCTION: `--fold -1` holds nothing back. Each ff seat's twin is the seat it
replaces. Any number quoted for this entry is the BOARD's.

PRE-REGISTRATION, against tripleseat_v2's 0.946, fullseat_v1 (pending) and `BOARD_RESOLVABLE = 0.0015`:

- **0.948+** — the data axis is real at two seats. Re-pin; refit `labelwise_dense_drop40_long` at
  `--fold -1 --ema` and ship all three drop40 seats full-fit.
- **0.947** — directional, one tick. Do not re-pin on it alone; read against fullseat_v1 for the dose.
- **0.946** — FLAT: two seats' extra fifth of data is under board resolution over seven.
- **0.945 or below** — full-fit hurts at this dose (EMA or fit-all overshoot on drop40 recipes); stop
  full-fitting and keep fold-0 bases.
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
        "labelwise_depth24_drop40_long_ff",
        "labelwise_ext_drop40_long_ff",
        "labelwise_dense_drop40_long_s2026",
    )
    # The ext and depth24 fold-0 seats out, their `--fold -1 --ema` refits in: same recipes, whole corpus.
    SeatedEntry(
        names=chosen,
        # A quality-weighted share of two SYSTEMS, not `raptorseat_v1`'s 1/7 seat-count convention.
        weight=1.0 / 3.0,
    ).run()


if __name__ == "__main__":
    main()
