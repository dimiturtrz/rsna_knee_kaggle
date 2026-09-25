"""warmseat — tripleseat_v2 plus one more seat: `warm_calib_lowlr_drop40_long_s2026` as the eighth.

ADDITION, not a swap: every tripleseat_v2 member stays, serving untouched (`LogitAverage`, `overlap=False`,
raptor at 1/3). One variable: n 7 -> 8, the new seat on the second shelf (uploaded for this entry).

Priced on ledger75, 909 fold-0 rows, fold-0 twin proxy (`<scratchpad>/warm.py`, dose.py with the new
member; twin n=6, `final_ws_cf` has no fold-0 twin): ADD reads **+0.0007** [+0.0001, +0.0012]
(200 bootstraps). The member alone: 0.8994 ledger75 (full 24-epoch run, peak epoch 8), and
`seat_value --against cleanseat_v1 --ruler ledger75` +0.0018 [+0.0011, +0.0026].
tripleseat's proxy said +0.0084 and +0.002 landed (about a quarter), so the expected board move is
~+0.0002: under the board's 0.001 resolution. The CI clears 0 (by +0.0001), so by rule this is an
IMPROVEMENT ATTEMPT, not an instrument — but a sub-resolution one: it reads whether ADDITION (n=8) moves
where a removal-swap (doseseat_v1) is the other half of the same question.

PRE-REGISTRATION, against tripleseat_v2's 0.946 and `BOARD_RESOLVABLE = 0.0015`:

- **0.948+** — addition carries more than the proxy sees. Re-pin; add the next drop40 base (n=9).
- **0.947** — directional, one tick. Do not re-pin on it alone.
- **0.946** — FLAT: an eighth seat at this strength is saturated. Check `run_check`'s census FIRST
  (8 of 8). Pin stays.
- **0.945 or below** — dilution: the eighth seat costs more than it adds; stop growing n and spend seats
  by swap only.
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
        # The trio lives on the SECOND shelf (`dimiturnt/kneemri-members-2`) — the one dataset this
        # kernel mounts that the champion does not. All three ship: the +0.0040 seat floor is cleared
        # by the trio and by none of them alone.
        "labelwise_depth24_drop40_long_s2026",
        "labelwise_ext_drop40_long_s2026",
        "labelwise_dense_drop40_long_s2026",
        # The eighth seat, also second shelf: full run, 0.8994 ledger75 (knee-rkegi).
        "warm_calib_lowlr_drop40_long_s2026",
    )
    # `grade_ladder_cf` and `lr4e-5_ws_cf` are dropped on a priced reading, not trimmed to fit — see
    # knee-me2nv. The exposure worth carrying here: the twins used to price it are NOT these seats
    # (both are `--fold -1 --ema` refits, stronger than their 0.8868 / 0.8840 twins), so the entry
    # reads two things at once — whether it beats 0.944, and whether twin-proxy seating transfers.
    SeatedEntry(
        names=chosen,
        # A quality-weighted share of two SYSTEMS, not `raptorseat_v1`'s 1/7 seat-count convention.
        weight=1.0 / 3.0,
    ).run()


if __name__ == "__main__":
    main()
