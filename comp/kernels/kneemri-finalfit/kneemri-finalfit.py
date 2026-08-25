"""The one axis that has ever cleared this project's bar, tested at the only step left on it.

WHAT IT SWAPS. `gradeswap`'s exact seven — the pinned 0.933 champion — with `band_dense_s2026` out and
`final_ws_e12` in. Count, kit (`cf6a99f`), vote, serving config and the other six are held identical.

WHAT `final_ws_e12` IS. `band_dense`'s flags copied verbatim, plus `--fold -1`: no study is held out, so
it trains on 4,406 rather than 3,497. Same 336 px, same `--slices 20`, same 0.10-0.90 band, same 4e-5,
same encoder, same twelve epochs. Training DATA VOLUME, 79% -> 100%, is the only design difference.

WHY THIS AXIS. Data volume is the ONLY axis that has ever cleared the 1.5% house bar here: `half_data`
scored 0.8344 against `band_dense`'s 0.8578, a -0.0234 gap at 3.8x the offline noise floor. The curve is
CONCAVE (each doubling buys about 0.48 of the last), so the 80->100% step is 0.32 of a doubling and
prices at about +0.0029 offline before any penalty.

TWO VARIABLES, NOT ONE, AND THE SECOND IS NOT REMOVABLE. `--ema` rides along, because with `--fold -1`
there is no best-epoch selection and the trainer keeps the LAST epoch — ema is the only variance control
the deliverable has. The read is "data step + ema" and a null cannot separate them. Named here rather
than discovered in the result.

THE OFFLINE RULER DOES NOT APPLY TO THIS MEMBER AT ALL. It has seen every holdout study, so it can never
be seat-scored, curve-scored or distance-scored, and it must never enter a pool that reports an offline
mean. `knee-7d1`'s 4.67x board-per-holdout compression cannot be used either: that converts an HONEST
offline delta into board points, and this member has none. So this pre-registration is written directly
on the board.

PRE-REGISTRATION, written before the run finished.

- 0.936 or above — the free data step transfers. Licenses refitting the other six at `--fold -1`
  (about 7 h of card) on a mechanism that needs no new idea and no external data. This is the only path
  still priced above the 0.933 -> 0.940 gap.
- 0.934-0.935 — directional, under the bar. Keep the member; do NOT license the seven-member refit.
- 0.933 — a tie. The data step does not survive the swap and the 7 h refit is retired before it is spent.
  That is the cost-tiered shape working: 93 min of card decides 7 h of it.
- 0.932 or below — suspect the SWAP, not the data. `band_dense` is the strongest single member (0.8578)
  and displacing it costs whatever it was carrying; the next test would ADD `final_ws_e12` as an eighth
  rather than swap it in.

No leakage concern for the submission: the board's test set is disjoint from the training corpus, so
training on 100% of train is exactly what a final fit is for.
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
        "calib_span_s2026",
        "grade_ladder_s2026",
        "band_dense_long_s2026",
        "base_dense_s2026",
        "lr4e-5_ws_s2026",
        "through_dense_s2026",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
