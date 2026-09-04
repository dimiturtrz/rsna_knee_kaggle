"""Take `--calibrated-targets` OUT of `fiveship_v1` and leave everything else where it is.

THE CONTRADICTION THIS RESOLVES. Two board pairs measure "fold-0 twins -> full-fit members" at four
seats and disagree by five times the single-pair floor:

    foldfour_v1  0.936  ->  fitfour_v1   0.937   +0.001
    twinfive_v1  0.935  ->  fiveship_v1  0.941   +0.006

They are not the same operation, and the kernels say so where the board rows do not. `fitfour` seats
four `_ff` members: `--fold -1 --ema`, refit and nothing else. `fiveship` seats ONE `_ff` and THREE
`_cf`, and `_cf` is the `_ff` flags PLUS `--calibrated-targets` -- a change to the SUPERVISION, not to
the split. So the clean refit term is +0.001 and the extra +0.005 is either calibration or noise, and
which one it is decides whether roughly thirty hours of fleet-refit card is licensed at all.

WHAT THIS ENTRY IS. `fiveship_v1`'s exact seating with the three `_cf` seats replaced by their `_ff`
twins, which exist on disk and are already packaged:

    grade_ladder_cf -> grade_ladder_ff
    lr4e-5_ws_cf    -> lr4e-5_ws_ff
    through_dense_cf -> through_dense_ff

`calib_span_ff` is unchanged because it is already `_ff`. `merged_blend_cf` is unchanged because it has
no `_ff` twin on disk -- it is the fold-0 member of that group (roster line 387) and it sits in BOTH
halves of the twinfive/fiveship pair, so holding it fixed here keeps this entry comparable to both.

NOT A SEATING OPERATION. The member axis is closed on seven board flats and this does not touch it: the
seat COUNT is five in both entries, the members are the same five runs, and nothing was ranked, swapped
for a different run, reweighted or re-lensed. The one variable is a training flag.

PRE-REGISTERED, on the DELTA against fiveship_v1's 0.941.

  0.934 to 0.937 -- calibration carries essentially the whole +0.006. The refit constant is +0.001, the
    fleet-refit step is NOT licensed, and `--calibrated-targets` becomes the lever to price across the
    fleet -- which is a training flag on runs we already know how to make, not thirty hours of refit.
    This is the outcome the instrument is most valuable for.
  0.938 to 0.940 -- the two split. Both terms are real and both are at or under the floor individually;
    neither is plannable alone and the pair has to be re-read at more seats before either is spent on.
  0.941 to 0.942 -- calibration is nothing here. The +0.006 and the +0.001 are then two readings of the
    same refit operation that disagree by 0.005, which makes the disagreement the finding and puts the
    single-pair floor, not the constant, at the centre of the next reading.
  above 0.942 -- removing calibration IMPROVES the entry. Then `_cf` is a liability at full fit, and the
    champion, which ships five `_cf` seats, is carrying it.

COST: one slot. No card -- every member exists, is on `comp/members.roster` and is already mounted.
INSTRUMENT, not an improvement: nothing here is expected to beat the champion's 0.944.
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

    # fiveship_v1's seating with --calibrated-targets removed from the three seats that have an _ff
    # twin. merged_blend_cf stays because it has none, and calib_span_ff was never _cf.
    chosen = (
        "calib_span_ff",
        "grade_ladder_ff",
        "merged_blend_cf",
        "lr4e-5_ws_ff",
        "through_dense_ff",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
