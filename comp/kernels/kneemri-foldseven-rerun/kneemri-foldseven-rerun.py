"""foldseven-rerun — `foldseven_v1`'s seats, byte-identical, submitted a second time. An INSTRUMENT.

It is the CONTROL for `kneemri-eightseat`, and it exists because without it that entry's delta cannot
be read. `eightseat` changes one seat worth +0.0005 offline and asks what the board does; but any
board move it shows is confounded with whatever this pipeline's run-to-run scatter is on Kaggle's
hardware, and NOTHING in this repo has ever measured that. Every board number we hold is a single
draw. The compression constant, the 0.0020 blur, the six-point pricetop/foldseven gap and every
ensemble comparison in `comp/board.jsonl` all assume a member set maps to ONE score.

WHAT IT MEASURES. The same members, the same vote, the same kit (`c3effa61`), the same mounts, the
same code. Anything other than 0.944 is pipeline nondeterminism -- T4 kernel selection, cudnn
autotune, non-associative float reduction over a different batch order -- and whatever it is, it is
the error bar every other row in that file is missing.

PRE-REGISTRATION:

- **exactly 0.944** -- the pipeline is deterministic end to end. `eightseat`'s delta is then
  attributable to the seat alone, and the pricetop/foldseven six-point gap is about seats too.
- **0.943 or 0.945** -- one board point of pipeline scatter. Tolerable, but it means the board's
  three-decimal resolution is at the edge of our own reproducibility.
- **0.941 or wider** -- run-to-run scatter is a large fraction of the six-point gap that motivated
  this pair of entries, and the honest reading is that single-draw board numbers cannot separate
  entries in this band at all. That would retire a large amount of believed structure, including
  several closures priced on differences of 0.002-0.004.

This is a DELIBERATE re-ship of a set the board already holds: `submit_check` refuses set equality by
design, so it goes through `--anyway`. The reason is the whole point of the entry and is repeated in
the submission message.

Check `run_check`'s census FIRST: 7 of 7 loaded or this reads nothing.
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

    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - the kit is on sys.path only after boot
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same

    chosen = (
        "calib_dense_hold_s2026",
        "calib_span_s2026",
        "stack_increments_s2026",
        "through_dense_s2026",
        "labelwise_depth24_drop40_long_s2026",
        "labelwise_ext_drop40_long_s2026",
        "labelwise_dense_drop40_long_s2026",
    )
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage())).run()


if __name__ == "__main__":
    main()
