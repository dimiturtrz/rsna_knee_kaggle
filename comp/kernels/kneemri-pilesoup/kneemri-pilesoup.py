"""pilesoup — TEMPLATE, the repeatable defence: a new public best plus our two soups at 1/12 each, rank space.

The one increment in our entries with off-public evidence is the soup pair (offline +0.0015, P>0 1.00 on the replicas;
board s6r7solo 0.957 vs r7half 0.956), `interpretations/kneemri/2026-10-15_public_overfit_exposure.md`. When a public
notebook beats our best, this kernel runs that notebook's scoring source unmodified (mounted as `pile_best.py` from
`dimiturnt/pile-best-src`, its own inputs added to the metadata) and keeps 5/6 of its ranks. No weight is fitted to
the board. Before shipping: fill the source dataset, add the notebook's inputs, visible run, run_check, submit_check.
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

    import shutil  # noqa: PLC0415
    import time  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    from core.mounts import SERIES as MOUNTED_SERIES  # noqa: PLC0415 - the kit is on sys.path only after boot
    from core.mounts import Mounts  # noqa: PLC0415
    from kneemri.inference.budget import KERNEL_SECONDS, Budget  # noqa: PLC0415
    from kneemri.inference.ensemble import Serving  # noqa: PLC0415
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415
    from kneemri.kaggle.raptor_seat import RaptorSeat  # noqa: PLC0415

    working = Path("/kaggle/working")
    started = time.time()
    mounts = Mounts()
    script = mounts.one("pile_best.py", "the public best's scoring source") / "pile_best.py"
    finished = subprocess.run([sys.executable, "-u", str(script)], check=False, cwd=working)  # noqa: S603 - our mounted copy, fixed argv
    logging.getLogger(__name__).info("pile exited %d at %.0fs", finished.returncode, time.time() - started)
    if not (working / "submission.csv").is_file():
        message = f"the pile left no table (exit {finished.returncode}), so this run is not the entry it is named for"
        raise RuntimeError(message)
    shutil.move(working / "submission.csv", working / "pile_seat.csv")
    spent = time.time() - started

    serving = Serving(vote=LogitAverage(), budget=Budget(seconds=max(KERNEL_SECONDS - spent, 0.0) / 4.0))
    ours_mounts = Mounts(skip=MOUNTED_SERIES)
    KernelRun(names=("basinsoup_ff_k4_s2026",), overlap=False, serving=serving, mounts=ours_mounts).run()
    shutil.move(working / "submission.csv", working / "soup_ff.csv")
    KernelRun(names=("basinsoup_dmlb_k7_s2026",), overlap=False, serving=serving, mounts=ours_mounts).run()
    shutil.move(working / "submission.csv", working / "soup_dmlb.csv")

    frames = [
        pd.read_csv(path, dtype={"StudyInstanceUID": str})
        for path in (working / "soup_ff.csv", working / "soup_dmlb.csv", working / "pile_seat.csv")
    ]
    RaptorSeat.combined(frames[0], frames[1:], [1.0 / 12.0, 5.0 / 6.0]).to_csv(working / "submission.csv", index=False)


if __name__ == "__main__":
    main()
