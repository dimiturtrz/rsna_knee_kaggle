"""s6r7res384 — `kneemri-s6r7solo` with nartaa's 384 reader scored on its full 384 px view, not the 320 centre crop.

nartaa's source trains and scores `coatnet_rmlp_2_rw_384` on 320 px crops cut out of a native 384 px, 140 mm box
(`INPUT_RES = 320`, `INPUT_MODE = "crop"`), so the outer ~12 mm on every side of the box is never read at test time.
This scores the same checkpoint at `INPUT_RES = 384`: same pixel scale (crop, never resize), 1.44x the field of view,
which is where the collateral ligaments and the lateral meniscus sit. The rmlp relative-position bias is an MLP over
coordinates, so the weights load unchanged at 384. Everything else is s6r7solo: nartaa 1/2 (with its mirror TTA),
R7 1/3, soups 1/12 each, rank space.
PRE-REG (B = s6r7solo_v1 0.957, resolution 0.001, top-band residual sd 0.0013 @ledger75):
- **> B** — the clipped periphery carries signal the 320 crop misses; new pin; re-read the 224 reader at full view.
- **= B** — below board resolution; public RANK inside the tie is the reading.
- **< B** — train/test field-of-view mismatch costs more than the periphery adds; 320 stays.
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

    import gc  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    import time  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from core.mounts import SERIES as MOUNTED_SERIES  # noqa: PLC0415 - the kit is on sys.path only after boot
    from core.mounts import Mounts  # noqa: PLC0415
    from kneemri.inference.budget import KERNEL_SECONDS, Budget  # noqa: PLC0415
    from kneemri.inference.ensemble import Serving  # noqa: PLC0415
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415
    from kneemri.kaggle.raptor_seat import RaptorSeat  # noqa: PLC0415

    tolerant_r7 = {
        "if rc != 0:": "if rc != 0 and not (pred / 'submission.csv').is_file():",
        "str(manifest.get('eval_stride', 3))": "'2'",
    }
    heliosli_mounts = frozenset({"rsna-knee-compact-r7cnxfsss", "rsna-orthofoundation-fold0-epoch7-20261004"})
    working = Path("/kaggle/working")
    started = time.time()
    mounts = Mounts()
    top, rest = (
        (mounts.one("nartaa_mirror.py", "nartaa's scoring source") / "nartaa_mirror.py")
        .read_text(encoding="utf-8")
        .split("INPUT_RES = 320\n")
    )  # unpacking refuses a source holding the anchor zero or twice
    middle, bottom = rest.split("(94,3,320,320)")
    Path("/tmp/nartaa_full_view.py").write_text(
        top + "INPUT_RES = 384\n" + middle + "(94,3,384,384)" + bottom, encoding="utf-8"
    )
    subprocess.run([sys.executable, "/tmp/nartaa_full_view.py"], check=True, cwd=working)
    shutil.move(working / "submission.csv", working / "nartaa_seat.csv")
    spent = time.time() - started

    serving = Serving(vote=LogitAverage(), budget=Budget(seconds=max(KERNEL_SECONDS - spent, 0.0) / 4.0))
    ours_mounts = Mounts(skip=MOUNTED_SERIES | heliosli_mounts)
    for soup, table in (("basinsoup_ff_k4_s2026", "soup_ff.csv"), ("basinsoup_dmlb_k7_s2026", "soup_dmlb.csv")):
        KernelRun(names=(soup,), overlap=False, serving=serving, mounts=ours_mounts).run()
        shutil.move(working / "submission.csv", working / table)
    gc.collect()
    torch.cuda.empty_cache()

    environment = dict(os.environ, S6_T0=str(started + 0.3 * 3600.0))
    source = (mounts.one("helios_r7.py", "heliosli's R7 scoring source") / "helios_r7.py").read_text(encoding="utf-8")
    for anchor, replacement in tolerant_r7.items():
        if source.count(anchor) != 1:
            message = f"R7 source holds {source.count(anchor)} copies of {anchor!r}, expected one"
            raise ValueError(message)
        source = source.replace(anchor, replacement)
    table = "'ours_sub.csv'"
    if source.count(table) != len(("copy", "read", "write")):
        message = f"R7 source names its table {source.count(table)} times, expected three"
        raise ValueError(message)
    source = source.replace(table, "'r7_sub.csv'")
    Path("/tmp/helios_r7_tolerant.py").write_text(source, encoding="utf-8")
    finished = subprocess.run(
        [sys.executable, "-u", "/tmp/helios_r7_tolerant.py"], check=False, cwd=working, env=environment
    )
    logging.getLogger(__name__).info("R7 exited %d at %.0fs", finished.returncode, time.time() - started)
    if not (working / "r7_sub.csv").is_file():
        message = f"R7 left no table (exit {finished.returncode}), so this run is not the entry it is named for"
        raise RuntimeError(message)

    frames = [
        pd.read_csv(path, dtype={"StudyInstanceUID": str})
        for path in (
            working / "soup_ff.csv",
            working / "soup_dmlb.csv",
            working / "r7_sub.csv",
            working / "nartaa_seat.csv",
        )
    ]
    RaptorSeat.combined(frames[0], frames[1:], [1.0 / 12.0, 1.0 / 3.0, 1.0 / 2.0]).to_csv(
        working / "submission.csv", index=False
    )


if __name__ == "__main__":
    main()
