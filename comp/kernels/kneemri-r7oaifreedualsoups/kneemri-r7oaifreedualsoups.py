"""r7oaifreedualsoups — OAI-FREE HEDGE. `kneemri-r7oaifreedual` (0.956) with the seven swapped for our dmlb soup.

R7 1/4 + R6 1/4 + basinsoup_ff_k4 1/4 + basinsoup_dmlb_k7 1/4, rank space; same mounts, no nartaa dataset, so still
OAI-free by its mounts. The reader-share axis is closed (peak at 1/2); this is the one untested axis on the OAI-free
line, the composition of OUR half, set to the pair the 0.958 pf line seats. Matters more since the host declined
to rule on OAI-trained weights (747455, 10-19), which leaves the OAI-free pick as the hedge.
PRE-REGISTRATION (vs r7oaifreedual_v1 0.956, resolution 0.001):
- **>= 0.957** — the soup pair beats seven beside the readers; this becomes the OAI-free pick.
- **= 0.956** — interchangeable; r7oaifreedual stays (reached first).
- **<= 0.955** — seven carries more than the dmlb soup off the nartaa line; r7oaifreedual stays.
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
    tolerant_r6 = {
        "if rc != 0:": "if rc != 0 and not (pred / 'submission.csv').is_file():",
        "if manifest.get('flip_tta'):": "if True:",
        "BUNDLE_ID = '8c7b666e27dc0e8f'": "BUNDLE_ID = 'fe055406fad6a12f'",
        "C_SLUG = 'rsna-knee-compact-r7cnxfsss'": "C_SLUG = 'rsna-knee-r6-frozen-fe055406-w1'",
    }
    heliosli_mounts = frozenset(
        {"rsna-knee-compact-r7cnxfsss", "rsna-knee-r6-frozen-fe055406-w1", "rsna-orthofoundation-fold0-epoch7-20261004"}
    )
    working = Path("/kaggle/working")
    started = time.time()
    mounts = Mounts()

    ours_seconds = KERNEL_SECONDS / 4.0
    ours_mounts = Mounts(skip=MOUNTED_SERIES | heliosli_mounts)
    KernelRun(
        names=("basinsoup_ff_k4_s2026",),
        overlap=False,
        serving=Serving(vote=LogitAverage(), budget=Budget(seconds=ours_seconds)),
        mounts=ours_mounts,
    ).run()
    shutil.move(working / "submission.csv", working / "soup_ff.csv")
    KernelRun(
        names=("basinsoup_dmlb_k7_s2026",),
        overlap=False,
        serving=Serving(vote=LogitAverage(), budget=Budget(seconds=ours_seconds)),
        mounts=ours_mounts,
    ).run()
    shutil.move(working / "submission.csv", working / "soup_dmlb.csv")
    logging.getLogger(__name__).info("soups done at %.0fs", time.time() - started)
    gc.collect()
    torch.cuda.empty_cache()

    environment = dict(os.environ, S6_T0=str(started + 0.3 * 3600.0))
    pristine = (mounts.one("helios_r7.py", "heliosli's R7 scoring source") / "helios_r7.py").read_text(encoding="utf-8")
    for patches, table in ((tolerant_r7, "r7_sub.csv"), (tolerant_r6, "r6_sub.csv")):
        source = pristine
        for anchor, replacement in patches.items():
            if source.count(anchor) != 1:
                message = f"R7 source holds {source.count(anchor)} copies of {anchor!r}, expected one"
                raise ValueError(message)
            source = source.replace(anchor, replacement)
        if source.count("'ours_sub.csv'") != len(("copy", "read", "write")):
            message = "R7 source does not name its table exactly three times"
            raise ValueError(message)
        source = source.replace("'ours_sub.csv'", f"'{table}'")
        (helios := Path("/tmp") / f"helios_{table}.py").write_text(source, encoding="utf-8")
        finished = subprocess.run([sys.executable, "-u", str(helios)], check=False, cwd=working, env=environment)  # noqa: S603 - our mounted copy, patched, fixed argv
        logging.getLogger(__name__).info("%s exited %d at %.0fs", table, finished.returncode, time.time() - started)
        if not (working / table).is_file():
            message = f"no {table} (exit {finished.returncode}), so this run is not the entry it is named for"
            raise RuntimeError(message)

    frames = [
        pd.read_csv(path, dtype={"StudyInstanceUID": str})
        for path in (working / "soup_ff.csv", working / "soup_dmlb.csv", working / "r7_sub.csv", working / "r6_sub.csv")
    ]
    RaptorSeat.combined(frames[0], frames[1:], [1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0]).to_csv(
        working / "submission.csv", index=False
    )


if __name__ == "__main__":
    main()
