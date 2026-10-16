"""heliospf — heliosli's 10-15 public blend (credited 0.958), replicated with nothing of ours seated.

The control for `kneemri-s6r7pf`: the same producers minus our two soups. nartaa 384 + 224 (0.65 / 0.35 inside the
Raptor half), heliosli's R6 bundle `fe055406` (mount `rsna-knee-r6-frozen-fe055406-w1`; NOT the R7 `8c7b666e`
every s6r7 entry seats) under its own flip TTA at the manifest's own eval stride (as `blend-mine` runs it), and
the per-finding Raptor share: 0.80 Synovitis, 0.20 Lateral OA, 0.60 PF OA, 0.35 Baker's, 0.50 elsewhere, the bundle taking the rest. Fusion is linear in each table's percentile rank, as theirs is.
The per-finding table was very likely fitted on the PUBLIC board, so this is a FITTING-pick candidate only.
PRE-REG (resolution 0.001): **0.958** — the replica is faithful, and it is the baseline `s6r7pf` is read against;
**0.957** — a producer differs from theirs or the credit was a rounding, read s6r7pf against 0.957; **<= 0.956** —
the replica is broken, not a result; find the producer that differs before reading s6r7pf at all.
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

RAPTOR_SHARE = {"Synovitis": 0.80, "Lateral OA": 0.20, "PF OA": 0.60, "Baker's": 0.35}
"""heliosli's per-finding Raptor share of the (Raptor + R7) pair, 10-15 version of blend-mine."""
RAPTOR_SHARE_ELSEWHERE = 0.50


def main() -> None:
    boot()

    import shutil  # noqa: PLC0415
    import time  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    from core.mounts import Mounts  # noqa: PLC0415 - the kit is on sys.path only after boot
    from kneemri.kaggle.raptor_seat import RaptorSeat  # noqa: PLC0415

    tolerant_r7 = {
        "if rc != 0:": "if rc != 0 and not (pred / 'submission.csv').is_file():",
        "if manifest.get('flip_tta'):": "if True:",
        "BUNDLE_ID = '8c7b666e27dc0e8f'": "BUNDLE_ID = 'fe055406fad6a12f'",
        "C_SLUG = 'rsna-knee-compact-r7cnxfsss'": "C_SLUG = 'rsna-knee-r6-frozen-fe055406-w1'",
    }
    working = Path("/kaggle/working")
    started = time.time()
    mounts = Mounts()
    for scorer in ("nartaa_mirror.py", "nartaa224.py"):
        script = mounts.one(scorer, "nartaa's scoring source") / scorer
        subprocess.run([sys.executable, str(script)], check=True, cwd=working)  # noqa: S603 - our mounted copy, fixed argv
    shutil.move(working / "submission.csv", working / "nartaa_seat.csv")

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
    helios = Path("/tmp/helios_r7_tolerant.py")
    helios.write_text(source, encoding="utf-8")
    finished = subprocess.run([sys.executable, "-u", str(helios)], check=False, cwd=working, env=environment)  # noqa: S603 - our mounted copy, patched, fixed argv
    logging.getLogger(__name__).info("R7 exited %d at %.0fs", finished.returncode, time.time() - started)
    if not (working / "r7_sub.csv").is_file():
        message = f"R7 left no table (exit {finished.returncode}), so this run is not the entry it is named for"
        raise RuntimeError(message)

    bundle, *raptor = (
        pd.read_csv(working / name, dtype={"StudyInstanceUID": str})
        for name in ("r7_sub.csv", "nartaa_seat.csv", "raptor224_sub.csv")
    )
    blends = {
        share: RaptorSeat.combined(bundle, raptor, [share * 0.65, share * 0.35])
        for share in {RAPTOR_SHARE_ELSEWHERE, *RAPTOR_SHARE.values()}
    }
    blends[RAPTOR_SHARE_ELSEWHERE].assign(
        **{target: blends[share][target] for target, share in RAPTOR_SHARE.items()}
    ).to_csv(working / "submission.csv", index=False)


if __name__ == "__main__":
    main()
