"""s6r7pftrioown4x — `kneemri-s6r7pftrioown` with the goodpjw2008 reader's end-blend weights at four times heliosli's.

The trio stack (soups and mirrorthird's seven at 1/12 each, readers 3/4) with the end leg at 0.012-0.14 per finding.
Built for the 10-21 rule's branch A: shipped only if pfflatown4x 57455911 moves the team row. Priced on 562 honest
rows with an R7 stand-in seat: +0.0002 at 4x over none, P 0.72, the dose curve's peak. If the leg runs late or fails,
pfheavytrio's table is kept (knee-rrqxc).
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

READER_SHARE = {
    "Baker's": 0.14,
    "Contusion": 0.12,
    "Medial OA": 0.1,
    "ACL": 0.1,
    "Medial Meniscus": 0.08,
    "MCL": 0.08,
    "Lateral Meniscus": 0.032,
    "Fracture": 0.032,
    "Lateral OA": 0.032,
    "PF OA": 0.032,
    "Effusion": 0.016,
    "Synovitis": 0.012,
}
"""heliosli's OWN_W_PER_TARGET (v11), times four: the reader's weight in the end blend, per finding."""
READER_DEADLINE = 8.2 * 3600.0
"""heliosli's cut-off: past this many seconds the leg is skipped and the entry's own table is kept."""
READER_WHEELS = ("pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "pyjpegls", "python-gdcm")

RAPTOR_SHARE = 0.50
"""pfheavyflat's reader split: half of the readers' 3/4 to nartaa (384 / 224 at 0.65 / 0.35), half to R7."""


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
        "if rc != 0:": ("if rc != 0 and not (pred / 'submission.csv').is_file():", 1),
        "if manifest.get('flip_tta'):": ("if True:", 1),
        "BUNDLE_ID = '8c7b666e27dc0e8f'": ("BUNDLE_ID = 'fe055406fad6a12f'", 1),
        "C_SLUG = 'rsna-knee-compact-r7cnxfsss'": ("C_SLUG = 'rsna-knee-r6-frozen-fe055406-w1'", 1),
        "'ours_sub.csv'": ("'r7_sub.csv'", len(("copy", "read", "write"))),
    }
    heliosli_mounts = frozenset({"rsna-knee-r6-frozen-fe055406-w1", "rsna-orthofoundation-fold0-epoch7-20261004"})
    working = Path("/kaggle/working")
    started = time.time()
    mounts = Mounts()
    for scorer in ("nartaa_mirror.py", "nartaa224.py"):
        subprocess.run(  # noqa: S603 - our mounted copy, fixed argv
            [sys.executable, str(mounts.one(scorer, "nartaa's scoring source") / scorer)], check=True, cwd=working
        )
    shutil.move(working / "submission.csv", working / "nartaa_seat.csv")

    ours_mounts = Mounts(skip=MOUNTED_SERIES | heliosli_mounts)
    seven = (
        "final_ws_cf",
        "calib_span_ff",
        "stack_increments_ff",
        "through_dense_cf",
        "labelwise_depth24_drop40_long_s2026",
        "labelwise_ext_drop40_long_s2026",
        "labelwise_dense_drop40_long_s2026",
    )
    for names, share_of_left, table in (
        (("basinsoup_ff_k4_s2026",), 1.0 / 4.0, "soup_ff.csv"),
        (("basinsoup_dmlb_k7_s2026",), 1.0 / 4.0, "soup_dmlb.csv"),
        (seven, 1.0 / 3.0, "seven.csv"),
    ):
        left = max(KERNEL_SECONDS - (time.time() - started), 0.0)
        serving = Serving(vote=LogitAverage(), budget=Budget(seconds=left * share_of_left))
        KernelRun(names=names, overlap=False, serving=serving, mounts=ours_mounts).run()
        shutil.move(working / "submission.csv", working / table)
    gc.collect()
    torch.cuda.empty_cache()

    source = (mounts.one("helios_r7.py", "heliosli's R7 scoring source") / "helios_r7.py").read_text(encoding="utf-8")
    for anchor, (replacement, sites) in tolerant_r7.items():
        if source.count(anchor) != sites:
            message = f"R7 source holds {source.count(anchor)} copies of {anchor!r}, expected {sites}"
            raise ValueError(message)
        source = source.replace(anchor, replacement)
    Path("/tmp/helios_r7_tolerant.py").write_text(source, encoding="utf-8")
    finished = subprocess.run(
        [sys.executable, "-u", "/tmp/helios_r7_tolerant.py"],
        check=False,
        cwd=working,
        env=dict(os.environ, S6_T0=str(started + 0.3 * 3600.0)),
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
            working / "seven.csv",
            working / "r7_sub.csv",
            working / "nartaa_seat.csv",
            working / "raptor224_sub.csv",
        )
    ]
    RaptorSeat.combined(
        frames[0],
        frames[1:],
        [
            1.0 / 12.0,
            1.0 / 12.0,
            3.0 / 4.0 * (1.0 - RAPTOR_SHARE),
            3.0 / 4.0 * RAPTOR_SHARE * 0.65,
            3.0 / 4.0 * RAPTOR_SHARE * 0.35,
        ],
    ).to_csv(working / "submission.csv", index=False)

    reader = mounts.one("cnxt_v0_fold0.pt", "goodpjw2008's ConvNeXt reader")
    subprocess.run(  # noqa: S603 - pip installing the reader's own mounted wheels, fixed argv
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-index",
            "--no-deps",
            "--target",
            str(working / "_reader_env"),
            "--find-links",
            str(reader),
            *READER_WHEELS,
        ],
        check=False,
    )
    ran = (
        time.time() - started < READER_DEADLINE
        and subprocess.run(  # noqa: S603 - the reader's own infer.py from its mount, fixed argv
            [
                sys.executable,
                str(reader / "infer.py"),
                str(mounts.one("test_series.csv", "competition test data")),
                "test",
                str(working / "reader_seat.csv"),
                *sorted(str(path) for path in reader.glob("cnxt_v0_fold[012].pt")),
            ],
            check=False,
            env={
                **os.environ,
                "PYTHONPATH": os.pathsep.join([str(working / "_reader_env"), os.environ.get("PYTHONPATH", "")]),
            },
        ).returncode
        == 0
    )
    entry = pd.read_csv(working / "submission.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    own = (
        pd.read_csv(working / "reader_seat.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
        if ran
        else entry
    ).reindex(index=entry.index, columns=entry.columns)
    blended = ran and not own.isna().any().any() and bool((own.nunique() > 1).all())
    share = pd.Series(READER_SHARE).reindex(entry.columns)
    (
        (entry.rank(pct=True) * (1.0 - share) + own.rank(pct=True) * share).rank(pct=True)[entry.columns]
        if blended
        else entry
    ).reset_index().to_csv(working / "submission.csv", index=False)
    logging.getLogger(__name__).info(
        "READER LEG %s (ran %s) at %.0fs",
        "BLENDED" if blended else "SKIPPED, pfheavytrio kept",
        ran,
        time.time() - started,
    )


if __name__ == "__main__":
    main()
