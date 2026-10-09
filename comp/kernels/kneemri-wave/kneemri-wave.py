"""wave — the public 0.950 blend alone: nartaa's CoAtNet and goodpjw2008's ConvNeXt reader, ranked per finding.

The FALLBACK beside `kneemri-triseat`, which is this same table with our seven seated at 1/3. Nothing of
ours is in the output: it is the replication half of triseat shipped on its own, and its submission message
says so. It exists to answer one question if triseat reads below the wave — did 1/3 of ours dilute, or did
the wave not reproduce — and to hold the 0.950 line while that is answered.

PRE-REGISTRATION (wave reported 0.950 by `rabari9999/rsna-knee-lb-0-95`, board resolution 0.001):
- **0.950** — reproduced. triseat's delta against this is ours' worth at 1/3.
- **<= 0.949** — not reproduced; the reader or nartaa's pass differs from the public notebooks' run.
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
    "Baker's": 0.35,
    "Contusion": 0.30,
    "Medial OA": 0.25,
    "ACL": 0.25,
    "Medial Meniscus": 0.20,
    "MCL": 0.20,
    "Lateral Meniscus": 0.08,
    "Fracture": 0.08,
    "Lateral OA": 0.08,
    "PF OA": 0.08,
    "Effusion": 0.04,
    "Synovitis": 0.03,
}
"""The ConvNeXt reader's share of the wave, per finding, verbatim from rabari9999's V1 (LB 0.950)."""


def main() -> None:
    boot()

    import shutil  # noqa: PLC0415
    import time  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    from core.findings import STUDY, TARGETS  # noqa: PLC0415 - the kit is on sys.path only after boot
    from core.mounts import Mounts  # noqa: PLC0415

    if set(READER_SHARE) != set(TARGETS):
        message = f"reader shares name {sorted(READER_SHARE)}, the board scores {sorted(TARGETS)}"
        raise ValueError(message)
    log = logging.getLogger(__name__)
    working = Path("/kaggle/working")
    started = time.time()
    mounts = Mounts()

    script = mounts.one("nartaa_mirror.py", "nartaa's scoring source") / "nartaa_mirror.py"
    subprocess.run([sys.executable, str(script)], check=True, cwd=working)  # noqa: S603 - our mounted copy, fixed argv
    shutil.move(working / "submission.csv", working / "nartaa_seat.csv")
    log.info("nartaa's pass done at %.0fs", time.time() - started)

    reader = mounts.one("cnxt_v0_fold0.pt", "goodpjw2008's ConvNeXt reader")
    data = mounts.one("test_series.csv", "competition test data")
    folds = sorted(str(path) for path in reader.glob("cnxt_v0_fold*.pt"))
    target = working / "_reader_env"
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
            str(target),
            "--find-links",
            str(reader),
            "pylibjpeg",
            "pylibjpeg-libjpeg",
            "pylibjpeg-openjpeg",
        ],
        check=False,
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(target), os.environ.get("PYTHONPATH", "")])}
    subprocess.run(  # noqa: S603 - the reader's own infer.py from its mount, fixed argv
        [sys.executable, str(reader / "infer.py"), str(data), "test", str(working / "reader_seat.csv"), *folds],
        check=True,
        env=env,
    )
    log.info("reader's pass (%d folds) done at %.0fs", len(folds), time.time() - started)

    nartaa = pd.read_csv(working / "nartaa_seat.csv", dtype={STUDY: str}).set_index(STUDY)
    convnext = pd.read_csv(working / "reader_seat.csv", dtype={STUDY: str}).set_index(STUDY).reindex(nartaa.index)
    if convnext[list(TARGETS)].isna().any().any() or (convnext[list(TARGETS)].nunique() <= 1).any():
        message = "the reader's table is missing studies or constant in a finding"
        raise ValueError(message)
    wave = pd.DataFrame(index=nartaa.index)
    for finding, share in READER_SHARE.items():
        wave[finding] = (1.0 - share) * nartaa[finding].rank(pct=True) + share * convnext[finding].rank(pct=True)
    wave = wave[list(TARGETS)].rank(pct=True).reset_index()
    wave.to_csv(working / "wave_seat.csv", index=False)

    wave.to_csv(working / "submission.csv", index=False)


if __name__ == "__main__":
    main()
