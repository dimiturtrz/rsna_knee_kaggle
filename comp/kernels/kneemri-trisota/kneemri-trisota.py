"""trisota — triseat with the public wave rebuilt to rabari9999's `rsna-knee-sota-0-954` fusion.

Same seven, same 1/3 for ours as `kneemri-triseat` (0.952, ref 57001137). Only the WAVE moves, to the cell
rabari9999 published on 2026-10-10 01:00Z: nartaa's 384 CoAtNet and his 224-crop CoAtNet rank-mixed 0.7/0.3
into one CoAtNet side (both served by nartaa's own scoring sources, mounted from `dimiturnt/nartaa-mirror-src`
and `dimiturnt/nartaa-0945-src` as in `kneemri-mirrorpair`), then goodpjw2008's ConvNeXt reader at a
per-finding share that differs from triseat's on 8 of 12 findings — Lateral OA 0.08 -> 0.50 the big one.

AN INSTRUMENT, NOT AN IMPROVEMENT. The notebook's 0.954 is a TITLE TARGET with no score shown, and its table
was tuned against the public board. What this buys is the answer to whether a public-board-tuned per-finding
table transfers when a third family sits on top of it — the question every remaining public retune poses,
asked once.

PRE-REGISTRATION (triseat 0.952, mirrorthird 0.953, top-band residual sd ~0.002, resolution 0.001):
- **>= 0.954** — the retuned wave carries; this wave replaces nartaa-alone under ours and pick 2 is re-read.
- **0.952-0.953** — public share tables do not transfer under a third family; closes the retune axis.
- **<= 0.951** — the retune overfit the public board; closes it harder.
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
    "Lateral OA": 0.50,
    "Baker's": 0.42,
    "Contusion": 0.36,
    "ACL": 0.30,
    "Medial OA": 0.28,
    "MCL": 0.25,
    "Medial Meniscus": 0.22,
    "Lateral Meniscus": 0.08,
    "Fracture": 0.08,
    "PF OA": 0.08,
    "Effusion": 0.04,
    "Synovitis": 0.03,
}
"""The ConvNeXt reader's share of the wave, per finding, verbatim from rabari9999's `sota-0-954` cell 10."""

CROP_SHARE = 0.30
"""nartaa's 224-crop CoAtNet inside the CoAtNet side (`W_C224`), verbatim from the same cell."""

FOREIGN = (
    ("nartaa_mirror.py", "nartaa's 384 scoring source"),
    ("nartaa_0945.py", "nartaa's 224-crop scoring source"),
)
"""Each script writes /kaggle/working/submission.csv; the first is the 384 CoAtNet, the second the 224 crop."""

OURS = 1.0 / 3.0
"""One family's seat of three. See the module docstring."""


def main() -> None:
    boot()

    import shutil  # noqa: PLC0415
    import time  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    from core.findings import STUDY, TARGETS  # noqa: PLC0415 - the kit is on sys.path only after boot
    from core.mounts import Mounts  # noqa: PLC0415
    from kneemri.inference.budget import KERNEL_SECONDS, Budget  # noqa: PLC0415
    from kneemri.inference.ensemble import Serving  # noqa: PLC0415
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415
    from kneemri.kaggle.raptor_seat import RaptorSeat  # noqa: PLC0415

    chosen = (
        "final_ws_cf",
        "calib_span_ff",
        "stack_increments_ff",
        "through_dense_cf",
        "labelwise_depth24_drop40_long_s2026",
        "labelwise_ext_drop40_long_s2026",
        "labelwise_dense_drop40_long_s2026",
    )
    if set(READER_SHARE) != set(TARGETS):
        message = f"reader shares name {sorted(READER_SHARE)}, the board scores {sorted(TARGETS)}"
        raise ValueError(message)
    log = logging.getLogger(__name__)
    working = Path("/kaggle/working")
    started = time.time()
    mounts = Mounts()

    coats = []
    for name, label in FOREIGN:
        script = mounts.one(name, label) / name
        subprocess.run([sys.executable, str(script)], check=True, cwd=working)  # noqa: S603 - our mounted copy, fixed argv
        seat = working / f"{Path(name).stem}_seat.csv"
        shutil.move(working / "submission.csv", seat)
        table = pd.read_csv(seat, dtype={STUDY: str}).set_index(STUDY)[list(TARGETS)]
        coats.append(table.reindex(coats[0].index) if coats else table)
        log.info("%s done at %.0fs", label, time.time() - started)

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

    reading = pd.read_csv(working / "reader_seat.csv", dtype={STUDY: str}).set_index(STUDY)[list(TARGETS)]
    for table in [*coats, reading.reindex(coats[0].index)]:
        if table.isna().any().any() or (table.nunique() <= 1).any():
            message = "a foreign table is missing studies or constant in a finding"
            raise ValueError(message)
    nartaa = ((1.0 - CROP_SHARE) * coats[0].rank(pct=True) + CROP_SHARE * coats[1].rank(pct=True)).rank(pct=True)
    convnext = reading.reindex(nartaa.index)
    wave = pd.DataFrame(index=nartaa.index)
    for finding, share in READER_SHARE.items():
        wave[finding] = (1.0 - share) * nartaa[finding].rank(pct=True) + share * convnext[finding].rank(pct=True)
    wave = wave[list(TARGETS)].rank(pct=True).reset_index()
    wave.to_csv(working / "wave_seat.csv", index=False)

    spent = time.time() - started
    KernelRun(
        names=chosen,
        overlap=False,
        serving=Serving(vote=LogitAverage(), budget=Budget(seconds=max(KERNEL_SECONDS - spent, 0.0))),
        mounts=mounts,
    ).run()
    ours = pd.read_csv(working / "submission.csv", dtype={STUDY: str})
    RaptorSeat(weight=1.0 - OURS).blended(ours, wave).to_csv(working / "submission.csv", index=False)


if __name__ == "__main__":
    main()
