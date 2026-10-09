"""tiltseat — triseat with our seven's share set PER FINDING inside the board-flat 1/3..1/2 band.

Same wave and same seven as `kneemri-triseat`; only the final share moves, and only per finding. The board
read ours at 1/3 (`triseat`) and 1/2 (`trihalf`) identically at 0.952, so the global level is flat there and
this spends no slot looking for it. What two independent holdout instruments agree on is the PROFILE: ours
against raptor-recipe replicas (2026-09 instrument and
`interpretations/kneemri/2026-10-09_no-stratum-flips-the-blend-share.md`, ledger75, 896 studies) leads by
>= 0.03 on ACL, MCL, Lateral Meniscus, PF OA and Synovitis, by less on the meniscal/OA middle,
ties on Effusion, Baker's and Contusion, and loses only Fracture. Only the SIGN of a per-finding holdout
delta has ever transferred, so the tilt moves inside the tested band and one step past it for Fracture,
never to the 0.9 the holdout's own optimum asks for.

PRE-REGISTRATION (triseat and trihalf both 0.952, board resolution 0.001):
- **>= 0.953** — the per-finding profile transfers; tilt is the new floor.
- **0.952** — flat; the profile is real offline and below board resolution.
- **<= 0.951** — the holdout profile anti-transfers on the strong findings; per-finding shares are closed.
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

HALF, THIRD, QUARTER = 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0
OURS = {
    "ACL": HALF,
    "MCL": HALF,
    "Lateral Meniscus": HALF,
    "PF OA": HALF,
    "Synovitis": HALF,
    "Medial Meniscus": THIRD,
    "Medial OA": THIRD,
    "Lateral OA": THIRD,
    "Effusion": THIRD,
    "Baker's": THIRD,
    "Contusion": THIRD,
    "Fracture": QUARTER,
}
"""Our seven's share of the final blend, per finding. See the module docstring."""


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
    if set(READER_SHARE) != set(TARGETS) or set(OURS) != set(TARGETS):
        message = f"shares name {sorted(READER_SHARE)} / {sorted(OURS)}, the board scores {sorted(TARGETS)}"
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

    spent = time.time() - started
    KernelRun(
        names=chosen,
        overlap=False,
        serving=Serving(vote=LogitAverage(), budget=Budget(seconds=max(KERNEL_SECONDS - spent, 0.0))),
        mounts=mounts,
    ).run()
    ours = pd.read_csv(working / "submission.csv", dtype={STUDY: str})
    blends = {share: RaptorSeat(weight=1.0 - share).blended(ours, wave) for share in set(OURS.values())}
    tilted = blends[THIRD].copy()
    for finding, share in OURS.items():
        tilted[finding] = blends[share][finding]
    log.info("per-finding shares %s", OURS)
    tilted.to_csv(working / "submission.csv", index=False)


if __name__ == "__main__":
    main()
