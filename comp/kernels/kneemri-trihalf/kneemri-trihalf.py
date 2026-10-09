"""trihalf — triseat with our seven at 1/2 of the final rank blend instead of 1/3.

Same wave (nartaa's CoAtNet + goodpjw2008's ConvNeXt reader, rabari9999's V1 per-finding shares) and the
same seven as `kneemri-triseat`; only OURS moves. It is the branch `triseat`'s pre-registration names for a
0.950 read: the replicated floor reproduced and ours at 1/3 added no tick, so the next question is whether
the third family is under-weighted. Evidence for a larger share: ours7 reads 0.945 solo against the wave's
0.950, a gap of five ticks, much closer than the 0.929 reader the wave already seats at up to 0.35; and on
the 896-study holdout a blend of ours with raptor-recipe replicas rose monotonically to ours at 2/3
(+0.0157 at 1/3, +0.0206 at 1/2, ledger75) — a ruler that understates nartaa, so the share does not
transfer, only its direction.

PRE-REGISTRATION (wave 0.950, ours7 0.945, triseat at 1/3 read first, board resolution 0.001):
- **>= 0.952** — the third family was under-weighted; 1/2 is the new floor.
- **0.951** — one tick: the medal line (444 teams tie at 0.950, so 0.950 ranks ~646, 0.951 ~201).
- **<= 0.950** — no gain from weight; the third family adds nothing at either share.
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

OURS = 0.5
"""Half the final blend. See the module docstring."""


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
    RaptorSeat(weight=1.0 - OURS).blended(ours, wave).to_csv(working / "submission.csv", index=False)


if __name__ == "__main__":
    main()
