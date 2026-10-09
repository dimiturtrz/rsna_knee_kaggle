"""heliseat — mirrorseat with a THIRD family: heliosli's compact convnext_small inside nartaa's half.

DRAFT, NOT RUN. heliosli's weights (`heliosli/rsna-knee-compact-v1`, code `heliosli/rsna-orthofoundation-
fold0-epoch7-20261004`) are public Kaggle datasets and so admissible under
rules 2.6.a, but they carry no licence grant, and using them waits on the owner's word (knee-dbj03).
Their stage runs verbatim from the private `dimiturnt/heli-compact-src`: stage 2 of
`heliosli/rsna-knee-blend-gold` (bundle c7ebab1eb953f1a6, arm E_lrprior), cut at its own `ours_sub.csv`.
Never vendored here.

The wave is heliosli's own recipe: nartaa 0.7 / compact 0.3 rank blend, which read 0.951 public against
nartaa's 0.949 alone. Ours then takes 1/2 against the wave exactly as in mirrorseat (0.953), so the
effective shares are ours 0.50, nartaa 0.35, compact 0.15. No share is fitted; all three are named before
the number.

PRE-REGISTRATION (mirrorseat 0.953, top-band residual sd ~0.002, board resolution 0.001):
- **>= 0.955** — the compact family is complementary to ours too; re-pin.
- **0.953-0.954** — directional at most; the +0.002 it gave nartaa is shared with ours.
- **<= 0.952** — the compact model's signal is already in our seven; close the third-family axis.
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


COMPACT_SHARE = 0.3
"""The compact model's share of the wave, verbatim from heliosli's blend-gold (0.951 public)."""


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
    log = logging.getLogger(__name__)
    working = Path("/kaggle/working")
    started = time.time()
    mounts = Mounts()
    for name, label, seat in (
        ("nartaa_mirror.py", "nartaa's scoring source", "submission.csv"),
        ("heli_compact.py", "heliosli's compact stage", "compact_seat.csv"),
    ):
        script = mounts.one(name, label) / name
        subprocess.run([sys.executable, str(script)], check=True, cwd=working)  # noqa: S603 - our mounted copy, fixed argv
        if seat == "submission.csv":
            shutil.move(working / seat, working / "nartaa_seat.csv")
        log.info("%s done at %.0fs", label, time.time() - started)

    nartaa = pd.read_csv(working / "nartaa_seat.csv", dtype={STUDY: str}).set_index(STUDY)
    compact = pd.read_csv(working / "compact_seat.csv", dtype={STUDY: str}).set_index(STUDY).reindex(nartaa.index)
    if compact[list(TARGETS)].isna().any().any() or (compact[list(TARGETS)].nunique() <= 1).any():
        message = "the compact table is missing studies or constant in a finding"
        raise ValueError(message)
    wave = (1.0 - COMPACT_SHARE) * nartaa[list(TARGETS)].rank(pct=True) + COMPACT_SHARE * compact[list(TARGETS)].rank(
        pct=True
    )
    wave = wave.rank(pct=True).reset_index()

    spent = time.time() - started
    KernelRun(
        names=chosen,
        overlap=False,
        serving=Serving(vote=LogitAverage(), budget=Budget(seconds=max(KERNEL_SECONDS - spent, 0.0))),
        mounts=mounts,
    ).run()
    ours = pd.read_csv(working / "submission.csv", dtype={STUDY: str})
    RaptorSeat(weight=0.5).blended(ours, wave).to_csv(working / "submission.csv", index=False)


if __name__ == "__main__":
    main()
