"""mirrorprobit — mirrorthird with the rank blend replaced by probit (latent-normal) fusion. INSTRUMENT.

Same seven, same nartaa seat, same 1/3 : 2/3 shares as `kneemri-mirrorthird` (0.953, ref 57023373). Only
the combiner moves: each table's per-label mid-rank percentile goes through `ndtri`, the z-scores are
averaged, and `ndtr` maps back (monotone, so AUC sees the z-average). The idea is from the public "0.954"
probit notebooks. Offline on replica c it priced +0.0008 [+0.0002, +0.0014] — under the board's 0.001
resolution, so this is an instrument: it asks whether a combiner effect transfers at all.

PRE-REGISTRATION (mirrorthird 0.953, top-band residual sd ~0.002):
- **>= 0.955** — the combiner transfers above its offline price; probit becomes the default blend.
- **0.952-0.954** — inside scatter; the combiner is not a lever at board resolution. Keep rank.
- **<= 0.951** — probit's tail weighting hurts on private-like studies; rank blend stays.
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
    from scipy.special import ndtr, ndtri  # noqa: PLC0415

    from core.mounts import Mounts  # noqa: PLC0415 - the kit is on sys.path only after boot
    from kneemri.inference.budget import KERNEL_SECONDS, Budget  # noqa: PLC0415
    from kneemri.inference.ensemble import Serving  # noqa: PLC0415
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415

    chosen = (
        "final_ws_cf",
        "calib_span_ff",
        "stack_increments_ff",
        "through_dense_cf",
        "labelwise_depth24_drop40_long_s2026",
        "labelwise_ext_drop40_long_s2026",
        "labelwise_dense_drop40_long_s2026",
    )
    working = Path("/kaggle/working")
    started = time.time()
    mounts = Mounts()
    script = mounts.one("nartaa_mirror.py", "nartaa's scoring source") / "nartaa_mirror.py"
    subprocess.run([sys.executable, str(script)], check=True, cwd=working)  # noqa: S603 - our mounted copy, fixed argv
    foreign = working / "nartaa_seat.csv"
    shutil.move(working / "submission.csv", foreign)
    spent = time.time() - started
    logging.getLogger(__name__).info("nartaa's pass cost %.0fs", spent)

    KernelRun(
        names=chosen,
        overlap=False,
        serving=Serving(vote=LogitAverage(), budget=Budget(seconds=max(KERNEL_SECONDS - spent, 0.0))),
        mounts=mounts,
    ).run()
    ours = pd.read_csv(working / "submission.csv")
    theirs = pd.read_csv(foreign, dtype={"StudyInstanceUID": str})
    ours["StudyInstanceUID"] = ours["StudyInstanceUID"].astype(str)
    labels = [column for column in ours.columns if column != "StudyInstanceUID"]
    aligned = theirs.set_index("StudyInstanceUID").reindex(ours["StudyInstanceUID"])[labels]
    if aligned.isna().any(axis=None):
        message = f"{int(aligned.isna().any(axis=1).sum())} studies are absent from nartaa's table"
        raise ValueError(message)
    count = len(ours)
    probit_ours = ndtri((ours[labels].rank(method="average").to_numpy() - 0.5) / count)
    probit_theirs = ndtri((aligned.rank(method="average").to_numpy() - 0.5) / count)
    fused = pd.DataFrame(ndtr(probit_ours / 3.0 + 2.0 * probit_theirs / 3.0), columns=labels)
    fused.insert(0, "StudyInstanceUID", ours["StudyInstanceUID"].to_numpy())
    fused.to_csv(working / "submission.csv", index=False)


if __name__ == "__main__":
    main()
