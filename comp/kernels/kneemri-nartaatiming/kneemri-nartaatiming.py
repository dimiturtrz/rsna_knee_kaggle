"""INSTRUMENT, never an entry: what nartaa 384 and the two soups cost per hidden study, before S6 starts.

The champion runs nartaa, then `basinsoup_ff_k4`, then `basinsoup_dmlb_k7`, and only then S6, whose deadline is
fixed at kernel start + 8.6 h. Whatever those three spend is taken out of S6's window, and on the hidden test it
has never been measured: the visible run is three studies. S6 at stride 3 needs ~1.52 h and stride 2 ~1.67 h
(kneemri-s6timing), so stride 2 is safe if the three finish inside 6.9 h and stride 1 (~3.5 h) only inside
5.1 h (knee-1oxsy). Each runs on TRAIN studies at two sizes, disjoint so no decode cache
is reused, through a staged copy of the competition tables: the slope is the per-study cost and the intercept
the load. In-sample, so TIME only.
"""

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


SIZES = (40, 120)
HIDDEN = 1322


def main() -> None:
    boot()

    import json  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    import time  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    from core.mounts import SERIES as MOUNTED_SERIES  # noqa: PLC0415 - the kit is on sys.path only after boot
    from core.mounts import Mounts  # noqa: PLC0415
    from kneemri.inference.budget import KERNEL_SECONDS, Budget  # noqa: PLC0415
    from kneemri.inference.ensemble import Ensemble, Serving  # noqa: PLC0415
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415
    from kneemri.kaggle import kernel_run  # noqa: PLC0415
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415
    from kneemri.kaggle.submission import Split, Submission  # noqa: PLC0415

    working = Path("/kaggle/working")
    mounts = Mounts()
    competition = mounts.one("sample_submission.csv", "competition data")
    original = (mounts.one("nartaa_mirror.py", "nartaa's scoring source") / "nartaa_mirror.py").read_text()
    ours_mounts = Mounts(skip=MOUNTED_SERIES)
    serving = Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS / 4.0))
    train = pd.read_csv(competition / "train.csv", dtype={"StudyInstanceUID": str})
    series = pd.read_csv(competition / "train_series.csv", dtype={"StudyInstanceUID": str})
    anchor = "ROOT = find_test_root()"
    if original.count(anchor) != 1:
        message = f"nartaa source holds {original.count(anchor)} copies of {anchor!r}, expected one"
        raise ValueError(message)
    timing: dict[int, dict[str, float]] = {}
    for size in SIZES:
        staged = Path(f"/tmp/staged{size}")
        staged.mkdir(parents=True)
        chosen = train.iloc[sum(timing) : sum(timing) + size]
        chosen.to_csv(staged / "test.csv", index=False)
        series[series["StudyInstanceUID"].isin(chosen["StudyInstanceUID"])].to_csv(
            staged / "test_series.csv", index=False
        )
        shutil.copyfile(competition / "sample_submission.csv", staged / "sample_submission.csv")
        (staged / "test_series").symlink_to(competition / "train_series", target_is_directory=True)
        script = Path(f"/tmp/nartaa_staged{size}.py")
        script.write_text(original.replace(anchor, f"ROOT = {str(staged)!r}"), encoding="utf-8")

        class Staged(Submission):
            root = staged

            def __init__(self, ensemble: Ensemble, _competition: Path, working: Path, *, overlap: bool = True) -> None:
                super().__init__(
                    ensemble, self.root, working, overlap=overlap, split=Split(series_root=self.root / "test_series")
                )

        kernel_run.Submission = Staged
        started = time.time()
        subprocess.run([sys.executable, str(script)], check=True, cwd=working)  # noqa: S603 - our mounted copy, one root patched
        row: dict[str, float] = {"nartaa": round(time.time() - started, 1)}
        for name in ("basinsoup_ff_k4_s2026", "basinsoup_dmlb_k7_s2026"):
            started = time.time()
            KernelRun(names=(name,), overlap=False, serving=serving, mounts=ours_mounts).run()
            row[name] = round(time.time() - started, 1)
        timing[size] = row
        logging.getLogger(__name__).info("TIMING %d studies %s", size, json.dumps(row))
    small, large = (timing[size] for size in SIZES)
    projection = {
        stage: round(
            (small[stage] + (large[stage] - small[stage]) * (HIDDEN - SIZES[0]) / (SIZES[1] - SIZES[0])) / 3600, 2
        )
        for stage in large
    }
    projection["total_hidden_h"] = round(sum(projection.values()), 2)
    (working / "timing.json").write_text(json.dumps({"runs": timing, "projection": projection}, indent=1))
    logging.getLogger(__name__).info("PROJECTION %s", json.dumps(projection))


if __name__ == "__main__":
    main()
