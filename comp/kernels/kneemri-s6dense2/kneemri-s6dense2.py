"""s6dense2 — PICK CANDIDATE. s6third (champion 0.956, ref 57063964) with S6 at eval stride 2 instead of 3.

Same producers, shares and blend as `kneemri-s6third`: nartaa 384 at 1/2, heliosli's S6 at 1/3, the two soups at
1/12 each, rank space. One change: S6's `--eval-stride` 3 -> 2. Training sampled window centres at any slice, so
every centre is in distribution and a denser stride is a test-time average over 1.44x the windows, with no new input.

TIMED FIRST (kneemri-s6timing v1, T4, 120 train studies each): stride 3 8.3 s/study/worker, stride 2 9.1, zero
fallbacks; hidden projection 1.52 h -> 1.67 h of S6's window. S6 is decode-bound, so the extra windows cost ~10%.
In-sample per-label Spearman stride 3 vs 2: 0.971-0.995, so the ranking does move. Stride 1 (~3.5 h) is NOT
shipped: it needs nartaa + soups to finish inside 5.1 h on the hidden set, which has never been measured.

PRE-REGISTRATION (s6third 0.956, resolution 0.001, top-band residual sd 0.0013 @ledger75):
- **>= 0.957** — denser S6 lifts; becomes pick 1. Stride 1 earns a nartaa+soups timing run.
- **0.956** — neutral at board resolution; stride axis closed for S6 at 1/3.
- **<= 0.955** — denser windows hurt or S6 fell back; read the log's fallback count before blaming the mechanism.
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

    tolerant = {
        "'--amp', 'auto',": "'--max-fallback-fraction', '1.0', '--amp', 'auto',",
        "if rc != 0:": "if rc != 0 and not (pred / 'submission.csv').is_file():",
        "str(manifest.get('eval_stride', 3))": "'2'",
    }
    heliosli_mounts = frozenset({"rsna-knee-compact-v1", "rsna-orthofoundation-fold0-epoch7-20261004"})
    working = Path("/kaggle/working")
    started = time.time()
    mounts = Mounts()
    script = mounts.one("nartaa_mirror.py", "nartaa's scoring source") / "nartaa_mirror.py"
    subprocess.run([sys.executable, str(script)], check=True, cwd=working)  # noqa: S603 - our mounted copy, fixed argv
    foreign = working / "nartaa_seat.csv"
    shutil.move(working / "submission.csv", foreign)
    spent = time.time() - started
    logging.getLogger(__name__).info("nartaa's pass cost %.0fs", spent)

    serving = Serving(vote=LogitAverage(), budget=Budget(seconds=max(KERNEL_SECONDS - spent, 0.0) / 4.0))
    ours_mounts = Mounts(skip=MOUNTED_SERIES | heliosli_mounts)
    KernelRun(names=("basinsoup_ff_k4_s2026",), overlap=False, serving=serving, mounts=ours_mounts).run()
    shutil.move(working / "submission.csv", working / "soup_ff.csv")
    KernelRun(names=("basinsoup_dmlb_k7_s2026",), overlap=False, serving=serving, mounts=ours_mounts).run()
    shutil.move(working / "submission.csv", working / "soup_dmlb.csv")
    logging.getLogger(__name__).info("soups done at %.0fs", time.time() - started)
    gc.collect()
    torch.cuda.empty_cache()

    source = (mounts.one("helios_s6.py", "heliosli's S6 scoring source") / "helios_s6.py").read_text(encoding="utf-8")
    for anchor, replacement in tolerant.items():
        if source.count(anchor) != 1:
            message = f"S6 source holds {source.count(anchor)} copies of {anchor!r}, expected one"
            raise ValueError(message)
        source = source.replace(anchor, replacement)
    helios = Path("/tmp/helios_s6_tolerant.py")
    helios.write_text(source, encoding="utf-8")
    environment = dict(os.environ, S6_T0=str(started + 0.3 * 3600.0))
    finished = subprocess.run([sys.executable, "-u", str(helios)], check=False, cwd=working, env=environment)  # noqa: S603 - our mounted copy, patched, fixed argv
    logging.getLogger(__name__).info("S6 exited %d at %.0fs", finished.returncode, time.time() - started)
    s6 = working / "ours_sub.csv"
    if not s6.is_file():
        message = f"S6 left no table (exit {finished.returncode}), so this run is not the entry it is named for"
        raise RuntimeError(message)

    frames = [
        pd.read_csv(path, dtype={"StudyInstanceUID": str})
        for path in (working / "soup_ff.csv", working / "soup_dmlb.csv", s6, foreign)
    ]
    RaptorSeat.combined(frames[0], frames[1:], [1.0 / 12.0, 1.0 / 3.0, 1.0 / 2.0]).to_csv(
        working / "submission.csv", index=False
    )


if __name__ == "__main__":
    main()
