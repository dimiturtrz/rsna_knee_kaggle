"""oaifreeshare — `oaifree` with our seven at 2/3 and goodpjw2008's reader at 1/3: the hedge's second share point.

Same members, same reader, same code as `kneemri-oaifree` (ours 1/2); only `OURS` moves. The two read together
are the hedge's share curve, which the board has never seen without nartaa. The pre-registered rule (knee-kl48k,
17:07Z) was 2/3 if `kneemri-readeroof`'s argmax lands above 1/2, else 1/3. On the 562 studies the reader never
trained on (leak control +0.033, fold map holds), ours 1/2 reads 0.9080, 2/3 0.9108, 3/4 0.9114 (argmax): 2/3.
A tie-break, not a finding: the board has not reproduced offline share preferences before.

The rest of this docstring is `oaifree`'s, unchanged.

Every entry above the 0.950 pile seats nartaa's CoAtNet, whose public weights were fitted with OAI data, and the
forum holds three unanswered threads (747455, 744056, 743416) asking the host whether public weights trained on
OAI are eligible. If the ruling goes against them, `mirrorthird` and `mirrorseat` both fall, and so does the whole
0.950 pile. Our best entry with no foreign weights is `basinsoupff_v1` at 0.946. This asks what the second pick
would be if it had to stand on weights whose provenance is the competition alone: the reader is timm ImageNet
ConvNeXt-T trained on the competition's own studies against labels LLM-read from the competition's own reports
(`make_labels.py`, `train.py`, `knee.py` from its mount; no external images, no external checkpoint).

Half each because neither side has a measured ordering against the other on any ruler: the reader trained on
every study our holdout holds, so no offline number can price the share, and the board has only seen it inside
nartaa's wave.

PRE-REGISTRATION (ours alone 0.945-0.946, triseat 0.952, top-band residual sd ~0.002, resolution 0.001):
- **>= 0.950** — an OAI-free entry at the pile: pick 2 becomes this, the hedge that survives a ruling.
- **0.948-0.949** — the reader is a real second family; pick 2 stays `mirrorseat` unless the host rules.
- **<= 0.947** — the reader carries little without nartaa beside it; the hedge is `basinsoupff_v1`'s level.
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


OURS = 2 / 3
"""Our seven's share against the reader. See the module docstring."""


def main() -> None:
    boot()

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
    if reading.isna().any().any() or (reading.nunique() <= 1).any():
        message = "the reader's table is missing studies or constant in a finding"
        raise ValueError(message)
    reading.rank(pct=True).reset_index().to_csv(working / "reader_ranks.csv", index=False)

    spent = time.time() - started
    KernelRun(
        names=chosen,
        overlap=False,
        serving=Serving(vote=LogitAverage(), budget=Budget(seconds=max(KERNEL_SECONDS - spent, 0.0))),
        mounts=mounts,
    ).run()
    ours = pd.read_csv(working / "submission.csv", dtype={STUDY: str})
    wave = pd.read_csv(working / "reader_ranks.csv", dtype={STUDY: str})
    RaptorSeat(weight=1.0 - OURS).blended(ours, wave).to_csv(working / "submission.csv", index=False)


if __name__ == "__main__":
    main()
