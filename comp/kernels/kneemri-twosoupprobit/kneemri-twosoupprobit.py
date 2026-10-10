"""twosoupprobit — twosoupmirror with the rank blend replaced by probit (latent-normal) fusion. INSTRUMENT.

The fourth corner of a 2x2: ours side {ours7, two soups} x combiner {rank, probit}, nartaa 384 at 2/3
throughout. `kneemri-mirrorthird` (ours7, rank) read 0.953; `kneemri-twosoupmirror` (soups, rank) and
`kneemri-mirrorprobit` (ours7, probit) are the other two. Each table's per-label mid-rank percentile goes
through `ndtri`, the z-scores take the shares 1/6 + 1/6 + 2/3, and `ndtr` maps back.

Offline fold-0 analog (vs ours6+nartaa rank, replicas a/c, n=896): +0.0020/+0.0024 [+0.0008,+0.0035],
P>0 1.00 both; rank twosoup reads +0.0015/+0.0016 on the same rows. Best of several compositions read,
so mildly selected.

PRE-REGISTRATION (mirrorthird 0.953, top-band residual sd ~0.002):
- **>= 0.955** — the two knobs compose on the board; pick-1 candidate over mirrorthird.
- **0.953-0.954** — inside scatter; read against twosoupmirror and mirrorprobit to split the corners.
- **<= 0.952** — the offline gain does not transfer; mirrorthird stays.
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

    chosen = ("basinsoup_ff_k4_s2026",)
    working = Path("/kaggle/working")
    started = time.time()
    mounts = Mounts()
    script = mounts.one("nartaa_mirror.py", "nartaa's scoring source") / "nartaa_mirror.py"
    subprocess.run([sys.executable, str(script)], check=True, cwd=working)  # noqa: S603 - our mounted copy, fixed argv
    foreign = working / "nartaa_seat.csv"
    shutil.move(working / "submission.csv", foreign)
    spent = time.time() - started
    logging.getLogger(__name__).info("nartaa's pass cost %.0fs", spent)

    ours_seconds = max(KERNEL_SECONDS - spent, 0.0) / 2.0
    KernelRun(
        names=chosen,
        overlap=False,
        serving=Serving(vote=LogitAverage(), budget=Budget(seconds=ours_seconds)),
        mounts=mounts,
    ).run()
    first = working / "soup_ff.csv"
    shutil.move(working / "submission.csv", first)
    KernelRun(
        names=("basinsoup_dmlb_k7_s2026",),
        overlap=False,
        serving=Serving(vote=LogitAverage(), budget=Budget(seconds=ours_seconds)),
        mounts=mounts,
    ).run()
    tables = [
        pd.read_csv(path, dtype={"StudyInstanceUID": str}) for path in (first, working / "submission.csv", foreign)
    ]
    ours = tables[0]
    labels = [column for column in ours.columns if column != "StudyInstanceUID"]
    count = len(ours)
    fused = 0.0
    for table, share in zip(tables, (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0), strict=True):
        aligned = table.set_index("StudyInstanceUID").reindex(ours["StudyInstanceUID"])[labels]
        if aligned.isna().any(axis=None):
            message = f"{int(aligned.isna().any(axis=1).sum())} of {count} studies are absent from a table"
            raise ValueError(message)
        fused = fused + share * ndtri((aligned.rank(method="average").to_numpy() - 0.5) / count)
    frame = pd.DataFrame(ndtr(fused), columns=labels)
    frame.insert(0, "StudyInstanceUID", ours["StudyInstanceUID"].to_numpy())
    frame.to_csv(working / "submission.csv", index=False)


if __name__ == "__main__":
    main()
