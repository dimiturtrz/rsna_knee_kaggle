"""A DELIBERATELY WEAK seven, fired to give the offline-to-board map a second anchor.

THIS ENTRY IS AN INSTRUMENT, NOT A CANDIDATE. It is pre-registered to score WORSE than the champion and
it is expected to. Its only job is to place one point far enough off the existing chain that the chain's
slope becomes measurable.

**THE HOLE IT FILLS (knee-825t).** Every board reading we own sits in a 0.0043 offline band: six pairs
spanning 0.8648-0.8691 offline against 0.927-0.934 board. A slope fitted inside that band is fitted
inside our own 0.0062 holdout noise floor, so the recorded median of +4.67 board points per offline
point is unfalsifiable from the data that produced it. No amount of further training fixes this — the
missing quantity is a point OUTSIDE the band, and that costs no GPU at all.

**WHY THESE SEVEN.** `record_mix` over the 28 members packaged in `kneemri-members`, taking the seven
with the weakest solo holdout AUC (0.8215-0.8462): joint **0.8562**. That is 0.0086 below the chain's
floor - twice its total range and three times its span - and it is the furthest any point can sit while
every member stays rebuildable from `comp/members.roster`.

**COUNT AND VOTE ARE HELD FIXED.** Seven seats, `RankAverage`, same kit. Only member QUALITY moves.
A one- or two-member entry would reach further offline (0.8512-0.8636) but changes the VOTE STRUCTURE,
which confounds quality with ensemble width and makes the point unreadable against the chain.

PRE-REGISTRATION, written before the number. Predictions are for the offline gap of -0.0086.

- 0.885-0.890 - the recorded slope (4.67) holds out of band. The offline holdout is a real ruler over a
  wide range, seat-level offline deltas can be trusted at face value, and the chain's flatness is
  saturation near the top rather than a broken measurement.
- 0.905-0.920 - the slope in the large is ~1-2, not 4.67. The steep local slope is an artefact of six
  points inside the noise floor. Every offline delta we have used to justify a seat is worth a fraction
  of what it was priced at, and `board_transfer` must be refitted or retired.
- above 0.925 - the board is nearly INDIFFERENT to member quality at this width. Seven-member rank
  averaging is doing the work, not the members, and the entire member-search axis is closed: budget goes
  to width, to new data, and to loss/target design, never to swapping seats again.
- below 0.880 - the map is steeper than recorded out of band, meaning offline gains are UNDER-priced.
  That licenses spending slots on small offline wins we have been dismissing as noise.

There is no outcome on which this entry becomes a champion, and none on which it teaches nothing.
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

    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - the kit is on sys.path only after boot

    chosen = (
        "lora_r8_v3small_224_s2026",
        "dinov3_base_lora_s2026",
        "crop80_tissue_s2026",
        "deep24_s2026",
        "band_full_s2026",
        "intensity392_ws_s2026",
        "extended_ws_s2026",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
