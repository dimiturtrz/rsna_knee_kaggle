"""The HIGH anchor of the current-kit solo pair — `calib_dense_hold_s2026` alone, offline 0.8634.

**WHY THIS SLOT.** `comp/board.jsonl` holds exactly two entries that served ONE fold-0 member, and they
disagree with the ruler:

    solo_v1        2026-08-12  offline 0.8518  board 0.868   DINOv2, the kit of that week
    ortho_solo_v1  2026-09-03  offline 0.8452  board 0.907   DINOv3 continued on 1.25M knee slices

The lower offline scored 39 board points HIGHER. Those are the record's first two resolved inversions in
342 pairs (`onedecode_v2` supplies the second), and both have `ortho_solo_v1` on the same side. Against
the fitted whole-record slope its 0.8452 predicts about 0.886; it read 0.907.

**TWO READINGS, AND THE RECORD CANNOT SEPARATE THEM.** Either the ortholarge trunk transfers to the
hidden test set better than our fold-0 holdout says, or the CURRENT KIT transfers better than the August
kits did. Offline level and date are collinear across all 37 priced entries — every low-offline entry is
2026-08-12..08-24 and every late entry is high-offline — so no row in the record breaks the tie. This
pair of slots is a low-offline point in the CURRENT kit, which is the cell the record has never held.

**PRE-REGISTRATION, written before either score.** Read the LEVEL first; the slope between the two is
secondary. Members are `calib_span_s2026` (offline 0.8545) and `calib_dense_hold_s2026` (0.8634), both
`--fold 0`, both withholding all 909 holdout studies, both already packaged and proven to mount by
`kneemri-foldfour`. They span 0.0089, above the 0.0053 resolution floor.

- **both at or above 0.900** — the kit-era term is large and `ortho_solo_v1`'s 0.907 is explained by it.
  ortholarge is not special, `ortho_stack`'s middle branch closes the in-domain trunk lane exactly as
  written, and `board_transfer`'s whole-record slope is contaminated by an era term that must be fitted.
- **both at or below 0.885** — the kit does not explain it. At equal offline the knee-continued DINOv3
  trunk sits roughly +0.02 board above the DINOv2 line, the in-domain trunk lane stays OPEN whatever
  `ortho_stack` reads offline, and every offline number this project quotes for a non-DINOv2 member is
  an under-statement of its board value.
- **0.886 to 0.899** — partial. Report the split, close nothing. The era term is real but smaller than
  the residual, and the remainder is still the trunk's or still unexplained.

**WHAT IT CANNOT DO.** Two board reads cannot separate "the current kit" from "these two particular
members", and nothing here touches SEATING — one member has no vote rule and no pass-through. It buys
no rank: the public board keeps the best entry, and the champion stands at 0.944.

**THIS HALF** is the high anchor. Its partner is `kneemri-spansolo`, the same construction at 0.8545.
Either half alone answers the LEVEL question; the pair adds the current-kit local slope.

bead: knee-l67zk · interpretation: interpretations/kneemri/2026-09-04_the-ruler-had-four-rows-outside-it.md
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

    # The high anchor of the current-kit solo pair. Partner: kneemri-spansolo.
    chosen = ("calib_dense_hold_s2026",)
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
