"""The four chosen by their JOINT holdout, with the encoder-rate member in place of slice attention.

This is `kneemri-chosen` (LB 0.918) with one member swapped, and the swap is the only thing measured
since: `lr4e-5_ws_s2026` is the control recipe at encoder rate 4e-5 instead of 8e-6, which scored 0.8525
solo against 0.8457 — +0.0068, the first single change to clear the 0.0053 floor in weeks (knee-wnkv).
It also reaches that at 336 px, where `res392` needed a 64 GB cache for the same number, because 392 px
at the control's own rate is worth +0.0002 and the gain was never the pixels (knee-a8x).

Re-scored over the eight cached same-split members on the 909-study fold-0 holdout, the best FOUR are
now res392 + intensity392 + calibrated + lr4e-5 at 0.8625, against 0.8612 for the four that scored
0.918. So the rate member displaces `sliceattn` from the subset rather than joining it, and the offline
gain is +0.0013 — inside the floor, which is why the board is being asked rather than told.


`strongest(n)` sorts members by what each scores ALONE, which is the only ordering a package can offer
by itself and a weak one for an ensemble. Measured 2026-08-16 on the 909-study fold-0 holdout, over the
six members this package mounts plus two more from the cache:

    subset                                                            holdout
    res392 + calibrated              (= strongest(2), LB 0.919)        0.8581
    strongest(4) by solo holdout                                       0.8588
    res392 + intensity392 + calibrated + sliceattn                     0.8612
    the same four plus extended_ws                                     0.8610
    all six mounted members                                            0.8603

So the peak is a NAMED four, and the solo sort misses it by 0.0031 — it keeps `extended_ws` (solo
0.8462) over `intensity392_ws` (solo 0.8459) and loses more than the difference between them. That is
the whole content of this kernel: `KernelRun(names=...)`, and a package that refuses if a name is not
mounted rather than quietly entering with three members.

The ceiling on any cleverer combine was measured at the same time and is not there: the per-label
ORACLE — best member per label, using the answers — scores 0.8590 against the flat average's 0.8588,
and the split-half estimate of per-label selection peaks at 0.8603 against the same average. Per-label
weighting is worth nothing here, which is why this entry is a plain rank-average of a chosen few.

RE-SCORED 2026-08-17 over thirteen same-split checkpoints rather than nine, in case a wider pool moved
the named four. It does not move them usefully: the best subset over thirteen is a FIVE at 0.8617, and
it wants `deep24` — a member this package does not mount, whose solo 0.8352 is BELOW the pool's floor,
and which joint selection nonetheless takes at every size from four up while never taking `crop80` at
0.8345. Adding it buys 0.0005 over the four named below, an eighth of the 0.0062 noise floor, for a
dataset push. The load-bearing half of that scoring is the shape, not the subset: the best subset falls
monotonically past five, down to 0.8547 flat over all thirteen. Naming few is not a thrift here, it is
the higher score.

Windows are disjoint because every member here reads the whole stack, for which overlap TTA is worth
exactly 0.0000 (knee-eeo). Rotation TTA was measured on the cache the same day: +0.0004 to +0.0020 for
three times the forward passes, so it is not bought either.
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

    chosen = ("res392_ws_s2026", "intensity392_ws_s2026", "calibrated_targets_s2026", "lr4e-5_ws_s2026")
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
