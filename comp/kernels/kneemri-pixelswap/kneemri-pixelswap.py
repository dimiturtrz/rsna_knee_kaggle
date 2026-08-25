"""The champion five with `base_dense` REPLACED by `through_dense` — the pixel axis, at a fixed count.

Every member the champion holds reads the SAME pixels: an axial slice replicated into three identical
input channels. `through_dense_s2026` is the one checkpoint in the pool of twenty-four for which that is
false — it fills the three channels with slices *i-1, i, i+1*, so its trunk sees a different input array
than any incumbent before a single weight is applied. Nothing else about it moved: same recipe as
`band_dense`, same band, same crop, same rate, same schedule.

WHY A SWAP AND NOT A SIXTH. The count axis is under test right now by `sixstrong` and `sixfresh`, and a
six-member entry here would land its answer on top of theirs and read neither cleanly. Holding the count
at five makes this entry attributable to the pixel change alone. It is also where `through_dense` prices
best: **+0.0010** swapping for `base_dense`, against **+0.0008** as a sixth.

THE OFFLINE READING, UNROUNDED. Rank-averaged over the 909-study fold-0 holdout the champion five read
**0.8690**; this five reads **0.8700**. **+0.0010** — a sixth of the 0.0062 three-seed noise floor and
below the **+0.0015** `BOARD_RESOLVABLE` line. On its own that is not a reason to spend a slot. It is,
however, the LARGEST move any of the twenty-four candidates makes in any seat: every mounted member was
priced as an add and as a swap and the whole distribution spans +0.0002 to -0.0019, so this is five times
the best the mounted pool offered and the only candidate that moved the mix at all.

THE ARGUMENT I FIRST WROTE HERE WAS ALREADY REFUTED IN THIS REPO, AND THE RECORD IS KEPT RATHER THAN
EDITED AWAY. I justified this entry by knee-nas5's measured **2.4x** under-pricing of diversity, carrying
+0.0010 offline to ~+0.0024 on the board. `seat_value`'s own docstring retires that move under knee-c0u9:
the 2.4x came off the DECODE axis, by collapsing three members into one pixel group, and it does not
generalise to a member SWAP — two entries were fired on exactly that theory and both were refuted. What
three paired entries measure instead is that the holdout RANKS correctly and errs only in magnitude, in
the direction of understating what a WEAK member costs. There is no measured licence to inflate a
positive.

AND THE POSITIVE SATURATES (knee-825t). Six entries now pair an offline joint-holdout score with a board
score, and the chain rises over the first ~0.003 of offline spread and then stops. `longseat` carried
**+0.0012** — the largest single move in a sweep of all 56 same-split checkpoints, six times the
second-best — and TIED the board. This entry reads +0.0010, below that. The precedent for a swap of this
size is a tie, not a tick.

SO THIS IS BUILT AND HELD, NOT QUEUED. Under the repo's settled rule a candidate earns a slot only by
clearing `BOARD_RESOLVABLE` (+0.0015), and +0.0010 does not. It stays on disk because the pixel axis is
the one axis no board entry has ever tested and because `sixstrong`/`sixfresh` may return an answer that
changes what a fifth or sixth seat is worth; it is fired only if that answer supplies a reason the
offline number cannot.

PRE-REGISTERED. **0.932 or above** says the pixel axis pays and pays past the champion: the next entries
push depth harder — a wider window needs a volume-pretrained trunk (knee-jucq), and that becomes the
priority lane rather than a probe. **0.931** is a tie: through_dense is a free substitute for base_dense,
the diversity correction did not materialise at the input, and the pixel axis is neutral rather than
live. **0.930 or below** says the swap cost, which retires the input-diversity argument for THIS window
size and leaves it open only for a window wider than three — the one thing a 2D-pretrained three-channel
stem structurally cannot deliver, and therefore the cleanest possible motivation for the volume trunk.
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
        "band_dense_s2026",
        "calib_span_s2026",
        "depth_regions_narrow_s2026",
        "band_dense_long_s2026",
        "through_dense_s2026",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
