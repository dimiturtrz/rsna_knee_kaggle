"""The champion's weakest seat, given to the member built from two corpora and finer labels.

**ONE SEAT MOVES AND NOTHING ELSE.** This is `kneemri-votefit` — the current champion at 0.938, six
members served under `LogitAverage` with `overlap=False` — with `base_dense_ff` replaced by
`merged_blend_s2026` and every other seat, the vote rule, the kit and the serving config left alone.
`base_dense_ff`'s fold-0 twin is the weakest of the six on the fold-0 ruler (0.8512), so it is the seat
with the least to lose.

**WHAT THE ENTERING MEMBER IS.** `merged_blend_s2026` composes the two corpus mechanisms this project
has: the merged scheme (our studies plus Rijeka) and the finer report labels. It reads 0.8589 holdout —
a tie with `band_dense_long` and behind `small_batch6_s2026` at 0.8598, so it is NOT our best solo
member and nothing here claims it is. What earns it the seat is that it appeared in EVERY best subset
from two members to nine when the ten strongest cached members were searched exhaustively, and it moved
the best-subset score from 0.8696 to 0.8713 (+0.0017). Priced twice — once against roster twins
(+0.0020), once against the strong ten (+0.0017) — so the seat is not an artefact of a weak pool.

**THE SWAP MOVES TWO THINGS AT ONCE, AND ONE OF THEM IS ALREADY CONTROLLED.** The outgoing member is
`_ff` (all-data, `--fold -1`); the entering one is `_s2026` (fold-0). So this changes both the corpus
and the fold status of the seat. `knee-97ib` settled the second half on the board: a fold-0 seat reads
flat against its own `_ff` twin, which is why no refit was queued and none is allowed. Under that
result the fold status of ONE seat is worth ~0 board points, and what remains varying is the corpus.
If knee-97ib is wrong, this entry cannot separate the two — stated here rather than discovered later.

**THE SIZE, SAID PLAINLY.** +0.0017 offline against a transfer constant of ~+1.7 board points per
holdout point is roughly +0.003 board — one tick, inside the board's own scatter. votefit is 0.938 and
the leader is 0.952. This entry is not priced to close that and does not pretend to. It earns the day's
first slot because it is the best thing available to put there, and because the corpus axis has never
had a board number: every merged-corpus result this project owns is offline.

PRE-REGISTRATION, written before the number, against votefit's 0.938.

- **0.942 or above** — the corpus axis TRANSFERS, and transfers larger than the ruler predicted. Build
  merged counterparts for the other seats and re-serve; the merged scheme becomes a live lever rather
  than a member.
- **0.939-0.941** — as predicted, and inside the noise. `merged_blend` keeps the seat because nothing
  argues for giving it back, but the corpus axis is a member-sized effect, not a design lever, and no
  further slot goes to it without a bigger mechanism.
- **0.935-0.938** — flat or a small loss. Either the offline seat did not survive the test
  distribution, or the seat's `_ff` status was worth more than knee-97ib priced it at. The honest read
  is that we cannot tell which from one entry, and the next slot on this axis must be the
  `merged_blend_ff` refit that separates them — which is a 7 h card cost this entry is trying to avoid
  paying blind.
- **0.934 or below** — a real regression from a swap the offline ruler called an improvement, at the
  top of the range where the holdout and the board have already disagreed in SIGN once (0.8731 offline
  scored 0.932; 0.8712 scored 0.937). That would close member-selection-by-holdout for good, not just
  bruise it, and it retires the merged corpus as a seat-filler.

COST: one slot, zero card time, nothing retrained. The member already exists and is already mounted.
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

    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - the kit is on sys.path only after boot
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same

    # `merged_blend_s2026` stands where `base_dense_ff` stood. The order is the champion's, unchanged,
    # so the diff against `kneemri-votefit` is one line and reads as one.
    chosen = (
        "final_ws_e12",
        "calib_span_ff",
        "grade_ladder_ff",
        "merged_blend_s2026",
        "lr4e-5_ws_ff",
        "through_dense_ff",
    )
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage())).run()


if __name__ == "__main__":
    main()
