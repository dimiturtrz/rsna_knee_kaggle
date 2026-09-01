"""corpusdose — the champion seven with the EXTERNAL-CORPUS dose raised from one seat to four.

WHY THIS AND NOT A BETTER MEMBER. The champion's own record prices the member-quality lane: improving
all six seats on the strongest measured axis at once moved the board +0.002 and the arithmetic there
ceilings that lane near 0.943 against a leader at 0.952. The stack lane closed the same week, 7 of 7
second terms landing below their shared control. What that record names as still open is external data,
and this project already owns members fitted on one - the Rijeka expert image reads, the `merged_*`
family - with exactly ONE of them seated.

THE SHAPE IS TAKEN FROM THE ALL-DATA DOSE, which is the only axis here that has ever been read as a dose
rather than as a one-off: one seat trained on the extra studies bought about +0.001, the next five about
+0.003 together, roughly +0.0006 a seat, and the sub-linearity was the finding. If the external corpus
behaves the same way, moving one seat to four predicts about +0.0018 - just at BOARD_RESOLVABLE = 0.0015,
which is why the dose is raised by three seats and not by one. A single swap predicts +0.0006 and would
be unreadable by construction; running it would be a slot spent on a number the ruler cannot see.

THE SWAPS, chosen to hold decode and fit-mode rather than to hold solo score:
  calib_span_ff  -> merged_span_ff   both DARK full-fits at the SPAN corner; the corpus is the variable
  final_ws_cf    -> merged_span_cf   0.8618 solo, the second-best single member this project owns
  lr4e-5_ws_cf   -> merged_dense_cf  the dense corner, replacing the near-twin of a seat that stays
The two departing `_ws_cf` members are the pool's whole-stack near-twins, which is where a seat can be
taken with the least structural loss - the pixel-distant (`through_dense_cf`) and objective-distant
(`grade_ladder_cf`) seats both stay, and so does `merged_blend_cf`, so external seats go 1 -> 4 of 6.

WHAT IT IS NOT. This is an INSTRUMENT, not a bid for the board. Count, vote (LogitAverage), overlap,
serving precision, the foreign seat and its 1/3 weight are the champion's untouched, so the single
moving quantity is how many of the six saw the external corpus. It cannot be scored offline at all -
four of the six are `--fold -1` and there is no comparison across that line - so the board is the only
ruler it has, and one read at this resolution is what it buys.

PRE-REGISTERED, against the champion's 0.944 on the same hidden set:
  >= 0.9455  THE CORPUS DOSE IS A LANE. Above the all-data slope's own prediction, which would mean
             external-corpus seats compound BETTER than extra in-domain studies did. Then the fastMRI
             and MRNet work is not a side errand, it is the plan, and the next arm trains on more of it.
  0.9455 > s >= 0.9435  AS PREDICTED AND SUB-LINEAR. The dose is real and behaves like the all-data one.
             This licenses more external seats but prices the whole lane at well under the gap to first,
             so it must be spent alongside another route, not instead of one.
  0.9435 > s > 0.9425  FLAT. Four external seats read the same as one. Then the corpus a member saw is
             not what its seat is worth, the `merged_blend_cf` tick was the plateau, and downloading
             more corpora is not licensed by anything measured here.
  <= 0.9425  THE SWAP COST MORE THAN THE CORPUS BOUGHT. Note this branch cannot separate the corpus
             from the three departing members; if it fires, the reading is about THESE swaps and the
             next question is which seat carried the loss, not whether external data works.
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

FOREIGN_WEIGHT = 1.0 / 3.0
"""The foreign side's share of the rank blend. `raptorseat_v1` shipped 1/7 = 0.1429 as a convention about
SEAT COUNT; this is a quality-weighted share of two SYSTEMS, named before the number and never fitted.

Below the bootstrap block rather than above it: the block opens with imports, and a constant in front of
them makes every one of them E402. The kernel that RAN carried it above and the ruff gate caught it after
the fact — the number is identical either way.
"""


def main() -> None:
    boot()

    import time  # noqa: PLC0415 - the bootstrap block above is byte-identical across kernels

    import pandas as pd  # noqa: PLC0415 - the kit is on sys.path only after boot
    import torch  # noqa: PLC0415 - same

    from core.mounts import Mounts  # noqa: PLC0415 - same
    from kneemri.device import VisibleCards  # noqa: PLC0415 - same
    from kneemri.evaluation.raptor_model import RaptorMember  # noqa: PLC0415 - same
    from kneemri.inference.budget import KERNEL_SECONDS, Budget  # noqa: PLC0415 - same
    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - same
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same
    from kneemri.kaggle.raptor_run import WORKING, RaptorEntry  # noqa: PLC0415 - same
    from kneemri.kaggle.raptor_seat import RaptorSeat  # noqa: PLC0415 - same

    # The raptor member runs FIRST and into a scratch directory, because both producers write a file
    # called `submission.csv` and the one that runs last is the one Kaggle would otherwise score.
    scratch = WORKING / "raptor"
    scratch.mkdir(parents=True, exist_ok=True)
    started = time.time()
    mounts = Mounts()
    device = str(VisibleCards().device())
    member = RaptorMember.loaded(RaptorEntry.weights(mounts=mounts), device)
    theirs = RaptorEntry(mounts.one("sample_submission.csv", "competition data"), working=scratch).run(member, device)
    # The six that follow load onto the same T4. Dropping the reference is not enough on its own —
    # torch keeps the freed blocks in its caching allocator, and a seventh model's worth of them is
    # exactly the margin an OOM would eat, in a kernel whose whole point is running both.
    del member
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    chosen = (
        "merged_span_cf",
        "merged_span_ff",
        "grade_ladder_cf",
        "merged_blend_cf",
        "merged_dense_cf",
        "through_dense_cf",
    )
    # What the raptor pass cost, handed to the six so their nine-hour budget is the SESSION's and not
    # a fresh one. Without it both producers aim at the same ceiling and the kernel is killed at the
    # real one with no submission — a budget defeated by its own assumption that it starts at zero.
    spent = time.time() - started
    logging.getLogger(__name__).info(f"the raptor pass cost {spent:.0f}s; the six get what is left")
    left = Budget(seconds=max(KERNEL_SECONDS - spent, 0.0))
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=left)).run()
    ours = pd.read_csv(WORKING / "submission.csv")

    RaptorSeat(weight=FOREIGN_WEIGHT).blended(ours, theirs).to_csv(WORKING / "submission.csv", index=False)


if __name__ == "__main__":
    main()
