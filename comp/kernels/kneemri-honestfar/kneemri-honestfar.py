"""honestfar — the champion's SERVING recipe with the champion's six members replaced by six honest ones.

WHAT IS BEING PRICED, AND IT IS NOT A RESEATING. `stackseat_v1` (0.944, the champion) is six of ours by
`LogitAverage` joined in rank space to the foreign `raptor_v4_swa` at weight 1/3. Five of its six are
`--fold -1`, fitted on all 4,406 studies, which is why `comp/board.jsonl` carries `offline: None` for it
and for every one of the 21 ensemble rows at board >= 0.940. D372's finding is that the record therefore
cannot price the leader gap at all: the highest rows holding BOTH an offline and a board number are
0.935-0.937, and any slope quoted at 0.944 extrapolates past the last resolved pair by the full width of
the band.

This kernel keeps the champion's serving EXACTLY — same `LogitAverage` over ours, same rank-space join,
same `FOREIGN_WEIGHT = 1/3`, same raptor checkpoint, same budget hand-off — and moves one thing: the six.
They become the six honest members `kneemri-honestsix` ships, every one of which holds out all 909 study
IDs. So this entry has an honest offline number for its own side, which no entry at the champion's score
has ever had.

THE SIX, and the rule that chose them. Not a search. The 133 packaged members passing all three honesty
filters (`epoch >= 0`; `studies` vector identical to the canonical holdout909; binarised `truth`
identical cell-for-cell), ranked by SOLO macro AUC, first six:

    clean_calib_s2026          0.8649
    chainfour_s2026            0.8648
    stack_increments_s2026     0.8642
    stack_increments_base_s2026 0.8635
    calib_dense_hold_s2026     0.8634
    seat_clean_distill_s2026   0.8634

LogitAverage of the six reads **0.8737 macro AUC @holdout909, cut-free** — +0.0088 over its own best
member, and flat in seat count (five seats 0.8738, seven 0.8735), which is what says it is not a lucky
cut. The no-search rule is doing real work: a maximum picked by combination search on this same 909
carries optimism, measured twice tonight. Per-column member routing read +0.0112 in sample and
+0.0008 +/- 0.0033 across 20 half-splits — zero. Greedy forward seat selection DOES survive the same
protocol, but at +0.0013 +/- 0.0013 at k=8 against +0.0031 in sample, with the selected sets churning at
mean pairwise Jaccard 0.22 across 40 runs. A greedy seating is therefore never named in a kernel here:
the small edge lives in the SEARCH, not in any particular set.

`oof_teacher_s2026` is NOT among the six despite reading 0.8685, the top of the honest pool. Its arm and
`seat_clean_distill.sh` are identical `run_train` blocks apart from `--teacher-targets` — leaky
`teacher_oof_corpus.npz` against clean `teacher_clean.npz` — and they read 0.8685 against 0.8634. The
fold-0 leak transfers at +0.0051, which is the entire margin by which that member led. The honest pool
max is 0.8649.

WHY THIS IS A CONTENDER RATHER THAN AN INSTRUMENT. D370's class table says ensemble seat swaps have gone
0 for 18 and fixed-count reseating at >= 4 seats 0 for 57, so swapping seats inside the champion is a
closed class. This is not that move: it replaces the whole of our side at once, with members chosen by a
different rule, and keeps the foreign system that the champion's 0.944 is partly built on. D373 measured
why that could pay — over 45,451 member pairs the mean pairwise Spearman of column-centred AUC profiles
is +0.0144 with within-member split-half reliability +0.4965, so the plateau is 303 different functions
failing on different columns and the ensemble headroom implied by that was never spent by the solo
plateau.

PRE-REGISTERED against the champion 0.944 and against `kneemri-honestsix`, which carries the identical
six WITHOUT the foreign seat and goes to the board in the same UTC day:

  >= 0.945   the champion is beaten by an all-honest our-side, and the offline ruler orders at the
             operating point after all. Every arm in the tree becomes priceable against holdout909
             again, which is the single most valuable thing that could happen to this project tonight.
  0.940-0.944  at the champion. The `--fold -1` seats were buying nothing the honest six cannot buy,
             and the top band gains its first resolved (offline, board) pair.
  0.933-0.939  below the champion but at or above honestsix's own level: the foreign seat is carrying
             its share and the gap to 0.944 is the fold -1 fitting, priced for the first time.
  <= 0.932   at or below `stackfit_s2026`'s solo board score. Then the `--fold -1` seats carry real
             board value rather than only leakage, and "honest members only" is a measurably worse
             seating. That is the most useful branch of the four and it is named here so it cannot be
             re-read afterwards as a disappointment.

THE PAIR IS THE POINT. honestfar and honestsix differ by exactly one thing — the foreign seat at 1/3 —
and share their six members and their serving. Whatever the two score, their DIFFERENCE is a clean read
of what the foreign system contributes at the operating point, which the record has never isolated
because every foreign-seat entry also changed our side.

NO ATTRIBUTION IS CLAIMED for the six individually. The seating rule is stated so the number can be
reproduced, not so any member can be credited.
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
        "clean_calib_s2026",
        "chainfour_s2026",
        "stack_increments_s2026",
        "stack_increments_base_s2026",
        "calib_dense_hold_s2026",
        "seat_clean_distill_s2026",
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
