"""stackvote — the champion seven, combined by a LEARNED rule. The first fitted combiner this project ships.

**WHAT IS BEING PRICED, AND WHY IT IS NOT `aucvote` AGAIN.** Every entry on this board so far, champion
included, combines its seats with a rule chosen in advance: average the logits, average the ranks, weight
by each seat's own AUC (`aucvote_v1`, submitted 17:29Z today). All of them price a seat by how good it is
ALONE. `StackedLogit` prices a seat by what it ADDS GIVEN THE OTHERS — a ridge regression of the twelve
targets on the members' logits, one weight per (seat, finding), fitted on 909 holdout studies.

**THE MECHANISM, and it is a different one.** A quality-weighted rule cannot go below zero: every seat's
AUC is above chance, so every seat gets a positive share on every finding, and a seat whose errors are
already covered by the pool still adds its noise. The fitted weights are free to subtract, and they do —
`grade_ladder` lands at **-0.0103 on Lateral OA**, and near zero (0.00003) on Medial OA. If the pool's
redundancy is structured, that is the structure, and no AUC-derived weighting can represent it. This is
the only mechanism by which this entry can beat `aucvote` rather than repeat it, and it is why this is not
a re-run of a rule the board has already priced.

**WHAT IT MEASURED OFFLINE, STATED BEFORE THE READ.** On the five fold-0 twins: StackedLogit 0.8755,
LogitAverage 0.8745, RankAverage 0.8742. **+0.0010** — beats both, and under our 0.0062 holdout floor. At
the +0.98 board-per-holdout-point slope that is +0.001 on the board, under `BOARD_RESOLVABLE = 0.0015`. So
a flat read is the EXPECTED outcome and this entry is not a favourite to beat 0.944. It is here because
the learned-combiner axis is one of only two routes `comp/champion.json` names to 0.950 and it has never
been read on the board at any seating.

**THE SAME TWIN LIMIT AS `aucvote`, INHERITED WHOLE.** See `STACK_WEIGHTS`. Five rows are fitted on fold-0
twins; row 1 (`final_ws_cf`) is the pool mean because its ancestor is the roster's dark member. So a flat
read cannot separate "a learned combiner buys nothing here" from "the twin-derived weights are too noisy
to act on" — and `aucvote_v1`'s own band, read first, constrains which of those is live.

PRE-REGISTRATION, written before the number, against the champion's 0.944 and `BOARD_RESOLVABLE = 0.0015`.

- **0.9455 or above** — a learned combiner is a real lever at the shipped seating, which would be the
  first thing to move the board since the member axis closed. Re-pin `comp/champion.json`, and the next
  arm is an honest fold-0 twin for `final_ws` so the dark row stops being a mean.
- **0.9425-0.9455** — FLAT, which is what the offline +0.0010 predicts. Taken with `aucvote_v1`, the
  COMBINER axis closes: fixed rules and fitted rules both read flat at the seating we ship, and the +0.002
  offline gain does not survive transfer. No further slot goes to a vote rule of any kind.
- **0.940 or below** — the fitted weights actively MISPRICE the seats. Read it against `aucvote_v1` before
  concluding anything: if that entry also dropped, the fault is the twin assumption both share, not the
  fitting. If `aucvote` held and this dropped, then fitting on 909 studies overfits the pool and a
  shared-weight (one per member) combiner is the cheaper form to try.

Cost: one slot, no card time. Pool, kit and members are the champion's. The kit must be REBUILT for this
entry — `kneemri/inference/stacked.py` is new and no shipped kit contains it.
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

STACK_WEIGHTS: tuple[tuple[float, ...], ...] = (
    (
        0.024731,
        0.020174,
        0.029232,
        0.028185,
        0.030311,
        0.025664,
        0.029156,
        0.032183,
        0.032213,
        0.034278,
        0.030010,
        0.017839,
    ),  # final_ws_cf — pool mean, see below
    (
        0.018384,
        0.017078,
        0.042821,
        0.030424,
        0.048699,
        0.049696,
        0.054976,
        0.012609,
        0.034163,
        0.031370,
        0.026487,
        0.019710,
    ),  # calib_span_ff
    (
        0.044931,
        0.010899,
        0.029488,
        0.007291,
        0.000034,
        -0.010349,
        0.027388,
        0.033805,
        0.020745,
        0.024530,
        0.029466,
        0.009021,
    ),  # grade_ladder_cf
    (
        0.009825,
        0.050228,
        0.013230,
        0.035027,
        0.040348,
        0.036612,
        0.024287,
        0.040960,
        0.036844,
        0.056051,
        0.019263,
        0.031295,
    ),  # merged_blend_cf
    (
        0.017653,
        0.018922,
        0.032211,
        0.025537,
        0.022119,
        0.031618,
        0.012561,
        0.032455,
        0.020729,
        0.012724,
        0.033591,
        0.009803,
    ),  # lr4e-5_ws_cf
    (
        0.032860,
        0.003745,
        0.028408,
        0.042648,
        0.040353,
        0.020743,
        0.026566,
        0.041088,
        0.048583,
        0.046713,
        0.041241,
        0.019365,
    ),  # through_dense_cf
)
"""FITTED per-(seat, finding) weights over member logits, in `chosen` order over `core.findings.TARGETS`.

The difference from `kneemri-aucvote`, which is the whole point of this kernel: those weights were each
seat's own holdout AUC, so a seat was priced by how good it is ALONE. These are fitted jointly — a ridge
regression of the twelve targets on the five twins' logits — so a seat is priced by what it adds GIVEN the
others. That is a different quantity, and it can be negative: `grade_ladder` gets -0.0103 on Lateral OA,
meaning the pool is better with its opinion there subtracted. No AUC-derived rule can express that, and it
is the only mechanism by which this can beat `aucvote` rather than repeat it.

Produced by `python -m kneemri.evaluation.stack_fit --holdouts <models root> --only calib_span_s2026
grade_ladder_s2026 merged_blend_cf lr4e-5_ws_s2026 through_dense_s2026 --emit-weights`, fitted on all 909
holdout studies with the ridge penalty chosen on a nested split inside them. Fitted on everything rather
than half because these are shipped, not reported: the kernel is judged on the competition test set, which
no fit here can touch.

THE SAME TWIN ASSUMPTION `aucvote` STATES, inherited whole. Five of the six seats are `--fold -1` and have
no holdout predictions, so the weights come from their fold-0 twins, and a `_cf` seat is its twin plus
`--fold -1 --ema --calibrated-targets`.

ROW 1 IS THE POOL MEAN, not a fit. `final_ws_cf` descends from `final_ws_e12`, the roster's DARK MEMBER —
"no offline score exists or can exist" — so no twin exists to regress on. The mean row is the neutral
choice for the one seat that cannot be read.

WHAT IT MEASURED OFFLINE, on the five twins alone: 0.8755 against LogitAverage 0.8745 and RankAverage
0.8742. +0.0010 — under the 0.0062 holdout floor and under BOARD_RESOLVABLE at the +0.98 transfer slope.
That is stated up front because it means a flat board read is the EXPECTED outcome, and this kernel earns
a slot as the first test of a LEARNED combiner at the shipped seating, not as a favourite.
"""

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
    import numpy as np  # noqa: PLC0415 - same

    from kneemri.inference.stacked import StackedLogit  # noqa: PLC0415 - same
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
        "final_ws_cf",
        "calib_span_ff",
        "grade_ladder_cf",
        "merged_blend_cf",
        "lr4e-5_ws_cf",
        "through_dense_cf",
    )
    # What the raptor pass cost, handed to the six so their nine-hour budget is the SESSION's and not
    # a fresh one. Without it both producers aim at the same ceiling and the kernel is killed at the
    # real one with no submission — a budget defeated by its own assumption that it starts at zero.
    spent = time.time() - started
    logging.getLogger(__name__).info(f"the raptor pass cost {spent:.0f}s; the six get what is left")
    left = Budget(seconds=max(KERNEL_SECONDS - spent, 0.0))
    KernelRun(
        names=chosen,
        overlap=False,
        serving=Serving(vote=StackedLogit(np.asarray(STACK_WEIGHTS, dtype=np.float64)), budget=left),
    ).run()
    ours = pd.read_csv(WORKING / "submission.csv")

    RaptorSeat(weight=FOREIGN_WEIGHT).blended(ours, theirs).to_csv(WORKING / "submission.csv", index=False)


if __name__ == "__main__":
    main()
