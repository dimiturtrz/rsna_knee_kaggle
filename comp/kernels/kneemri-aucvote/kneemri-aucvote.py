"""aucvote — the champion seven, one dial moved: the six vote by PER-FINDING AUC, not by logit average.

**WHAT IS BEING PRICED.** `raptorweight_v1` (board 0.944, the pinned champion) combines its six with
`LogitAverage` — every seat counts the same on every finding. This entry replaces that one object with
`LabelQualityRank`, which ranks each member's column and weights it by how far that member's AUC on THAT
finding sits above chance. Nothing else moves: same six in the same order, same `overlap=False`, same kit,
same raptor checkpoint at `FOREIGN_WEIGHT = 1/3`, same rank-space join. One argument differs.

**THE MECHANISM.** The pool is not uniformly good across the twelve findings and the spread is not small:
over the fold-0 twins, MCL runs 0.776-0.820 while Baker's runs 0.901-0.917, and the member that is best on
one is not best on the other (`merged_blend_cf` leads MCL by 0.044 over `through_dense_cf` and trails
`grade_ladder_cf` on ACL by 0.025). A uniform average spends those differences. Weighting per finding
converts them, IF they are stable rather than holdout scatter.

**THE PRIOR SAYS THIS IS NULL, AND THAT PRIOR IS OFF-DISTRIBUTION.** Four offline measurements put
`LabelQualityRank` at +0.0002 to +0.0011 over `LogitAverage` — split-half twice, seats-per-decode twice —
and three combiners sat inside 0.0002 of each other on two further pools. Every one of those was taken on
a pool that could be scored offline, which means a pool of fold-0 members. The champion is six of six
`--fold -1`: no offline ruler can see it, and none of the four nulls contains it. This is the first reading
of the vote rule AT THE SEATING WE ACTUALLY SHIP.

**WHERE THE WEIGHTS COME FROM, AND THE HONEST LIMIT.** See `PER_LABEL`. Five rows are fold-0 twins, one
row (`final_ws_cf`) is the pool mean because its ancestor is dark. So the weights are an ESTIMATE of the
seats' per-finding shape, not a measurement of it. If the entry reads flat, that is consistent with both
"the vote rule is inert" and "the twin weights are too noisy to be worth acting on", and this entry alone
cannot separate them.

PRE-REGISTRATION, written before the number, against the champion's 0.944 and `BOARD_RESOLVABLE = 0.0015`.

- **0.9455 or above** — per-finding weighting is a real lever at the shipped seating and the four offline
  nulls were an artefact of fold-0 pools. Re-pin `comp/champion.json`; the next question is whether honest
  weights (a fold-0 twin for `final_ws`) are worth a card arm.
- **0.9425-0.9455** — FLAT. The vote rule is closed on the board as well as offline, this time at the
  seating that matters, and no further slot goes to a combiner.
- **0.940 or below** — the twin-derived weights actively MISPRICE the seats. That is a reading about the
  weights, not about weighting: it would say a `_cf` seat's per-finding shape does not survive
  `--fold -1 --ema --calibrated-targets`, which is a fact worth having about every number this project
  infers from a twin.

Cost: one slot, no card time. The pool, the kit and the members are the champion's, already built and
already mounted.
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

PER_LABEL: tuple[tuple[float, ...], ...] = (
    (0.8793, 0.7939, 0.8971, 0.8525, 0.8775, 0.8135, 0.8460, 0.8525, 0.8632, 0.9089, 0.8667, 0.8188),
    (0.8706, 0.7898, 0.9009, 0.8505, 0.8800, 0.8218, 0.8590, 0.8409, 0.8632, 0.9014, 0.8648, 0.8113),
    (0.8952, 0.7934, 0.9012, 0.8557, 0.8706, 0.8041, 0.8518, 0.8572, 0.8533, 0.9046, 0.8734, 0.8177),
    (0.8702, 0.8195, 0.8891, 0.8552, 0.8808, 0.8175, 0.8369, 0.8533, 0.8735, 0.9174, 0.8588, 0.8440),
    (0.8706, 0.7908, 0.8983, 0.8460, 0.8763, 0.8178, 0.8392, 0.8542, 0.8546, 0.9076, 0.8685, 0.8066),
    (0.8897, 0.7760, 0.8959, 0.8551, 0.8800, 0.8063, 0.8431, 0.8569, 0.8713, 0.9134, 0.8678, 0.8147),
)
"""Per-finding holdout AUC for the six seats, in `chosen` order, over the twelve `core.findings.TARGETS`.

WHERE EACH ROW COMES FROM, because five of the six seats are `--fold -1` and cannot have a holdout number
of their own. Rows 2-6 are their fold-0 twins' `.holdout.npz` records, read over the same 909 studies every
member in this project is ranked on: `calib_span_s2026`, `grade_ladder_s2026`, `merged_blend_cf` (fold 0,
so this row is the seat itself), `lr4e-5_ws_s2026`, `through_dense_s2026`. A `_cf` seat is its twin plus
`--fold -1 --ema --calibrated-targets`, so the twin's per-finding SHAPE is the best available estimate of
the seat's and its LEVEL is not claimed.

ROW 1 IS THE POOL MEAN, not a measurement. `final_ws_cf` descends from `final_ws_e12`, which the roster
calls THE DARK MEMBER — "no offline score exists or can exist" — so no fold-0 twin was ever trained. A mean
row weights that seat at the pool average on every finding, which is the neutral choice: it neither
promotes nor demotes the one seat we cannot read.
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

    import numpy as np  # noqa: PLC0415 - same
    import pandas as pd  # noqa: PLC0415 - the kit is on sys.path only after boot
    import torch  # noqa: PLC0415 - same

    from core.mounts import Mounts  # noqa: PLC0415 - same
    from kneemri.device import VisibleCards  # noqa: PLC0415 - same
    from kneemri.evaluation.raptor_model import RaptorMember  # noqa: PLC0415 - same
    from kneemri.inference.budget import KERNEL_SECONDS, Budget  # noqa: PLC0415 - same
    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - same
    from kneemri.inference.vote import LabelQualityRank  # noqa: PLC0415 - same
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
        serving=Serving(vote=LabelQualityRank(np.asarray(PER_LABEL, dtype=np.float64)), budget=left),
    ).run()
    ours = pd.read_csv(WORKING / "submission.csv")

    RaptorSeat(weight=FOREIGN_WEIGHT).blended(ours, theirs).to_csv(WORKING / "submission.csv", index=False)


if __name__ == "__main__":
    main()
