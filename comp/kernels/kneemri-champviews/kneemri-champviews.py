"""champviews — the champion's exact seating, served under SPATIAL TEST-TIME VIEWS. Nothing is reseated.

**WHY THIS AND NOT ANOTHER SEATING.** Four board rows now say the seating is not a lever.
`raptorweight_v1` moved the foreign SHARE by 2.3x for one unresolvable tick; `vjepaseat_v1` ADDED an
eighth seat — the most architecturally distant member the project owns — for -0.001; `holdswap_v1`
EXCHANGED a seat for the best-priced alternative a fold-0 swap table could name, for exactly 0.000;
`dark7_v1` added a seventh dark member to `dark6` for 0.937, flat. Reweight, add, swap, count: four
operations on WHO SITS, four flats, and knee-uco3's inert-seat reading now has four supports. What that
record leaves open, in its own words, is the levers that change what a member SEES or WHAT IT IS. This
entry is the first of those that costs no training run.

**THE MECHANISM.** `SpatialViews` runs each member three times — the identity view plus rotations and
shifts — and averages before the vote. It is not overlap TTA: `knee-eeo` measured overlapping windows at
exactly 0.0000 for a whole-stack member, because a member that already reads the whole stack has nothing
left to slide over. Rotating and shifting the PIXELS is a different perturbation, and it is the one
test-time axis a whole-stack member has not spent. Offline it prices at +0.0005 to +0.0021 solo and
+0.0007 on the best pair — under the 0.0053 paired floor, which is why it has never justified a slot on
its own number.

**WHAT JUSTIFIES THE SLOT IS THE SHAPE IT IS ATTACHED TO.** Every prior spatial-views measurement was
taken on a 0.919-era pair. This is the first time the axis is asked on the 0.944 shape, and it is asked
with NOTHING else moved: the same six members in the same order, the same `overlap=False`, the same
`LogitAverage`, the same raptor checkpoint at `FOREIGN_WEIGHT = 1/3`, the same rank-space join, the same
kit. One argument differs from `raptorweight_v1` — `Serving(views=SpatialViews())` — which is what makes
the read attributable.

**THE BUDGET IS THE REAL RISK, AND IT IS NAMED RATHER THAN ASSUMED.** Three passes per member against a
nine-hour ceiling, after the raptor pass has already spent from it. The kernel already hands the six what
is LEFT (`Budget(seconds=KERNEL_SECONDS - spent)`), and `Ensemble.afford` gives up the weakest member
rather than the entry when the ceiling is threatened — which means the failure mode here is not a killed
kernel but a SILENTLY SMALLER ENTRY, exactly the `farseat_v1` defect that read as a clean flat for five
days on a member that never loaded. So this row is not written until `run_check` has read the kernel log
and confirmed a `N of M` census with N == M. If members were dropped, the row is scoped `excluded` and
the reading is about the budget, not about views.

PRE-REGISTRATION, written before the number, against the matched control `raptorweight_v1` = 0.944 and
`BOARD_RESOLVABLE = 0.0015`.

- **0.946 or above** — spatial views are a real lever at the top of the range AND the entry lands in the
  top ten (0.946 floor as of the 2026-08-31T18:08Z board read). Re-pin `comp/champion.json`. The next
  slot asks whether the gain survives at more views, not whether it exists.
- **0.9455** — a resolvable gain over the champion (+0.0015, exactly the floor). Keep views on for every
  subsequent entry; the decode axis is open and the seating axis stays closed.
- **0.9425 to 0.9455 exclusive** — FLAT. A three-pass test-time perturbation buys nothing on this pool,
  which is a REAL result and not a null: it says the pool's error is not the kind that a spatial jitter
  averages away, and taken with the four seating flats it means neither WHO sits nor HOW they are served
  moves this score. What is left is what a member IS. No further slot goes to test-time perturbation.
- **0.942 or below** — views COST. Suspect the budget FIRST, in this order: read the kernel log's member
  census (`run_check`), confirm six of six loaded, confirm no `afford` line gave a member up, and confirm
  the raptor table is present in the blend. Only after all four come back clean does a negative read as a
  fact about spatial views rather than about a truncated entry.

**WHAT NO BRANCH LICENSES.** Nothing here says anything about views on a fold-0 pool (every seat but one
is `--fold -1`, so no offline number exists or can exist for this entry), about the NUMBER of views (three
is `SpatialViews`'s default, held, never swept before the axis is shown to transfer), or about views on a
member that does not read the whole stack — where overlap TTA is not already known to be spent and the two
axes would confound.
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
    from kneemri.inference.test_time import SpatialViews  # noqa: PLC0415 - same
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
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=left, views=SpatialViews())
    ).run()
    ours = pd.read_csv(WORKING / "submission.csv")

    RaptorSeat(weight=FOREIGN_WEIGHT).blended(ours, theirs).to_csv(WORKING / "submission.csv", index=False)


if __name__ == "__main__":
    main()
