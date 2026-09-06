"""onegroup — the champion entry, one dial moved: every seat served out of ONE read of the corpus.

**WHAT IS BEING PRICED.** Not who sits and not how much any seat counts — `raptorweight_v1`'s exact
seven, its exact weights, its exact vote. The one change is the READING. Our six seats span three read
groups today (20 slices at band 0.1-0.9, 12 at that band, 12 at the historic 0.2-0.8), and a group is
one full decode of the corpus: the champion's own Kaggle log decodes the same three visible studies
three times over — `decode 4s`, then `2s`, then `2s` — and projects 7.6 h of a 9 h cap. Served as one
group at 12 slices on 0.1-0.9, a study opens 12 files instead of 44.

**WHY IT IS AN EFFICIENCY INSTRUMENT AND NOT A SEATING.** The competition's efficiency track scores
`AUC/(0.5 - 0.95) + Runtime/32400`, so one AUC point is worth 72,000 s and the accuracy half is the
term to protect. This entry protects it by construction: the same weights, the same seats, only fewer
files opened per study. It also removes a TIMEOUT risk that has nothing to do with efficiency — 7.6 h
of a 9 h cap is not margin.

**PRICED OFFLINE FIRST, on the fold-0 twins.** Five of the champion's six members are `--fold -1` and
hold nothing out, so the price was taken on their fold-0 twins (`calib_span_s2026`, `grade_ladder_s2026`,
`through_dense_s2026`, `lr4e-5_ws_s2026`, `merged_blend_cf`) over 909 studies, through the submission
path, under the shipped `LogitAverage`:

    native reading (control)          0.8706
    one group, 12 slices, 0.1-0.9     0.8684      cost 0.0022

`RankAverage` agrees to the third decimal (0.8700 -> 0.8678), so the delta is not a vote artefact. The
board resolves an offline delta of about 0.0045 through the +1.55 `board_transfer`, so a 0.0022 cost is
BELOW what the accuracy board can see — which is the whole claim being tested.

**WHAT WOULD REFUTE IT.**

- **0.943-0.944** — the reading is free at the board's resolution and the champion has been paying
  three decodes for nothing. The one-group reading becomes the default for every later entry.
- **0.938-0.942** — the offline 0.0022 under-states the board cost. Either the twins are not a ruler
  for the dark six, or the foreign seat interacts with the reading; both are readable next.
- **below 0.938** — a shallower/narrower read costs real accuracy on the test distribution, the
  efficiency track has to buy its runtime somewhere else (the file-level memo, which changes no pixels
  any member sees), and the champion's reading stands.
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


SERVED_SLICES = 12
"""The depth every seat is read at. The champion's own majority depth, not a new value: four of its six
seats already read twelve, so this deepens nothing and shallows the 20-slice group only."""

SERVED_BAND = (0.1, 0.9)
"""The band every seat is read at. The WIDER of the two the champion serves, so no member loses slices
it was fitted to see — the historic 0.2-0.8 sits strictly inside it."""


def main() -> None:
    boot()

    import time  # noqa: PLC0415 - the bootstrap block above is byte-identical across kernels

    import pandas as pd  # noqa: PLC0415 - the kit is on sys.path only after boot
    import torch  # noqa: PLC0415 - same

    from core.mounts import Mounts  # noqa: PLC0415 - same
    from kneemri.device import VisibleCards  # noqa: PLC0415 - same
    from kneemri.evaluation.raptor_model import RaptorMember  # noqa: PLC0415 - same
    from kneemri.inference.budget import KERNEL_SECONDS, Budget  # noqa: PLC0415 - same
    from kneemri.inference.ensemble import Serving, SharedRead  # noqa: PLC0415 - same
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
    # THE ONE DIAL. `Serving.group` is applied in `Ensemble.served`, before `reads` groups the members,
    # so the three groups collapse to one DECODE rather than to one prediction after three decodes.
    KernelRun(
        names=chosen,
        overlap=False,
        serving=Serving(vote=LogitAverage(), budget=left, group=SharedRead(SERVED_SLICES, SERVED_BAND)),
    ).run()
    ours = pd.read_csv(WORKING / "submission.csv")

    RaptorSeat(weight=FOREIGN_WEIGHT).blended(ours, theirs).to_csv(WORKING / "submission.csv", index=False)


if __name__ == "__main__":
    main()
