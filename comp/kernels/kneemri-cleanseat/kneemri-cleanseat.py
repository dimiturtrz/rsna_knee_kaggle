"""cleanseat — the first seating this project has ever chosen on a ruler that is not `report_labels_v2`.

**WHAT IS BEING PRICED, AND IT IS NOT A RESEATING IN THE USUAL SENSE.** knee-urvf's standing result is
that every operation on the champion's SEATING is inert — seven board flats, each a permutation, a
weight, a count or a refit. Those seven were all chosen by the same selector on the same ruler. D340
measured that ruler wrong by +0.0010 to +0.0140 with a MEMBER-DEPENDENT sign, which means seat selection
has been optimising partly against the ruler's own error term. This entry re-runs the selection on a
corrected ruler and ships what it returns. The six that come back share **zero seats** with the
champion's six — the first time a selection has produced a different set rather than a permutation of
the same one.

**THE CASE IS THE RULER, AND NOTHING ELSE.** D342's greedy reads the six at **0.8843 on the corrected
ruler (T1)** against the T0-era selection's 0.8817, a gap of **+0.0026**. On the OLD ruler (T0) it reads
0.8763 against 0.8773 — it does NOT beat the T0 selection on T0. So no claim is made that this is a
better ensemble in general; the claim is that a ruler with a measured, mechanism-named bias removed
prefers it, and that the old ruler cannot tell the difference. D343 rescored the gap with 94 blind
human adjudications substituted (T1a: +0.0026) and again on a one-sided correction (+0.0025). Three
correction variants, one answer.

**EVERY NUMBER HERE IS A STAND-IN AND IS LABELLED AS ONE.** 94 of the 469 adjudication cells are read;
375 are not. Until all ten chunks land, the corrected ruler is a stand-in for the adjudicated one and
every score above inherits that word.

**LICENCE-CLEAN BY CONSTRUCTION, AND THE COST IS RECORDED.** The unconstrained greedy returned 0.8850
with two seats that cannot be distributed — Rijeka is CC BY-NC-ND (a fitted member is a derivative),
NYU's fastMRI DUA forbids distributing derived files, MRNet's Stanford agreement bars adaptations.
Re-run over the 277 licence-clean members of the 312-member pool, the joint lands at 0.8843. **The cost
of honesty on this axis is −0.0007**, which is inside every floor we have. That number belongs in the
ledger permanently: the unconstrained edge was seats that could never have shipped.

**WHAT DOES NOT MOVE.** The raptor member, its checkpoint, its weight at 1/3, `overlap=False`,
`LogitAverage`, the rank-space join by study, the budget hand-off. The foreign side is the champion's,
unchanged, so a board delta is attributable to the six and to nothing else.

PRE-REGISTRATION, written before the number, against the champion's 0.944 and `BOARD_RESOLVABLE = 0.0015`.

- **0.9455 and above** — the corrected ruler carries to the board, and seat selection on the v2 ruler
  was fitting its error term. Then EVERY closed seating result in knee-urvf was measured on a broken
  instrument and the whole axis re-opens, which would be the largest recovery this project has made.
  Re-pin `comp/champion.json`; finish the 469 immediately; re-run the greedy on the adjudicated ruler.
- **0.9425 to 0.9455** — FLAT, and the eighth. Then the ruler correction is real offline and does not
  transfer, which is a reading about the 909 holdout's distribution and not about the correction —
  D277's finding is that method ranking does not survive a cohort change. It does NOT retire the label
  axis: D343's 12.1% false-positive rate in `report_labels_v2.csv` is a TRAINING-side finding that this
  entry does not test at all, since all six were fitted on v2.
- **Below 0.9425** — check `run_check`'s census FIRST; a dropped member and a harmful set look identical
  from the board. If the census reads 6 of 6, the corrected ruler is anti-correlated with the board and
  that is a harder and more useful result than a flat.
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
"""The foreign side's share of the rank blend, identical to the champion's and to `cnnseat`'s. It is held
fixed on purpose: this entry moves the six seats and nothing else, so a board delta has one cause."""


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

    # D342's licence-clean greedy over the 277 distributable members, in the order the greedy added
    # them. Five are DINOv2-small at 20 or 12 slices; the last is the 48-epoch resnet50d, which is the
    # only member here whose encoder mounts from `dimiturnt/resnet50d-224`. It LOST its own branch (the
    # pre-registration asked for long − short ≥ 0.015 and it came back −0.0067) and is rostered anyway;
    # the greedy chose it for the joint, not for its solo, which is the whole argument for diversity the
    # owner has made repeatedly.
    chosen = (
        "stack_increments_s2026",
        "chainfour_s2026",
        "labelwise_extended_s2026",
        "blendhold_s2026",
        "stack_increments_base_s2026",
        "resnet_224_long_s2026",
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
