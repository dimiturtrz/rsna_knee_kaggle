"""labelseat — the champion six plus the one seat the offline ruler calls DISTINCT, shipped as an INSTRUMENT.

**THIS IS AN INSTRUMENT, NOT A PREDICTED GAIN, AND THE SUBMISSION MESSAGE SAYS SO.** Nothing on disk is
mechanism-backed to beat 0.944 today. The standing rule is that a slot with nothing mechanism-backed
behind it goes to an instrument with the instrument named, and this is the instrument: the offline ruler
has now stated IN WRITING that it cannot resolve this seat, and the board is the only ruler that can.

**WHAT IS BEING PRICED.** One seat, added. `seat_value --against cleanseat_v1` (roster of 6 at 0.8755,
cut 0.5, 2026-09-12) prices three labelwise candidates as a seventh seat:

    labelwise_depth24_s2026   +0.0008  [+0.0001, +0.0017]  DISTINCT  (rank 0.8985, js 0.0118, maha 3.907)
    labelwise_drop_s2026      +0.0006  [-0.0001, +0.0012]  TWIN      (rank 0.9219)
    distill_stack_clean_s2026 -0.0002  [-0.0007, +0.0003]  TWIN      (rank 0.9477)

One of three bands excludes zero and it is the DISTINCT one. The tool's own gate is quoted rather than
hidden: *0 of 3 clear +0.0040, under which this ruler has never been shown to pick a board tick.* So the
offline number does not license this entry — it only says which of the three is worth a slot.

**WHY THE SEAT IS NOT ANOTHER FLAT RESEATING (knee-urvf).** The standing result is that every operation
on the champion's SEATING is inert: a distant seat read 0.943/0.943/0.942, a swap read 0.944 dead flat
(`holdswap_v1`), and three offline candidates priced +0.0002/+0.0005/+0.0000 on one day. Every one of
those was a move WITHIN the pooling and fit regime the pool already held. The champion carries NO
labelwise member — all six pool slices with `max` or `attend`. `--slice-pooling labelwise` gives each of
the twelve findings its own slice-attention gate, so the member disagrees with the six about WHICH slice
carries a finding, not about how confident to be. That is the decorrelation channel, and the 2026-09-12
board reading is that decorrelation, not member quality, is the only channel that has ever added points.

**WHAT IT IS, ASSERTED FROM THE CHECKPOINT AND NOT FROM THE SCRIPT.** `labelwise_depth24_s2026`:
slices=24, slice_pooling=labelwise, calibrated_targets=True, hold_annotations=True, band 0.1-0.9,
batch 10, backbone_lr 4e-5, 12 epochs, peak epoch 8, CONVERGED (peak is not the last epoch), and
`slices.gate.3.weight` non-zero on 768 of 768. 0.8617 @holdout909 midpoint — DIRECTIONAL, a third of the
1.5% bar, and never quoted as a finding.

**WHAT THIS ENTRY DOES NOT CLAIM.** The 0.75 ruler ranks this family first and that ranking is NOT the
argument here. `board.jsonl` carries `offline75` on 0 of 133 rows, so no dual-ruler fit exists; and the
0.75 family has had exactly one board test — `l75seat_v1`, which predicted 0.949 under ledger75 and read
0.939. A 0.75 number may SUSPECT a midpoint null, never overturn one. Every number above is midpoint.

**EVERYTHING ELSE IS THE CHAMPION, UNCHANGED.** Same six, same order, same `overlap=False`, same
`LogitAverage`, same kit, same raptor checkpoint at weight 1/3, same rank-space join by study.

PRE-REGISTRATION, written before the number, against the champion's 0.944 and `BOARD_RESOLVABLE = 0.0015`.
The board quantises to 0.001, so no band edge below is a score the board cannot return.

- **0.947 and above** — a seat added board points on decorrelation alone at a size the offline ruler
  could not see. That refutes the offline ruler's +0.0040 gate as a SHIPPING gate and re-opens every
  sub-gate seat the project has declined on it. Re-pin, then ship `labelwise_drop` and `dropstack_light`
  as the next two seats before any other axis gets a slot.
- **0.9455 to 0.9469** — a real gain, one resolution step clear of 0.944. The labelwise family earns a
  permanent seat; re-pin and continue down the decorrelation lane.
- **0.9425 to 0.9455** — FLAT, inside the band `BOARD_RESOLVABLE = 0.0015` cannot separate from 0.944.
  Then a +0.0008 DISTINCT seat does not reach the board, and the instrument has done its job: the
  +0.0040 gate is CONFIRMED as the floor at which this ruler predicts a board tick, and no further slot
  goes to a sub-gate seat of any family. That closes the seat-addition lane on a measurement rather than
  on a suspicion, which is the whole reason this slot is being spent.
- **Below 0.9425** — a seat the offline ruler priced POSITIVE and DISTINCT cost board points. That is
  the first such reversal this project would own, and it makes `seat_value`'s sign, not only its
  magnitude, untrusted at this scale. Read `record_mix` over the seven before any roster move.
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
        "final_ws_cf",
        "calib_span_ff",
        "grade_ladder_cf",
        "stack_increments_ff",
        "lr4e-5_ws_cf",
        "through_dense_cf",
        # THE ONLY LINE THAT DIFFERS FROM THE CHAMPION. The six above are in the champion's own order,
        # unchanged; nothing about how they are weighted or joined moves. This member lives on the
        # SECOND shelf (`dimiturnt/kneemri-members-2`, live since 2026-09-11T20:28Z), which is why this
        # kernel mounts one dataset the champion does not. Its encoder is dinov2-small — already mounted
        # for the six — so no new encoder dataset enters with it.
        "labelwise_depth24_s2026",
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
