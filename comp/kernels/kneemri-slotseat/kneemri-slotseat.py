"""slotseat -- the champion six plus the one member the offline ruler scores at ZERO and
the cross-site ruler scores at +0.0090.

**THIS IS A PREDICTED GAIN, NOT AN INSTRUMENT, AND IT IS THE FIRST ONE IN A WEEK.** The standing rule is
that a slot with nothing mechanism-backed behind it goes to an instrument with the instrument named.
This entry does not take that exit. It carries a mechanism, a falsifiable prediction that was written
down before the deciding number existed, and a ruler that is not the one seven flat reseatings were
chosen on.

**WHAT IS DIFFERENT FROM EVERY PRIOR SEAT, IN ONE LINE.** Every previous candidate was chosen because
the OFFLINE ruler liked it. This one is chosen although the offline ruler scores it at nothing.

    seat_value --against cleanseat_v1 (roster of 6 at 0.8755, cut 0.5, 2026-09-12)
      slot_dropout_s2026   -0.0001  [-0.0008, +0.0006]   DISTINCT (rank 0.9051, js 0.0190, maha 3.618)
      fieldwide_s2026      +0.0008  [+0.0001, +0.0014]   TWIN     (rank 0.9188, js 0.0096, maha 3.563)

**AND THAT IS WHY IT IS HERE, NOT DESPITE IT.** `labelseat_v1` shipped the day before with a seat the
same ruler priced +0.0008 [+0.0001, +0.0017], DISTINCT on all three axes, band excluding zero. The board
returned 0.944 -- its pre-registered FLAT band -- which is what CONFIRMED +0.0040 as the shipping floor
rather than leaving it an assumption. So "+0.0008 offline, DISTINCT" is now a tested reason to ship, and
it tested NULL. Shipping another member selected that way would be the same experiment. This is the
opposite selection: the offline number is -0.0001 and the argument is a different ruler entirely.

**THE MECHANISM: A BLIND SPOT EVERY MEMBER SHARES CANNOT APPEAR AS VARIANCE BETWEEN THEM.** Three
`AugmentConfig` fields sit at the identity value every member on disk shares -- `scale_jitter = 0.08`,
`padding_mode = "border"`, and `slot_dropout = 0.0`. An offline solo ruler reads holdout909, drawn from
the same field distribution as training, so it is STRUCTURALLY blind to all three: a flat there is the
EXPECTED reading, never a null. `slot_dropout_s2026` is exactly that -- 0.8592 against its control
`band_dense` 0.8578, +0.0014, knob-sized, the kind of number this repository calls noise and declines to
plan around.

**THE RULER THAT CAN SEE IT, AND WHAT IT SAID.** `foreign_readout` scores a member three ways on the SAME
909 holdout: native, `kept` (our own pixels through the shifted code path -- a plumbing self-check), and
`foreign` (a second site's decode geometry). `decode_cost = foreign - kept`. Run 2026-09-12 over four
checkpoints in ONE invocation, one split, one foreign decode:

    member                    native   kept    foreign  slot_cost  decode
    fieldwide_s2026           0.8668   0.8668  0.8359   +0.0001    -0.0309
    labelwise_drop_s2026      0.8649   0.8649  0.8299   +0.0000    -0.0351   (fieldwide's matched control)
    slot_dropout_s2026        0.8592   0.8592  0.8321   -0.0000    -0.0271
    band_dense_hold_s2026     0.8573   0.8573  0.8213   +0.0000    -0.0360   (calibration, known -0.0360)
    band_dense_s2026          0.8578   0.8578  0.8217   +0.0000    -0.0361   (slot_dropout's OWN control)

The last row was taken SEPARATELY, after the first four, because the first reading of this entry named
`band_dense_hold_s2026` as slot_dropout's control and that was wrong by one checkpoint: the arm's own
header (`experiments/arms/slot_dropout.sh`) declares `band_dense_s2026` at 0.8578. The correct control
was then measured rather than the claim being softened, and it reads -0.0361. Two sibling checkpoints
five ten-thousandths apart natively return tolls one ten-thousandth apart, which is a free estimate of
this measurement's own stability: ~0.0001, about ninety times below the signal below.

**BOTH PLUMBING GATES WERE WRITTEN DOWN FIRST AND BOTH PASSED.** `slot_cost` lands inside +/-0.0001 on
all four rows, so the shifted path serves our own pixels unchanged and `decode_cost` is about pixels.
`band_dense_hold` reproduces its known -0.0360 exactly, so the -0.08 defect recorded in this tool's own
void notice -- where six 20-slice members were silently scored on eight slices of twenty, with plausible
numbers logged -- did not recur. Neither gate was added after the numbers were seen.

**THE DECIDING NUMBER.** `slot_dropout` against its own documented control `band_dense_s2026` (-0.0361)
is **-0.0271, a toll +0.0090 shallower** -- 2.4 sd of the eleven-member pool (mean -0.0336, sd 0.0038), the
shallowest toll-vs-matched-control this project has measured, and above the +0.0040 floor the board
itself confirmed the previous day. The same pass ran `fieldwide`, an arm built to test the SAME thesis on
a different shared field, and it came back FLAT (+0.0042 against its matched control, inside its own
pre-registered -0.0049..+0.0049 band) and a TWIN. So the pass refutes one field and supports the other,
which is what a working instrument looks like rather than a pass that agrees with whoever ran it.

**WHY THE CROSS-SITE RULER IS THE RIGHT ONE FOR A BOARD THIS RULER CANNOT MOVE.** `board_transfer` is a
saturating curve, not a constant: +1.72 board points per holdout point over the whole record, but +0.74
restricted to the regime we now ship in (offline >= 0.8661), and above 0.8679 no pair resolves at all.
A better solo therefore cannot clear the board's own 0.001 reporting resolution, and `knee-gcgbk` has
eleven distinct kernels sitting at exactly 0.944 with reseating retired. The one axis left is a term the
in-distribution ruler cannot price, on a test set the owner has stated plainly is NOT our dataset.

**PRE-REGISTERED BANDS, WRITTEN BEFORE THE SLOT IS SPENT.** The base is the champion pinned in
`comp/champion.json` -- `tripleseat_v2`, the seven, board 0.946. A band quoted against any other base
does not read this entry.

    >= 0.949       the cross-site ruler PICKED A BOARD TICK the offline ruler scored at zero. That is a
                   new selection instrument, and it re-opens the shared-field axis (padding_mode is still
                   unpriced, and slot_dropout has been run at exactly one dose).
    0.9475-0.9489  a real gain, smaller than the ruler's own scale. Seat kept, axis opened cautiously.
    0.9445-0.9474  FLAT. The cross-site ruler joins the offline ruler as a thing that cannot pick a board
                   tick at this magnitude, and the shared-blind-spot thesis is retired as a SHIPPING
                   argument -- it survives only as an explanation of why offline flats are uninformative.
    <  0.9445      the seat is actively harmful, which would say the cross-site toll measures robustness
                   to a shift the competition test set does NOT contain, and the foreign decode retires
                   as a selector entirely.

**EVERYTHING ELSE IS THE CHAMPION, UNCHANGED.** Same seven, same order, same `overlap=False`, same
`LogitAverage`, same raptor member at `FOREIGN_WEIGHT` 1/3, same budget hand-off. One line differs, and
it names a member that appears in 0 of 130 board rows -- so this is a SOURCE change, not a reseating.
`slot_dropout_s2026` is on the FIRST shelf (`dimiturnt/kneemri-members`, live, verified by asking Kaggle
for the file list at page_size 1000 -- the client's default page_size of 20 reports a 136-file dataset as
20 files and will tell you a live member is absent).

**WHAT THIS ENTRY DOES NOT CLAIM.** It does not claim slot dropout raises the board; nothing has measured
that, which is what the slot is for. It does not claim the cross-site number is a board predictor -- that
is the hypothesis under test, and the FLAT band above retires it. And it does not rest on the offline
ruler at any point: that ruler prices this seat at -0.0001 and is quoted here only to show it was asked.
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
        "stack_increments_ff",
        "through_dense_cf",
        "labelwise_depth24_drop40_long_s2026",
        "labelwise_ext_drop40_long_s2026",
        "labelwise_dense_drop40_long_s2026",
        # THE ONLY LINE THAT DIFFERS FROM THE CHAMPION. The seven above are in the champion's own order,
        # unchanged; nothing about how they are weighted or joined moves. This member lives on the FIRST
        # shelf (`dimiturnt/kneemri-members`), which the champion already mounts, so this kernel mounts
        # no dataset the champion does not and no new encoder enters with it. It is named in 0 of 130
        # board rows. The offline ruler prices it -0.0001; the cross-site ruler prices it +0.0090 against
        # its own matched control. The docstring says which of those two this entry is betting on.
        "slot_dropout_s2026",
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
