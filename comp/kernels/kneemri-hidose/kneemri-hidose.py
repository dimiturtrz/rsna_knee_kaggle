"""hidose -- the same seat as `kneemri-slotseat`, filled with the NEXT DOSE UP. A straight A/B.

**THIS ENTRY IS ONE VARIABLE AWAY FROM THE ENTRY SHIPPED AT THE MARK BEFORE IT**, and that is its whole
design. `slotseat` seats `slot_dropout_s2026` (dose 0.25); this seats `slot_dropout_hi_s2026` (dose 0.40).
Six champion members, same order, same join, same raptor weight, same budget hand-off -- and the only
difference between the two board rows will be the dose. That makes the PAIR the measurement, exactly the
way honestsix/honestfar were a pair, and it is the only way this project can ask whether the cross-site
decode toll it has been fitting a dose-response to moves the board at all.

**THE DOSE-RESPONSE IT IS BUILT ON**, measured against the matched control `band_dense_s2026` in one pass
each, calibration gate passed (the control reproduced -0.0361 exactly, and `slot_cost` landed inside
+/-0.0001 on every row, so the shifted code path served our own pixels correctly):

    dose        0.00      0.25      0.40
    decode_cost -0.0361   -0.0271   -0.0229
    vs control  --        +0.0090   +0.0132   (3.5 sd of the eleven-member pool the toll was calibrated on)

Monotone through the last measured point, which leaves dose 0.55 open and is NOT a claim that 0.40 is
optimal. One foreign corpus is one point on transfer.

**WHAT THE OFFLINE RULER SAYS, AND WHY IT IS NOT THE REASON.** Native holdout909 is 0.8551 against the
control's 0.8578: -0.0027. That is not a trade-off to be weighed, because holdout909 is drawn from the
same decode as training and is blind BY CONSTRUCTION to the axis this member exists for. The seat_value
table below prices the 0.25 dose at -0.0001 on that same blind ruler.

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
    slot_dropout_s2026        0.8592   0.8592  0.8321   -0.0000    -0.0271   (dose 0.25 -- slotseat's seat)
    slot_dropout_hi_s2026     0.8551   0.8551  0.8322   -0.0000    -0.0229   (dose 0.40 -- THIS entry's seat)
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

**THE DECIDING NUMBER.** The 0.40 dose against the same documented control `band_dense_s2026` (-0.0361)
is **-0.0229, a toll +0.0132 shallower** -- 3.5 sd of the eleven-member pool (mean -0.0336, sd 0.0038), the
shallowest toll-vs-matched-control this project has measured, deeper than the 0.25 dose's +0.0090, and
above the +0.0040 floor the board itself confirmed the previous day. The same pass ran `fieldwide`, an arm
built to test the SAME thesis on a different shared field, and it came back FLAT (+0.0042 against its
matched control, inside its own pre-registered -0.0049..+0.0049 band) and a TWIN. So the pass refutes one
field and supports the other,
which is what a working instrument looks like rather than a pass that agrees with whoever ran it.

**WHY THE CROSS-SITE RULER IS THE RIGHT ONE FOR A BOARD THIS RULER CANNOT MOVE.** `board_transfer` is a
saturating curve, not a constant: +1.72 board points per holdout point over the whole record, but +0.74
restricted to the regime we now ship in (offline >= 0.8661), and above 0.8679 no pair resolves at all.
A better solo therefore cannot clear the board's own 0.001 reporting resolution, and `knee-gcgbk` has
eleven distinct kernels sitting at exactly 0.944 with reseating retired. The one axis left is a term the
in-distribution ruler cannot price, on a test set the owner has stated plainly is NOT our dataset.

**PRE-REGISTERED BANDS, WRITTEN BEFORE THE SLOT IS SPENT.**

    >= 0.947    the cross-site ruler PICKED A BOARD TICK the offline ruler scored at zero. That is a new
                selection instrument, and it re-opens the shared-field axis (padding_mode is still
                unpriced, and the dose-response is monotone through its last measured point, 0.55 open).
    0.9455-0.9469  a real gain, smaller than the ruler's own scale. Seat kept, axis opened cautiously.
    0.9425-0.9455  FLAT. The cross-site ruler joins the offline ruler as a thing that cannot pick a board
                tick at this magnitude, and the shared-blind-spot thesis is retired as a SHIPPING
                argument -- it survives only as an explanation of why offline flats are uninformative.
    <  0.9425   the seat is actively harmful, which would say the cross-site toll measures robustness to
                a shift the competition test set does NOT contain, and the foreign decode retires as a
                selector entirely.

**AND THE SECOND READ, WHICH IS THE ONE THIS ENTRY EXISTS FOR: THE PAIR.** hidose minus slotseat, the two
rows differing in one member and nothing else. The board is quantised to 0.001 and D368's estimability
floor puts the MDE at ~0.003, so:

    >= +0.004   the dose-response TRANSFERS: a deeper cross-site toll bought a board tick, and dose 0.55
                becomes the next arm rather than a curiosity.
    -0.003..+0.003  NULL, and the honest reading is that the board cannot resolve a gradient of this size.
                It does NOT refute the toll; it prices the gradient below the ruler.
    <= -0.004   0.40 OVERSHOOTS. The toll keeps improving while the member gets worse, which would say the
                toll is measuring indifference-to-slots past the point where the member still uses them,
                and the dose axis closes at 0.25.

Neither read is licensed to be taken alone: an absolute band with no pair says only what eleven 0.944s
already said.

**EVERYTHING ELSE IS THE CHAMPION, UNCHANGED.** Same six, same order, same `overlap=False`, same
`LogitAverage`, same raptor member at `FOREIGN_WEIGHT` 1/3, same budget hand-off. One line differs, and
it names a member that appears in 0 of 130 board rows -- so this is a SOURCE change, not a reseating.
`slot_dropout_hi_s2026` is on the SECOND shelf (`dimiturnt/kneemri-members-2`), which is why this kernel
mounts one dataset the champion does not. The first shelf is ~10 GB and 127 members, so a member added to
it would reach the board only by re-pushing all of it; `comp/members2.roster` exists for exactly this, and
`MemberPackage.spanning` REFUSES an identifier two mounted roots both hold, so the member is named on the
second roster and on neither other. Verify the mount by asking Kaggle for the file list at page_size 1000
-- the client's default page_size of 20 reports a 136-file dataset as 20 files and will tell you a live
member is absent.

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
        "grade_ladder_cf",
        "stack_increments_ff",
        "lr4e-5_ws_cf",
        "through_dense_cf",
        # THE ONLY LINE THAT DIFFERS FROM THE CHAMPION, AND THE ONLY LINE THAT DIFFERS FROM `slotseat`.
        # The six above are in the champion's own order, unchanged; nothing about how they are weighted
        # or joined moves. This is the 0.40 slot-dropout dose against slotseat's 0.25 -- one variable,
        # one pair. It lives on the SECOND shelf (`dimiturnt/kneemri-members-2`), the only dataset this
        # kernel mounts that the champion does not; no new ENCODER enters with it. It is named in 0 of
        # the board's rows. Its cross-site decode toll is -0.0229 against the control's -0.0361: +0.0132
        # shallower, where the 0.25 dose is +0.0090. Its own-holdout 0.8551 is blind to that axis.
        "slot_dropout_hi_s2026",
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
