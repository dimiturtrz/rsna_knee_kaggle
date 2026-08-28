"""Our six plus TWO foreign members, one seat each — does a second foreign arm add again?

**THE QUESTION `raptorseat` LEFT.** `raptorseat_v1` (55831424) seated `raptor_v4_swa` beside our six at
1/7 and scored **0.943**, landing in its own pre-registered REAL GAIN band, whose text names exactly one
follow-up: *"whether a second foreign arm adds again or whether one foreign decode is the whole of the
effect."* This entry is that follow-up. Eight seats: our six, `raptor_v4_swa`, and `raptor_ft_effv2l480`.

**WHY `effv2l480` AND NOT ONE OF THE OTHER TWELVE.** Measured, not guessed. `teacher_distance` scored all
thirteen public raptor checkpoints over 400 shared studies with `RankAverage.agreement` — no labels, no
slot — and the fleet is **four families and nine copies**: ten are `coatnet_rmlp_2_rw_384` @384 differing
only in fitting corpus and agreeing at 0.93+, and `v4_full` × `v4_full_swa` agree at **0.9955** because
they are one model twice. `raptor_ft_effv2l480` is the most isolated member of the thirteen
(nearest-neighbour agreement **0.8223**), a fourth family, a different resolution, and its pairing with
the already-seated `v4_full_swa` is among the most distant available.

**WHY TWO SEATS AND NOT FOUR.** The three non-coatnets have **no measured quality at all** — every raptor
arm is `--fold -1`, so their holdout is inside their training set and only the board can price them.
Seating three unmeasured members at 3/10 of the blend would let a dilution read as a diversity verdict.
Two adds ONE unmeasured member to a blend whose other seven cells are all board-measured, which is the
largest step that stays readable.

**ONE DECODE, TWO MEMBERS.** `RaptorEntry.tables` scores both foreign checkpoints from a single pass over
the split. The decode is the whole cost — a fleet pass measured the card at 0% while pydicom worked — so
a second member is a second forward pass, not a second entry's worth of runtime.

**THE SEATING RULE IS UNCHANGED.** `RaptorSeat.fleet` gives each foreign member `1 / (6 + n)` = 1/8, in
rank space, joined by study. It is the same rule 1/7 was, applied to a wider roster, and it is named
before the number rather than fitted after it.

PRE-REGISTRATION, written before the number, against `BOARD_RESOLVABLE = 0.0015` and the pinned champion
`raptorseat_v1` = 0.943.

- **0.947 or above** — the second foreign arm adds at least as much as the first. Distance keeps paying
  per seat and the pool is nowhere near saturated: the next slots go to seating the remaining families
  (`cnv2b336`, `cnn336`) and to ACQUIRING further distant trunks, not to fitting more of our own.
- **0.9445-0.946** — it adds, but less than the first did. Diminishing and real; re-pin, and seat a third
  family only if a slot is otherwise idle. The first foreign decode carried most of the effect.
- **0.9415-0.9444** — flat against 0.943. One foreign decode WAS the whole of the effect. Says the gain
  came from crossing the architecture boundary once, not from distance accumulating, and closes the
  foreign-seating axis at one seat. Do not spend a further slot widening it.
- **Below 0.9415** — the second seat DILUTES. `effv2l480` is distant and bad enough for the distance not
  to pay for it, which separates the two terms the fleet measurement deliberately could not: distance
  without quality is not worth a seat. It does not retract `raptorseat` — `v4_swa` is board-measured at
  0.923 solo and this member is not measured at all.

**WHAT THIS ENTRY CANNOT DO.** It cannot attribute a null to the seat versus to the member. A flat
reading is consistent with "a second foreign arm adds nothing" and with "this particular foreign arm is
weak", and telling those apart costs a solo slot on `effv2l480` that is not being spent. This docstring
says so before the number rather than after it.
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

FOREIGN = ("raptor_ft_coatnet_v4_full_swa.pt", "raptor_ft_effv2l480.pt")
"""The seated arm from `raptorseat_v1`, and the most isolated member of the fleet. See the docstring."""


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

    # The foreign pass runs FIRST and into a scratch directory: both producers write a file called
    # `submission.csv` and the one that runs last is the one Kaggle would otherwise score.
    scratch = WORKING / "raptor"
    scratch.mkdir(parents=True, exist_ok=True)
    started = time.time()
    mounts = Mounts()
    device = str(VisibleCards().device())
    members = {name: RaptorMember.loaded(RaptorEntry.weights(name, mounts=mounts), device) for name in FOREIGN}
    theirs = RaptorEntry(mounts.one("sample_submission.csv", "competition data"), working=scratch).tables(
        members, device
    )
    members.clear()
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
    # What the foreign passes cost, handed to the six so their nine-hour budget is the SESSION's and
    # not a fresh one. Without it both producers aim at the same ceiling and the kernel is killed at
    # the real one with no submission.
    spent = time.time() - started
    logging.getLogger(__name__).info(f"the two foreign passes cost {spent:.0f}s; our six get what is left")
    left = Budget(seconds=max(KERNEL_SECONDS - spent, 0.0))
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=left)).run()
    ours = pd.read_csv(WORKING / "submission.csv")

    RaptorSeat.fleet(ours, theirs).to_csv(WORKING / "submission.csv", index=False)


if __name__ == "__main__":
    main()
