"""swinseat — the champion, plus the first seat this pool has ever held on the ARCHITECTURE axis.

**THE HOLE IN THE RECORD, and it is a hole in exactly one dimension.** Three entries have now moved the
foreign seat: `raptorseat_v1` added one foreign decode and took 0.941 -> 0.943; `fleettwo_v1` added a
SECOND and read 0.942; `raptorweight_v1` moved the foreign share 1/7 -> 1/3 and read 0.944. The last two
are flat inside `BOARD_RESOLVABLE = 0.0015`, and the ledger closed the axis on them. But both flat reads
added ANOTHER RAPTOR — the same family, drawn from the same thirteen public checkpoints. The axis has
never been moved with a different ARCHITECTURE at all.

`comp/champion.json` says so in its own words, twice. *"THE POOL IS AN ARCHITECTURE MONOCULTURE AND THAT
IS THE PLATEAU. All six of our members are facebook/dinov2-small at 336 px with one head ... Seven
differently-seated entries span 0.937-0.941 ... because they are seven reseatings of one lens."* And
*"THE SEAT COUNT IS NOT THE VARIABLE - THE SEVENTH MEMBER'S TYPE OF DISTANCE IS ... Buy a seat on the
axis, not on the count and not on the solo score."* A Swin-B trained by us on our corpus is the one type
of distance the pool has never bought.

**WHY THIS MEMBER AND NOT `swin_supervised_s2026`.** That checkpoint is DECLINED in
`comp/members.declines`, and the decline's own closing instruction is *"Re-open on THAT number, not on
this one"* — `swin_wholetop`, the byte-identical re-run on the block discovery `knee-94ec` fixed. Under
the old discovery `--unfreeze-last 6` opened `layers.2.blocks.12-17` and left stage 3 (1024 wide, the top
of the network) FROZEN for the whole run, so 0.8331 is a fact about a handicapped run. This entry seats
the un-handicapped one, which is answering the decline rather than overriding it.

**WHAT THE OFFLINE RULER CAN AND CANNOT SAY HERE.** The decline priced the seat at -0.0012 (later -0.0003
against `honest12`) on the fold-0 proxy. That is not this question. Six of the seven seats below are
`--fold -1`, and this pin's own principle is unambiguous: *"an offline number may only be compared to
another offline number fitted on the SAME studies, and there is no comparison across the --fold -1
line."* So this entry carries NO offline number by construction, exactly as its three predecessors did.
The fold-0 seat itself is not a confound — `lastseat_v1` (fold-0) against `dark7_v1` (the same member
refit at `--fold -1`) read 0.937 both ways.

**EVERYTHING ELSE IS THE CHAMPION, UNCHANGED.** Same six, same order, same `overlap=False`, same
`LogitAverage`, same kit, same raptor checkpoint at the same `FOREIGN_WEIGHT = 1/3`, same rank-space
join. One member enters. That is what makes the read attributable.

PRE-REGISTRATION, written before the number, against the champion's 0.944 and `BOARD_RESOLVABLE = 0.0015`.

- **0.946 or above** — architecture distance is a live lever AND the entry is inside the top ten (0.946
  floor). Re-pin `comp/champion.json`. The follow-up is a SECOND architecture (ConvNeXt-V2 at the fixed
  discovery), not a second Swin and not a weight sweep.
- **0.945** — a resolvable gain over 0.944. Architecture distance pays where family distance did not.
  Re-pin, and price a second architecture before any further seating work. Note there is no band edge
  between 0.944 and 0.945: the board quantises to 0.001, and stating a fractional edge is the error
  `raptorweight_v1` made when it put a "clears the floor" label on a value below its own resolution.
- **0.943-0.944** — FLAT. The seat is inert on the architecture axis too, which is the third independent
  support for `knee-uco3` and closes ensemble construction as a lever outright. The remaining routes are
  the ones `champion.json` already names: external data and stacking. Do NOT seat a third architecture.
- **0.942 or below** — the Swin member DILUTES. Its solo deficit against the DINOv2 members is real and
  is not paid for by decorrelation, which would CONFIRM the decline's argument on the board rather than
  offline, and retire the trunk for seating.

The refutation this cannot make: it cannot separate "architecture distance is inert" from "THIS Swin
member is too weak to carry it". Its solo number is the guard — the entry is built only on the arm's
LANDS branch (0.8560 or above), so it is not made out of a member the deficit argument already covers.
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
"""The foreign side's share of the rank blend, held at the champion's value.

`raptorweight_v1` moved this 1/7 -> 1/3 and read one unresolvable tick, so it is NOT shown to be a lever
and it is NOT the variable here. It stays where the pinned entry left it, so that the single difference
between this entry and the champion is the member that enters below.
"""

SWIN_SEAT = "swin_wholetop_s2026"
"""The entering member: Swin-B at 384, ours, trained on our corpus, at the fixed block discovery.

Documented here because it IS the hypothesis, but INLINED in `chosen` below rather than referenced, and
that is not a style choice. `mount_check` reads the member names statically, and a name it cannot resolve
to a literal is not an error there — the tuple reads as the WHOLE package, so the check then demands every
mount any member could need. This entry was refused for an undeclared `dimiturnt/vjepa2-hf` that none of
its seven members use, purely because one element of the tuple was a constant.
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

    # The champion's six in the champion's order, then the entering seat LAST. The order is not a free
    # choice: `Budget` drops members from the tail when the session runs short, so the member under test
    # is the one that goes if the T4 cannot afford it — which fails the entry back to the champion
    # rather than shipping a truncated pool that scores as neither.
    chosen = (
        "final_ws_cf",
        "calib_span_ff",
        "grade_ladder_cf",
        "merged_blend_cf",
        "lr4e-5_ws_cf",
        "through_dense_cf",
        "swin_wholetop_s2026",  # SWIN_SEAT, inlined so `mount_check` can resolve it — see that constant
    )
    # What the raptor pass cost, handed to the seven so their nine-hour budget is the SESSION's and not
    # a fresh one. Without it both producers aim at the same ceiling and the kernel is killed at the
    # real one with no submission — a budget defeated by its own assumption that it starts at zero.
    spent = time.time() - started
    logging.getLogger(__name__).info(f"the raptor pass cost {spent:.0f}s; the seven get what is left")
    left = Budget(seconds=max(KERNEL_SECONDS - spent, 0.0))
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=left)).run()
    ours = pd.read_csv(WORKING / "submission.csv")

    RaptorSeat(weight=FOREIGN_WEIGHT).blended(ours, theirs).to_csv(WORKING / "submission.csv", index=False)


if __name__ == "__main__":
    main()
