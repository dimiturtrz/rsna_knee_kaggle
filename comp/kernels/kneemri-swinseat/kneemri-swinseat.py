"""swinseat — the champion eight: our own non-DINO trunk in the seat its pin says is the plateau.

WHY THIS ENTRY EXISTS, and the argument is the champion's own record rather than a fresh idea.
`comp/champion.json` states in its principles that THE POOL IS AN ARCHITECTURE MONOCULTURE AND THAT IS
THE PLATEAU: all six of `raptorweight_v1`'s members are `facebook/dinov2-small` at 336 px with one head,
and seven differently-seated entries span 0.937-0.941 because they are seven reseatings of one lens. The
same file's arithmetic says the member-quality lane ceilings near 0.943 against a 0.952 leader, and
knee-orjq closed stacking at 7/7 (p = 0.008). The ONE entry that left that plateau did it by seating a
different ARCHITECTURE — the foreign raptor CoAtNet, 0.941 -> 0.943/0.944. This entry runs that axis from
the side we control: a supervised Swin-base trunk of ours, which is neither DINO nor raptor.

WHY NO OFFLINE READING DECIDES THIS, which is what earns a slot rather than another scan. The champion is
six `--fold -1` members and is fully blind by construction. `champion.json` records, measured, that
substituting fold-0 twins to score it produces a confident wrong answer — a fourteen-tick miss on
`peak8_v1`. Every offline reading this project has taken of the non-DINO lane (`seat_value` against
`greedyseat_v1`; `record_mix --seats honest` and `--seats diverse` under both vote rules) ran over a
fold-0 pool sharing ZERO members with the shipped entry. Those readings are sound about that pool and
say nothing about this one; carrying them across the fold line is the error the pin names.

AND THE NUMBER THEY DID PRODUCE IS UNRESOLVED, NOT NEGATIVE. The honest block delta was -0.0038 against a
0.0053 paired floor. `board_transfer` reports 9 inversions in 392 pairs and all nine are pairs the holdout
left UNRESOLVED, against 0 of the 252 it resolved. The offline ruler is not entitled to a verdict here.

PRECEDENT FOR SEATING A MEMBER ITS OWN SCORE ARGUES AGAINST. `base_dense` lost its matched control by
0.0066 and bought a board tick; `lora_r8_v3small` is rostered at 0.8215 as the only non-DINOv2 member;
`focus_mcl` is seated deliberately at 0.5332. `champion.json` states the rule outright: a member that
failed its own matched control can still win a seat, because alone-quality and seat-value are different
quantities. `swin_wholetop` is seated on the axis, not on its solo number.

ONE VARIABLE MOVES. `raptorweight_v1`'s exact six, the same raptor foreign seat at 1/3, the same
`LogitAverage`, the same kit, the same rank-space join — plus `swin_wholetop_s2026` as a SEVENTH member of
ours. Count and architecture move together and this entry cannot separate them; `fleettwo_v1` already read
the foreign COUNT axis flat, which is the control that makes the architecture reading the live one.

PRE-REGISTERED, and the band is drawn against the board's own 0.001 quantisation so it cannot be mis-drawn
the way `raptorweight`'s was. Champion 0.944.
- **0.946 or above** — architecture diversity from our own side is a real lever, two ticks clear of board
  resolution, and it lands in the top ten. Then the lane is open regardless of what `swin_calib_dense`
  reads, and the follow-up is a BETTER non-DINO member in the same seat.
- **0.943-0.945** — FLAT, inside one tick either way. Then our own non-DINO trunk does not do for us what
  the foreign CoAtNet did, and the question worth chasing is what makes the foreign seat different —
  corpus, or architecture, or both — rather than another trunk.
- **0.942 or below** — the eighth seat harms. Read it WITH `fleettwo_v1`'s flat count reading before
  concluding anything about the axis: a cost here is as consistent with seat count as with architecture.

THE REFUTATION THIS CANNOT MAKE: it cannot say whether a non-DINO trunk trained on the recipe that
actually seats members would do better. That is `swin_calib_dense` (knee-csxq), which runs tonight and is
not gated on this result either way.

BUDGET. Seven of ours plus the raptor pass on one T4. The Swin member reads 384 px at 20 slices, which is
the heaviest member in the entry, and `Budget` is handed what the raptor pass already spent so both
producers aim at the SAME ceiling. If the session runs short, `KernelRun` drops members — which is the
one failure that would score this entry without the thing it is for, so read the log's member list.

bead: knee-o1eg
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
        "merged_blend_cf",
        "lr4e-5_ws_cf",
        "through_dense_cf",
        # The seventh of ours and the reason this entry exists: the only member here that is not
        # dinov2-small. It reads 384 px because Swin-base's window tables are baked at 12 and only a
        # 384 px read tiles them exactly — see backbone.py SWIN_BASE, where that is why the arm pins it.
        "swin_wholetop_s2026",
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
