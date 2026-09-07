"""stack — the champion's CLEAN five with three EXTRA seats in the ours-block, the foreign token untouched.

**WHAT MOVES AND WHAT DOES NOT.** The foreign seat keeps its 1/3 and its checkpoint; the vote stays
`LogitAverage`; `overlap=False` stays; the champion's members stay in their order. The ours-block is a
mean over EIGHT members instead of six: `labelwise_dense_s2026`, `warm_calib_dense_s2026` and
`warm_calib_lowlr_s2026` are ADDED, and `merged_blend_cf` LEAVES. No weight is refitted and no selection
is run. It is the first entry that asks whether the ours-block gains from WIDTH at fixed quality.

**WHY ONE MEMBER LEAVES, AND WHY THAT IS NOT A RESEATING.** `merged_blend_cf` trains on the Rijeka merged
table, whose mirror is published CC BY-NC-ND; the ND term is the open question against competition rule
2.6a and the clause read is owed. D74 prices its seat beside the rest at -0.0001 [-0.0007, +0.0004] over
300 draws on the 896-study L75mid frame -- the seat is free to remove, so the exposure is closed at no
measurable cost rather than carried while the rules are read. A removal priced at zero is not a selection
step: nothing was ranked, and the entry would read the same with it in.

**THE MECHANISM.** D45 prices the three as extra seats against the champion block: +0.0009 for
`warm_calib_lowlr` alone at corr 0.898 to the rest, and +0.0026 for all three together. Each is a member
whose SOLO number is below the block's -- `warm_calib_lowlr` reads 0.8577 against its cold parent's
0.8634 -- which is exactly the case the champion's own record says to buy on the AXIS rather than on the
solo score (calib_span, base_dense, through_dense all entered seats their own controls said they lost).
What these three bring is decorrelation, not accuracy.

**WHY THIS IS READABLE AT ALL.** A +0.0026 offline delta was previously unquotable: the ledger's mixed-frame
`offline` column predicts the board at R2 0.802 with residual SD 0.0036, so a gain this size sits inside its
own noise. D50 rebuilt 36 honest board rows on ONE common frame (L75mid, 896 studies) and that ruler reads
R2 0.962, slope +1.14, resid SD 0.0016 -- which makes +0.0026 offline a ~+0.003 board prediction with a
residual half its size. THE FIT'S OWN BAND IS 0.904-0.937 AND THE CHAMPION'S 0.944 IS OUTSIDE IT, so the
slope is an extrapolation at the top of the board and the prediction inherits that.

**THIS IS AN INSTRUMENT, NOT A CHAMPION CANDIDATE.** The member axis is closed on seven board flats and
this entry does not re-open it: what it prices is the ONE dimension those flats never varied, the ours-block
WIDTH, and it does so without a selection step that could overfit. A flat read closes width too.

PRE-REGISTRATION, written before the number, against the champion's 0.944 and BOARD_RESOLVABLE = 0.0015.

- **0.947 or above** -- width is a real lever and it is bigger than the transfer predicts. The next slot
  widens further (the remaining priced extra seats), and D50's slope is under-reading at the top.
- **0.9455-0.9465** -- the predicted +0.003, above the resolution floor. Width pays at the priced rate;
  re-pin the champion and keep adding seats while their D45 price stays positive.
- **0.9425-0.945** -- FLAT. Three seats priced at +0.0026 offline bought nothing resolvable, which
  refutes the transfer AT THE TOP OF THE BOARD (not the transfer itself -- 0.944 is outside its fit band).
  The ours-block width axis closes with the count axis and the weight axis, and no further slot goes to one.
- **0.941 or below** -- the extra seats DILUTE. Three members whose solo numbers are below the block's
  drag it down when averaged in flat, which would say the seat-value ruler is measuring something the
  serving vote does not reproduce, and every D45 extra-seat price becomes suspect.

The refutation this cannot make: it cannot separate "width is flat" from "THESE three are flat". The
labelwise seat ships as its FULL-FIT twin (`labelwise_dense_ff`, --fold -1, found on disk and rostered for this
entry); the two warm seats have no such twin and ship fold-0, priced by the D35 sixth-seat fit at about
-0.0002 of block delta each. So +0.0026 is a FLOOR for the shipped version by roughly that much.
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
        "lr4e-5_ws_cf",
        "through_dense_cf",
        "labelwise_dense_ff",
        "warm_calib_dense_s2026",
        "warm_calib_lowlr_s2026",
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
