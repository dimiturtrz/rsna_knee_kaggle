"""colbaker — the champion entry with the column our own map calls STRONGEST replaced by a constant.

**THIS IS THE SECOND HALF OF A MEASUREMENT, AND `colmcl` ASKED FOR IT BY NAME.** That kernel closed on the
refutation it could not make: it could not separate "the columns are weighted" from "the board's MCL column
is much easier or much harder than our holdout's", and it said a second ablation on a different column
would separate them and was worth a slot only if the first landed outside its band. It landed outside its
band, downward and hard.

**WHAT `colmcl` READ, and why it demands a partner.** It scored 0.904 against the champion's 0.944.
`colnull` (a constant 0.5 in every cell of every column) had already read exactly 0.500, which fixes the
metric as a plain unweighted mean of twelve per-column ROC AUCs with no offset and no non-AUC term. So MCL's
board AUC is (0.944 - 0.904) x 12 + 0.5 = **0.980**, interval 0.968-0.992 on the rounding alone. That is 96%
of the largest drop one column can produce. The harness was checked before the number was believed:
`run_check` reads `loaded 6 of 6 members`, the scored file carries exactly the twelve `TARGETS` columns with
MCL at one unique value, and this kernel's `chosen` tuple and `FOREIGN_WEIGHT` are byte-identical to the
champion's.

MCL is the **worst** column of the seven-member fold-0 rank pool at 0.8104 and very nearly the **best**
column on the board. One of two things is true and they license opposite work.

**WHY BAKER'S.** It is the strongest column that pool has, 0.9208, and it is the cleanest possible contrast:
if our map carries any information at all, the column it ranks first should behave differently from the
column it ranks last. Ablating another weak column would confound "our map is wrong" with "weak columns are
special". Ablating the strongest one cannot.

PRE-REGISTRATION, written before the number, against the champion's 0.944. Board AUC of the ablated column
is (0.944 - score) x 12 + 0.5 throughout.

- **0.900-0.912** — a drop of 0.032-0.044, so Baker's board AUC is 0.88-1.00, i.e. near-ceiling like MCL.
  EVERY column is near-ceiling on the board and our per-column map carries no information about the board at
  all — neither level nor ordering. The board's own headroom is then tiny and spread thin, and no per-label
  arm of any kind is licensed. This is the branch I expect.
- **0.913-0.922** — a drop of 0.022-0.031, so Baker's lands at 0.76-0.87: BELOW MCL. The map is inverted
  broadly, our per-column supervision is anti-correlated with board truth, and the supervision axis becomes
  the only lane worth card. Licenses the full-corpus relabel at whatever it costs.
- **0.923-0.935** — a drop of 0.009-0.021, so Baker's lands at 0.61-0.75. The column our map ranks FIRST is
  a genuinely weak board column. Same conclusion as the band above, held more strongly.
- **above 0.935** — a drop under 0.009. Baker's board AUC is under 0.61 and the column is nearly worthless on
  the board while our map ranks it first. That would be the largest inversion this project has measured and
  the map would be actively misleading, not merely uninformative.

Note that the branch matching "our map transfers" — Baker's dropping ~0.035, MCL dropping ~0.020 — is
already dead, because MCL dropped 0.040. No score this entry can return restores the map. What it decides is
whether the map is UNINFORMATIVE about the board (first branch) or INVERTED (the rest), and those aim
different next arms: the first says stop reading the offline per-column table entirely, the rest say our
labels are anti-correlated with the board's on the columns we trust most, which is a much larger claim about
supervision than the MCL cell alone can support.

**COST, STATED HONESTLY.** This entry is a DELIBERATELY DAMAGED champion. It is expected to score around
0.91 and it will not improve the board position. It costs one slot and no card time, and it is the gate on
whether roughly four hours of card go to a full-corpus relabel. It is an INSTRUMENT, not a candidate.

The refutation this cannot make: two ablated columns is still two of twelve, and neither says what the other
ten do. If both land near-ceiling the honest reading is "the two we tested are near-ceiling", not "all twelve
are" — the fix for that is arithmetic, not another slot, since the twelve must average 0.944 exactly.
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

ABLATED = "Baker's"
"""The finding whose column is replaced by a constant. One column, named here rather than passed, so the
entry that scored can never be read against a different column than the one it damaged."""

CONSTANT = 0.5
"""What the ablated column is filled with. Any single value gives that column an AUC of exactly 0.5 under a
rank metric; 0.5 is chosen so a reader of the submission file can see at a glance that it is not a score."""

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
    )
    # What the raptor pass cost, handed to the six so their nine-hour budget is the SESSION's and not
    # a fresh one. Without it both producers aim at the same ceiling and the kernel is killed at the
    # real one with no submission — a budget defeated by its own assumption that it starts at zero.
    spent = time.time() - started
    logging.getLogger(__name__).info(f"the raptor pass cost {spent:.0f}s; the six get what is left")
    left = Budget(seconds=max(KERNEL_SECONDS - spent, 0.0))
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=left)).run()
    ours = pd.read_csv(WORKING / "submission.csv")

    entry = RaptorSeat(weight=FOREIGN_WEIGHT).blended(ours, theirs)
    # The whole point of the entry: one column is replaced AFTER the blend, so every other column
    # is the champion's own table and the board delta is attributable to this column alone.
    entry[ABLATED] = CONSTANT
    entry.to_csv(WORKING / "submission.csv", index=False)


if __name__ == "__main__":
    main()
