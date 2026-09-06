"""foreignsolo — mattiaangeli's five DINOv3 folds, rank-mean, and NOTHING else. An INSTRUMENT.

WHAT THIS BUYS, AND WHY IT IS NOT A SEATING OPERATION. It seats nothing: no member of ours is in it.
`foreignseat_v1` put this family in at 6/13 beside our seven and scored 0.929 against `raptorweight_v1`'s
0.944 on the identical seven — a -0.015 board move, of which -0.002 is the read collapse `onegroup_v1`
already priced, leaving -0.013 unexplained against an offline seat of +0.0178 at 6x. Three dials moved at
once, and no offline ruler we own has been able to name which. This entry reads the family's own board
number S directly, which is the one quantity every branch of that argument is a function of.

WHAT WAS CLEARED OFFLINE FIRST, SO THE SLOT IS NOT SPENT ON A BUG.
- Assembly: `run_check` on the hidden run's log reads 6 of 6 ours and five of five folds mounted
  (`m_f0..m_f4`), weights 0.3590 / 0.1795 / 0.4615, and the family runs its own pass at its own geometry.
- The fold key is transcribed, not inferred: `foreign_split._group` is their `md5(report_text)[:8] % 5`
  straight from `bend_the_knee_to_dinov3_ensembled.py:1508`, so the fold-matched pricing has zero
  contaminated cells by construction.
- The family is healthy on our 909: pairwise rank agreement 0.79 across the ten fold pairs WITH a +0.027
  gain from averaging them — disagreement that is useful, which is the opposite of a noisy family.
- The served band is the best band measured: fold-matched pooled 0.8812 at 0.12-0.88 against 0.8582 at
  0.08-0.92 and 0.8601 at 0.16-0.84. The all-fold reductions that read ~0.89 for every band are
  contaminated by +0.035 and must not be used to choose one.
- The seat was priced on the geometry it served (`logs/seat_weight_served.log`, 6x = +0.0162).

So the family is cleared on OUR ruler, and the board-vs-909 gap is the whole remaining question. The
vote rule cannot carry -0.013 alone: rank beats logit by +0.004-0.006 at this seat offline, and a rule
that flipped by 0.017 between two rulers would be a larger anomaly than the one being explained.

WHAT S PRICES. With the ensemble at E = 0.942 on the collapsed read and the family at S, every
(weight, rule) pair on this axis becomes computable by the same rank-mix arithmetic already run offline,
instead of costing a slot each. That is why the solo goes ahead of the LogitAverage twin: the twin has a
pre-registered dead band (0.934-0.941, "not separately actionable") and reads S only through a blend.

WHAT WOULD REFUTE IT.
- S >= 0.940 — the family is board-strong, the -0.013 is the blend and not the member, and the seat is
  re-run at a low weight under LogitAverage.
- 0.920 <= S < 0.940 — a diversity seat only, weight bounded near 1/10; the foreign axis is a small
  lever rather than the +0.018 one, and no further slot goes to it this cycle.
- S < 0.920 — the family does not transfer to the hidden split, the foreign axis closes for THIS family,
  and a second family needs its own board reading before any card time is spent on it.

LICENCE. Unchanged from `foreignseat`: their weights are the public CC0 dataset
mattiaangeli/knee-mri-fold-weights, MOUNTED and never redistributed; the DINOv3 lvd1689m weights beneath
them are under Meta's DINOv3 Licence, whose two conditions are met here — this kernel is Built with
DINOv3 and a copy of the Agreement travels in the kit.
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

DINOV3 = "Built with DINOv3"
"""Meta's DINOv3 Licence asks for this line prominently displayed; the Agreement ships in the kit."""


def main() -> None:
    boot()

    import logging as _logging  # noqa: PLC0415 - nothing above the bootstrap block is importable

    from core.mounts import Mounts  # noqa: PLC0415 - the kit is on sys.path only after boot
    from kneemri.kaggle.foreign_entry import CHECKPOINTS, SOURCE, ForeignEntry  # noqa: PLC0415 - same
    from kneemri.kaggle.raptor_run import WORKING  # noqa: PLC0415 - same

    logger = _logging.getLogger(__name__)
    logger.info(DINOV3)
    mounts = Mounts()
    competition = mounts.one("sample_submission.csv", "competition data")

    # ForeignEntry writes and validates submission.csv in `working` itself, so the single producer runs
    # straight into WORKING. The scratch directories foreignseat used exist only because three producers
    # each write a file of that name; with one producer there is nothing to disambiguate.
    frame = ForeignEntry(
        competition=competition,
        kernel=mounts.one(SOURCE, "their serving program, carried in our kit") / SOURCE,
        checkpoints=mounts.one("m_f0.pt", f"the {CHECKPOINTS} fold checkpoints"),
        # Their program builds a study's path as SERIES_ROOT / study / series, so the root is the
        # test_series DIRECTORY and not the competition mount.
        series_root=competition / "test_series",
        working=WORKING,
    ).run()
    logger.info("foreign family solo, %d studies, no member of ours in this table", len(frame))


if __name__ == "__main__":
    main()
