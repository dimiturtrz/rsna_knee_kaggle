"""The same greedy search as `greedyseat_v1`, run on a SHARPER RULER. The two rulers disagree about
this entry, which is what makes it an experiment rather than a sixth reseating.

**THE FINDING THIS SHIPS AGAINST (2026-09-10, `experiments/README.md`).** Every offline number in this
project is read at `LabelScores()` — `scores.py:19 MIDPOINT = 0.5` — because 47 call sites in 41 files
take that default and exactly three paths can produce a recut number. Fitting the board from each ruler
over the 40 entries that carry both, the default is the WORSE instrument by more than a factor of two:

    ruler              n    resid sd   corr     2sd board resolution
    midpoint (0.5)     40   0.0036     +0.901   0.0071
    ledger75 (0.75)    40   0.0015     +0.982   0.0031

Leave-one-out over the 34 ensembles: midpoint rmse 0.0029, ledger75 rmse 0.0017, ledger75 closer on 29
of 34. So every seating verdict this project has written was read at roughly twice the necessary blur,
and the five closures on the seating axis were made by an instrument that could not see the band they
were closing.

**WHY THIS IS NOT A RESEATING.** `greedyseat_v1` (board 0.936, ref in `board.jsonl`) is the matched
control and it is already spent: the SAME greedy forward selection, the SAME `.holdout.npz` arrays, the
SAME LogitAverage vote — scored with `LabelScores()` at the midpoint. This entry changes ONE thing, the
cut the greedy maximises. The seat lists overlap 3 of 8.

**AND THE TWO RULERS PREDICT DIFFERENT SCORES FOR IT.** That is the whole design:

    entry              midpoint   ledger75
    greedyseat_v1      0.8740     0.9040     -> board 0.936, spent
    THIS ENTRY         0.8747     0.9138

At midpoint this entry is **+0.0007** over the control — a TENTH of that ruler's own 0.0071 resolution,
i.e. dead flat, predicting ~0.936-0.937. At ledger75 it is **+0.0098**, predicting 0.949 through the refit
`board = 1.1122 x ledger75 - 0.0673` (resid sd 0.0017). One entry, two instruments, two incompatible
answers. The board adjudicates.

PRE-REGISTERED, and these are the rulers' branches, not a hoped-for score:

- **0.948 or above** — ledger75 is the right ruler and the midpoint closures are void as evidence. The
  seating axis re-opens, and every "flat" in the 0.003-0.007 band gets re-read (M312 is the audit).
- **0.940-0.947** — ledger75 is directionally right and midpoint is not, but the linear fit
  over-extrapolates past its data edge (see the caveat below). Keep the ruler, refit the constant.
- **0.934-0.939** — MIDPOINT WAS RIGHT and ledger75's advantage does not survive out of sample. The
  ruler finding is retracted to "a better fit on the entries already scored", the P0 migration bead
  closes unbuilt, and the five seating closures stand.
- **Below 0.934** — worse than its own control. Suspect the greedy's selection over 226 members before
  the ruler: the split-half correction measured mean optimism -0.0002 but with sd 0.0094 over six
  trials, so a standard error of ~0.0038 is all that test established.

**THE CAVEAT THAT IS NOT WAVED AT.** ledger75 0.9138 is PAST THE EDGE of the data the fit was made on
(max on record 0.9040, `greedyseat_v1` itself), so the score predicted here was an extrapolation of
0.010 beyond a ~0.05 range — and the extrapolation MISSED: `greedyseat_v1` scored 0.936 on the board,
not the number this paragraph pre-registered. Kept as written, with its outcome attached. It was 0.010
beyond a ~0.05
range. Separately, the PRICEABLE population — entries whose every member is fold-0 and therefore
scoreable, which this one is — has never exceeded board 0.937, while the unpriceable population holds
0.944. This entry predicts the priceable population jumps 0.015 past its own record. The mechanism
offered for that is exactly the finding above: every prior priceable entry was SEATED on the blurry
ruler, so priceable entries were systematically under-seated. That is a mechanism, not a guarantee.

**WHAT ELSE MOVED, stated because it weakens the A/B.** `greedyseat_v1` searched 111 members; this
searched 226, the pool filtered to members whose own `metrics.jsonl` records `fold 0`. That filter is
load-bearing and not cosmetic: the unfiltered 306-member pool's first greedy pick was `raptor_v4_swa`
at 0.8938, the member voided on 2026-08-29 as 98.6% in-sample. So the ruler is not the only variable,
and a gain here cannot be attributed to the cut alone.

**THE POOL IS LICENCE-CLEAN AND THAT COST NOTHING.** The first run of this greedy seated
`merged_blend_cf`, and `submit_check` refused it: it trains on the Rijeka table, whose terms are
unresolved against competition rule 2.5. That is the exact exposure `comp/champion.json` moved the pin
AWAY from, on its own stated reasoning that "rule 2.5 binds the WINNING submission, so the exposure had
to be cleared before a result was won under it" — and an entry pre-registered at 0.95 is precisely one
that could win under it. Re-run with the Rijeka-trained members removed, the greedy reaches a HIGHER
ledger75 than the exposed seating did. The exposure bought nothing and is gone; `licence_refusals` over
these eight returns none.

**THE POOL IS WHAT IS ACTUALLY MOUNTED, WHICH IS NARROWER THAN THE ROSTER.** `mount_check --live` refused
the first licence-clean seating: `labelwise_extended_s2026` and `resnet_224_long_s2026` are on
`comp/members.roster` but are NOT in the uploaded `kneemri-members` package, so a kernel naming them
would have loaded six of seven and scored under a name it never assembled — the `farseat_v1` failure,
which went five days unnoticed. The pool here is therefore the 101 members that are fold-0, 909-study,
licence-clean AND mounted. That costs **0.0013 at ledger75** (0.9138 against the roster-wide 0.9151),
under half the ruler's resolution. Rebuilding the package to recover it is queued for a later slot, not
waved at: `labelwise_extended_s2026` alone reads **ledger75 0.9060**, above `greedyseat_v1`'s whole
seven-member 0.9040, and is the best ledger75 SOLO this project owns.

All eight seats are fold-0 with a 909-study `HoldoutRecord`, so this entry is PRICEABLE and whatever it
scores becomes a fit point for both rulers.
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


def main() -> None:
    boot()

    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - same
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - the kit is on sys.path only after boot

    # Greedy forward selection over 226 fold-0 members, maximising LabelScores(cut=0.75).mean on the
    # 909-study holdout. Named rather than ranked so the entry stays quotable; the order is the order
    # the greedy added them.
    chosen = (
        "stackfit_s2026",
        "labelwise_dense_s2026",
        "warm_calib_dense_s2026",
        "ramp_dense_gate",
        "blendemalabel224_s2026",
        "finding_calib_dense_s2026",
        "calib_span_s2026",
        "mae_matched_s2026",
    )
    # LogitAverage, not the RankAverage default: `greedyseat_v1` serves it, and holding the vote rule is
    # what leaves the CUT as the moving variable.
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage())).run()


if __name__ == "__main__":
    main()
