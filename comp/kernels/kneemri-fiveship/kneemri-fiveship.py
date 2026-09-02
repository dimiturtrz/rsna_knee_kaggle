"""The champion's five, minus one seat — the instrument that splits `twinfive_v1`'s 0.009.

**WHAT IS UNRESOLVED.** `twinfive_v1` shipped the fold-0 twins of the champion's seats and read
**0.935** against the champion's **0.944**. That 0.009 is the largest un-decomposed quantity on our
board and it is the term every twin-fitted combiner reading is confounded by — `aucvote_v1` and
`stackvote_v1` both fitted their weight tables on those twins and then served the `_ff`/`_cf`
members, and 0.009 is nine times the 0.001 spread those entries were reading between vote rules.

But it is two terms glued together. `twinfive_v1` has **five** seats; the champion has **six**. So
0.009 = (twin-to-shipped) + (one missing seat), and nothing on the board separates them. It cannot
be separated from the twin side: `final_ws_e12` is `--fold -1` and has no fold-0 twin, folds are
banned, so a sixth twin cannot be built.

**WHAT THIS SHIPS.** The same five seats as `twinfive_v1`, but the SHIPPED members instead of the
twins — `final_ws_cf` dropped, everything else identical, same `LogitAverage`, same `overlap=False`,
no foreign seat. One variable moved against `twinfive_v1` (twin vs shipped) and one moved against
the champion (six seats vs five). Subtracting this entry from each side gives both terms.

**WHY IT IS NOT A CANDIDATE.** It is the champion with a seat removed; the honest expectation is at
or just below 0.944, and seat count has read flat across seven board entries. Nothing here is being
sold as an improvement. Under CLAUDE.md's pacing rule this slot goes to an instrument because
nothing mechanism-backed is ready — `ortho_pool_read` is queued on the card and will not land for
hours — and an unspent slot is destroyed at 00:00 UTC.

**THE OPERATION HAS NEVER SHIPPED.** `comp/board.jsonl` grepped before the push: no five-seat subset
of the champion exists on the board. The shipped-member seat-count curve it does hold is
`calibsolo` 1 seat 0.923, `trioseat_v1` / `fasttrio_v1` 3 seats 0.938 / 0.939, six seats
0.941-0.944 — so the marginal seat is worth ~0.002 near six, and that prior is what the branches
below are written against.

PRE-REGISTRATION, written before the board number, against BOTH neighbours:

- **0.942-0.944** — the missing seat costs ~0.000-0.002, as the seat-count curve predicts. Then
  the twin-to-shipped gap is essentially the whole **0.007-0.009**, and it stands as measured: any
  rule fitted on fold-0 twins is fitted on a pool that far below the one it will serve. Every
  combiner reading keeps its stated confound at full size.
- **0.938-0.941** — the seat and the twin term SPLIT the 0.009 roughly evenly (~0.004 each). The
  twin gap then stops dominating the vote-rule spread by an order of magnitude and merely exceeds
  it; twin-fitted weights become a caveat rather than a disqualifier.
- **0.936 or below** — the missing seat carries most of the 0.009 and `twinfive_v1`'s twins were
  nearly as good as the shipped members. That would mean the seat-count axis is NOT flat at six
  after all, contradicting seven entries, and the first thing to re-read is those entries rather
  than this one.
- **above 0.944** — dropping a champion seat IMPROVES the board. Then `final_ws_cf` is a negative
  seat at the shipped pool, which no offline reading could have shown (it is `--fold -1` and dark),
  and the seating axis re-opens on the subtraction side rather than the addition side.

COST: one slot. No card — every member already exists and is already on `comp/members.roster`.
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

    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - the kit is on sys.path only after boot
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same

    # The champion's six with `final_ws_cf` dropped: the shipped counterparts of `twinfive_v1`'s five.
    # No offline number is possible - four of these are `--fold -1`, so every holdout study is
    # in-sample for part of the pool, and recording a twin's score here would poison `board_transfer`.
    chosen = (
        "calib_span_ff",
        "grade_ladder_cf",
        "merged_blend_cf",
        "lr4e-5_ws_cf",
        "through_dense_cf",
    )
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage())).run()


if __name__ == "__main__":
    main()
