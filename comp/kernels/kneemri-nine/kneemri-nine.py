"""nine — honest12 minus its collapsed seat and minus both rijeka-trained seats. Nothing else touched.

INSTRUMENT for BOTTOM-SEAT QUALITY, and simultaneously the first LICENCE-CLEAN entry this family has.
It is the follow-up `honest12`'s own docstring pre-registered and nobody took: *"one of them
(`mae_matched_s2026`, seat nine) reads 0.6735 solo … if this entry lands FLAT rather than up, that seat
is the first suspect."* honest12 landed 0.937 on 2026-08-27. The suspect sat in the roster twelve days.

THE OPERATION IS A PURE DROP — nine of honest12's twelve seats, same kit, same feed, same
`LogitAverage`, same order for the nine that stay. Three removed, and the two reasons are different:

- `mae_matched_s2026` (solo 0.6789) — the collapsed SSL-as-init member. **This is the measurement.**
- `merged_blend_cf`, `merged_dense_cf` — removed because `submit_check` REFUSES them: both train on the
  rijeka table, whose terms are unresolved against competition rule 2.5. Not a choice, a gate.

PRE-REGISTRATION, from the per-seat leave-one-out over all twelve of honest12's seats and from the
joint three-drop measured directly (researcher, D169 §1 and D170 §2):

```
  drop mae_matched_s2026   +0.0015 @l75                     <- the term being tested
  drop merged_dense_cf     -0.0003 @l75                     <- licence, and it subtracts
  drop merged_blend_cf     -0.0004 @l75                     <- licence, and it subtracts
  all three, joint         +0.0012 @l75   -0.0004 @mid      <- this entry
```

**+0.0012 offline at ledger75, untaxed** (a drop is not a selection over the pool), against honest12's
own board 0.937 → **~0.938**. The board quantizes at 0.001: UP reads 0.939+, FLAT 0.937-0.938. The
midpoint ruler reads the same operation at **−0.0004**, so the two rulers DISAGREE IN SIGN and the entry
is also a read on the recut.

**THE READ IS AMBIGUOUS ON THE FLAT BRANCH, AND THAT IS FORCED BY THE LICENCE GATE, NOT CHOSEN.** The
unambiguous instrument is honest12 minus `mae_matched_s2026` ALONE (+0.0015, one operation, no
cancellation). It was built first and `submit_check` REFUSED it: seating either merged seat trips rule
2.5. So a FLAT read here is consistent with *"the repair is worth nothing"* AND with *"the repair is
real and the two licence-forced drops ate 0.0007 of it"*, and nothing on the board separates them. That
limit is recorded here rather than discovered afterwards.

WHAT EACH BRANCH BUYS, and both are decisions rather than numbers:

- **FLAT or DOWN** — a member scoring 0.68 solo costs a twelve-seat ensemble nothing on the board. Then
  the board is insensitive to bottom-seat quality, **no card goes to member repair**, and the banked
  DROP term dies for one slot instead of for four arms.
- **UP** — the drop term is real, and it is *only* a member repair: the same table prices ten of twelve
  drops negative, so it generalises to no other roster and this is the last entry the drop axis needs.

WHAT THE ADD HALF IS AND IS NOT. An earlier draft of this entry claimed `distinct14` (honest12 plus two
DISTINCT-and-strong members, board 0.937 → 0.937) refutes the banked composite's ADD term. **It does
not**: those two adds are worth **+0.0002 offline**, a fifth of the board's own 0.001 grid, so that entry
could not have moved the board under any transfer and its flat carries no information (D170 §1). The add
term (+0.0021) remains UNTESTED. This entry carries the drop half alone because the drop half is the
larger one, not because the other was refuted.

WHY THIS IS NOT THE CLOSED RESEAT AXIS. All seven board flats ADDED, swapped, reweighted or re-lensed a
seat. **No entry this project owns has ever REMOVED one**, and none has measured what a collapsed member
costs on the board.

BASE: `kneemri-honest12`, board 0.937, ref 55817896, 2026-08-27 — the same twelve minus these three.
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

    chosen = (
        "calib_dense_hold_s2026",
        "finding_calib_dense_s2026",
        "blendhold_s2026",
        "margin_seat_s2026",
        "calib_span_s2026",
        "through_dense_s2026",
        "small_batch6_s2026",
        "base224_s2026",
        "calib_span_hold_s2026",
    )
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage())).run()


if __name__ == "__main__":
    main()
