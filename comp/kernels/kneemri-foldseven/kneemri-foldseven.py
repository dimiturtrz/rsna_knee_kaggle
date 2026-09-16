"""foldseven — `noraptor_v1`'s seven seats with every DARK one replaced by a fold-0 member. An INSTRUMENT.

It is expected to score BELOW `noraptor_v1`'s 0.945 and is shipped anyway, because what it buys is the
RULER rather than a rank.

Four of those seven are `--fold -1` refits, so `comp/board.jsonl` records that entry's offline as null
BY CONSTRUCTION — as it does for every row at or above 0.943. So every arm this project prices is
priced through `board_transfer`, whose compression is fitted on the only four entries carrying both
numbers: 0.8705/0.935, 0.8711/0.936, 0.8729/0.936, 0.8751/0.937. Those four span **0.002 of board**
between them, and the constant read off them has been quoted wrong three times (+4.67, +2.14, +1.55).
A fifth point inside that span buys nothing. A point near 0.942 roughly TRIPLES the range the curve is
fitted on, and the range — not the n — is what makes an extrapolation to where we live unsupported.

Nothing else moves: same kit, same `LogitAverage`, same count, no foreign seat. Dropping raptor is not
a choice made here but the reason this kernel was cloned from `noraptor` rather than from the champion:
a board number containing a member the offline number cannot see is the defect that voided
`raptor_solo`'s 0.8938, and the quantity this entry exists to produce is a PAIR.

Three seats are their own fold-0 twins. `final_ws_cf` has none and never will — `final_ws_e12` held out
nothing — so the pool's best solo takes that seat; it is the one place this is not a twin substitution.

PRE-REGISTRATION, against `noraptor_v1`'s 0.945. The quantity read is the PAIR, so no branch is a
failure:

- **0.943-0.946** — the four seats' quality gain repays the all-data dose. The pair lands where none
  has ever been; refit the curve on it before pricing the next arm.
- **0.939-0.942** — the expected band: six all-data seats were worth ~+0.004, so four moving the other
  way should cost ~0.003. Still a 2-3x extension of the fitted range, which is what the slot bought.
- **Below 0.939** — the all-data axis is worth MORE than its own dose-response says, and the four
  existing pairs are the ones to distrust: each is a fold-0 entry read as though it priced an all-data
  one.
- **0.947+** — refutes the all-data axis outright. Re-pin, then re-read the dose-response.

Check `run_check`'s census FIRST: 7 of 7 loaded or this reads nothing.
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
        "calib_span_s2026",
        "stack_increments_s2026",
        "through_dense_s2026",
        "labelwise_depth24_drop40_long_s2026",
        "labelwise_ext_drop40_long_s2026",
        "labelwise_dense_drop40_long_s2026",
    )
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage())).run()


if __name__ == "__main__":
    main()
