"""The champion six with the raptor member seated beside them, as a seventh seat in rank space.

**WHAT IS BEING PRICED.** Not the foreign member — `kneemri-raptor` priced that alone in the slot before
this one. This entry prices the DIVERSITY: whether a member trained by other people, on another
architecture, over a decode that is not ours, ADDS to a six-member roster that already agrees with
itself. That is the question `comp/champion.json` names as one of the two routes left toward 0.950,
and the one nothing in this repo has ever tested with a genuinely foreign member.

**EVERY PARAMETER IS FIXED BEFORE THE NUMBER.** The six are the champion's own tuple in the champion's
own order, served exactly as `kneemri-calibsix` serves them — `overlap=False`, `LogitAverage`. The blend
is `kneemri.kaggle.raptor_seat.RaptorSeat` at its default weight of 1/7: one member joining six takes
one seat of seven. Nothing here was tuned, because there is nothing to tune it on: the raptor arm is
`--fold -1` and every study of our holdout sits inside its training set, so the only ruler that can see
this blend is the board, and fitting a weight on the board costs one slot per value.

**RANK SPACE, NOT PROBABILITY SPACE.** Our table is a logit-average over six heads sharing one decode;
theirs is a sigmoid off a per-finding attention pool over a five-slot volume. The two calibrations have
nothing to do with each other and the metric reads rank order, so a probability-space average would let
whichever head is more confident decide the blend. `Vote.ranks` — the ranking every ensemble number this
project has reported was built on — does both sides.

PRE-REGISTRATION, written before the number, against the champion's 0.941 and `BOARD_RESOLVABLE = 0.0015`.
The raptor solo's own board score is the second input, and it is read from the entry before this one
rather than assumed here.

- **0.945 or above** — the diversity pays, and pays past what a better member could have bought. The seat
  is kept, `comp/champion.json` is re-pinned, and the next question is the weight rather than the member.
- **0.9425-0.944** — a real gain above the resolution floor. Kept as a seat; the follow-up is whether a
  second foreign arm (their `finespacing` or `arms` checkpoint) adds again or whether one foreign decode
  is the whole of the effect.
- **0.9395-0.9424** — flat. One seat of seven cannot move a roster by more than the floor, and the
  honest reading is that this member's disagreement with ours is not the useful kind. It does NOT license
  a larger weight fitted after the fact.
- **Below 0.9395** — the foreign member is DILUTING a roster that was already better than it. Consistent
  with a solo score under ours, and it closes the seating question rather than inviting a re-weighting.

**WHAT THIS ENTRY CANNOT DO.** It cannot separate "the member is weak" from "one seat of seven is too
small a weight". Both land in the flat band, and telling them apart needs a second slot at a different
weight — which is a decision to take after this number, not a reason to hedge the weight now.
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

    from kneemri.kaggle.seated_entry import SeatedEntry  # noqa: PLC0415 - the kit is on sys.path only after boot

    chosen = (
        "final_ws_cf",
        "calib_span_ff",
        "grade_ladder_cf",
        "merged_blend_cf",
        "lr4e-5_ws_cf",
        "through_dense_cf",
    )
    SeatedEntry(names=chosen).run()


if __name__ == "__main__":
    main()
