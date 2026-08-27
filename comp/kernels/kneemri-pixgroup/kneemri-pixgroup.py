"""Two decode paths added to the champion six, because pixel geometry is the one axis the board pays for.

**THE PIXEL-GROUP ENTRY.** `calibsix`'s six seats are UNCHANGED — same members, same order, same
`LogitAverage`, same `overlap=False`, same kit. Two seats are ADDED: `res392_ws_s2026` (392 px) and
`crop80_span_s2026` (crop 80 mm). Both are already in the mounted `kneemri-members` package; this
entry costs a kernel version and a slot, no dataset push.

**WHY THIS AXIS AND NOT ANOTHER — every other seat-lane axis closed in the last 24 hours.**

- *Count* is closed on BOTH rulers. `calibmix` (2026-08-27) took the same six to eleven seats with
  same-family twins and read **0.941, identical to `calibsix`**. Offline, the honest quality-ordered
  seat curve reads six 0.8720 -> nine 0.8729 -> seventeen 0.8736 -> peak 0.8738 at twenty-four, a
  +0.0018 that is under a third of the 0.0062 three-seed floor and inside its own +0.0017
  in-sample/held-out selection bias.
- *Blind spread* LOSES. The flat-distance diverse curve peaks at 0.8650 across ninety-nine seats,
  **0.0088 behind honest** (1.4 floors), because distance-first selection elects the pool's worst
  members first: `focus_latmen` at seat two drops the running score 0.8634 -> 0.7501, and it takes
  twenty-five seats to get back to what one good member does alone.
- *Member quality* ceilings near 0.943. `calibsix` improved all six seats at once, pre-registered
  ~0.944, and read 0.941 — a 23% pass-through that does not compound.
- *Encoder size* is measured, and measured flat-to-down: twenty-three kernels name a DINOv2-BASE
  member, board-scored entries seating one span 0.934-0.939, and neither of our two best entries
  seats one.
- *Overlap TTA* is already refuted for whole-stack members (`knee-eeo`).
- *QualityWeightedRank* is structurally unfittable here — its weights need a holdout, and the champion
  seats are `_cf`/`_ff` full-fit with none.

**WHAT IS LEFT, AND WHY IT IS THE ONE WORTH A SLOT.** `knee-nas5` measured the decode-path /
pixel-group axis at **0.0059 offline against 0.014 on the board — a 2.4x compression**. That is the
only super-linear board transfer this project has ever measured (`board_transfer`'s pooled slope is
+0.74 board points per holdout point at and above 0.8648), and it is the one axis NEITHER offline
selector can see: both the honest and the diverse ruler score on our 909 shared studies, where decode
diversity is worth its offline 0.0059 and nothing more. The champion six are ONE pixel group. This
entry takes it to THREE.

**THE HONEST OFFLINE NUMBER, stated before the board number.** Adding these two seats is worth about
**+0.0009** on the offline ruler — free, and BELOW that ruler's own +0.0015 resolution. So the offline
evidence for this entry is approximately nil, and its entire case is the measured 2.4x board
compression of this specific axis. That is a bet, and it is priced as one: at the measured compression
+0.0009 offline lands near +0.002 board, and only if the 2.4x holds at the top of the board does it
reach the +0.004 that would matter.

**THE ADD HAS NEVER BEEN RUN.** `resseat` (0.938) tested `res392_ws_s2026` as a SWAP — it dropped a
full-fit member to seat it, so it confounded "a new pixel group enters" with "a good member leaves"
and a fold-0 member replacing a full-fit one. This is the pure ADD: nothing leaves.

PRE-REGISTRATION, written before the board number, against `calibsix` and `calibmix` at 0.941.

- **0.945 or above** — the pixel-group axis is a live lever and the 2.4x compression holds at the top
  of the board. Extend it with slot 4: a fourth pixel group, and refit the offline selector to weight
  decode distance rather than flat feature distance.
- **0.942-0.944** — directional only, inside the floor. Keep the two seats in the roster and stop
  spending slots on the axis; it is worth what the arithmetic said and no more.
- **0.939-0.941** — FLAT. The seat lane closes on the board exactly as it closed offline, the 2.4x
  compression does not survive at the top, and the remaining routes are external data and stacking.
- **0.938 or below** — the two fold-0 seats DILUTE, matching `resseat`'s 0.938 with the swap confound
  removed. The axis closes, and the reading is that fold-0 members cannot sit beside full-fit ones.

COST: one slot. No card, no dataset push, no new training — both members were trained and mounted days
ago.
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

    # `calibsix`'s six, seat for seat and in its order, then the two added pixel groups LAST so the
    # diff against the champion is two lines and no existing seat moves.
    chosen = (
        "final_ws_cf",
        "calib_span_ff",
        "grade_ladder_cf",
        "merged_blend_cf",
        "lr4e-5_ws_cf",
        "through_dense_cf",
        "res392_ws_s2026",
        "crop80_span_s2026",
    )
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage())).run()


if __name__ == "__main__":
    main()
