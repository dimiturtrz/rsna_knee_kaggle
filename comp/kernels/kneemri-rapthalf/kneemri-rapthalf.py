"""raptorweight — the champion seven, one dial moved: the foreign seat's WEIGHT, 1/7 -> 1/3.

**WHAT IS BEING PRICED, and why it is not a permutation.** Every entry since `raptorseat_v1` has varied
WHO sits (a second foreign arm in `fleettwo_v1`, fewer of ours in `trioseat_v1`/`fourseat_v1`), and all of
them read flat inside 0.937-0.943. Nothing has ever varied HOW MUCH the foreign seat counts. That dial has
one value in this project's history — `RaptorSeat.weight = 1/7` — and its own docstring says why it was
picked: *"One member entering a six-member roster takes one seat of seven, which is the weight every
seating entry in this project's history has used."* That is a CONVENTION, not a measurement, and the
convention's premise is wrong about what is being blended.

**THE MECHANISM.** The blend is not seven members; it is two SYSTEMS. Ours is a six-member logit-average
that scores 0.941 on the board. Theirs is ONE model that scores 0.923 on the board (`raptor_solo`,
55829913) — held level with our own best single by measurement, not assumption. Averaging a 0.941 system
and a 0.923 system at 6:1 gives the weaker-but-independent side a seventh of the vote when its quality
justifies something near a third. Under equal quality the variance-minimising split of two independent
rankers is even; ours is stronger and is itself an ensemble (lower variance), so a third is the honest
correction and a half would over-serve them.

**WHY 1/3 AND NOT A SWEEP.** Fitting this on the board costs one slot per value, which is exactly the
objection `raptor_seat.py` raises. So this spends ONE, at the largest step the mechanism supports rather
than the smallest step that could be detected: 0.143 -> 0.333 is a 2.3x move. If the axis has slope, one
slot finds it; if it reads flat across that span, the axis is closed with one slot instead of four.

**IT IS ALSO AN INSTRUMENT FOR knee-uco3.** That P0 reads `dark6` at 0.937 as evidence the SEVENTH SEAT
IS INERT — that the foreign table barely moves the blend at all — which would retroactively void every
seat-swap entry. An inert seat cannot care about its weight. So a flat result here CONFIRMS knee-uco3 and
a move in either direction REFUTES it. The entry pays twice.

**EVERYTHING ELSE IS THE CHAMPION, UNCHANGED.** The same six in the same order, the same `overlap=False`,
the same `LogitAverage`, the same kit, the same raptor checkpoint, the same rank-space join by study. One
float differs from `raptorseat_v1`, which is what makes the read attributable.

PRE-REGISTRATION, written before the number, against the champion's 0.943 and `BOARD_RESOLVABLE = 0.0015`.

- **0.946 or above** — the weight is a real lever AND it lands in the top ten (0.946 floor). Re-pin
  `comp/champion.json`, and the next slot sweeps to 1/2 to find the turn.
- **0.9435-0.9455** — a real gain over the champion, above the resolution floor. The dial matters; keep
  the seat at 1/3 and price 1/2 only if a slot is otherwise idle.
- **0.9405-0.943** — FLAT. The foreign seat is weight-insensitive across a 2.3x span. Combined with
  `fleettwo_v1`'s flat count reading, the foreign-seating axis is closed in BOTH of its dimensions and
  no further slot goes to one. knee-uco3's inert-seat reading gains its second independent support.
- **0.940 or below** — our six carry the entry and the foreign table DILUTES once it is given real weight.
  1/7 was at or near a maximum, `raptorseat_v1`'s +0.002 stands as the whole of the effect, and the
  correct move is to stop widening the foreign side rather than to narrow it further.

The refutation this cannot make: it cannot separate "the weight is flat" from "THIS foreign member is
flat". Every raptor arm is `--fold -1`, so none of them has an offline quality number, and `fleettwo_v1`
already carries the same limit for the count dimension.
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
    SeatedEntry(
        names=chosen,
        # A quality-weighted share of two SYSTEMS, not `raptorseat_v1`'s 1/7 seat-count convention.
        weight=1.0 / 2.0,
    ).run()


if __name__ == "__main__":
    main()
