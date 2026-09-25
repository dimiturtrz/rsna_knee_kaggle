"""lensseat — the champion seating with its most-swapped seat given to a different LENS.

**WHAT MOVES, and why it is not a sixth reseating.** Five entries now read 0.944 (`raptorweight`,
`holdswap`, `spanseat`, `stackseat`, `halfcham`), and the seating axis is closed by every one of them:
reweight, add, swap, count and serving precision all landed inside the board's 0.001 resolution. This
entry moves the same seat `holdswap_v1` and `spanseat_v1` already moved — `lr4e-5_ws_cf` — so the
CONTROL is exact and already on the board twice.

What differs is WHAT goes into it. `calib_dense_hold_ff` and `merged_span_ff` are both max-pooled
whole-stack members: the swap changed which studies a member saw, never how it reads a stack. Every
one of the champion's six pools slices with `max` or `attend`. `labelwise_dense_s2026` is the only
member in the roster fitted with `--slice-pooling labelwise`, and it is the project's third-best solo
at 0.8613.

**WHY THAT IS THE ONE SEATING OPERATION LEFT WORTH A SLOT.** `competence_axis` reads 81.7-82% of what
separates 105 members as a single scalar over 909 studies — which is the mechanism under all five
flats: rearranging members that differ along one axis cannot buy diversity, because there is only one
axis. A pooling function is not a point on that scalar; it changes which slice evidence survives to
the head at all. If the 82% reading is complete, this reads flat like the rest and the pool is closed
on a sixth independent direction. If a different lens is genuinely off-axis, this is where it shows.

**CONTROL.** `raptorweight_v1` 0.944, same kit, same six-plus-raptor count, same `overlap=False`, same
`LogitAverage`, same `FOREIGN_WEIGHT = 1/3`, same order. One name differs.

**PRE-REGISTERED**, against BOARD_RESOLVABLE 0.0015:
  * >= 0.9455 — the lens is off-axis, the competence scalar is not the whole story, and member KIND
    (pooling, readout) is the axis the next arms should sweep rather than corpus or schedule.
  * 0.9425-0.9455 — FLAT, and the sixth independent direction closing the pool. Combined with the
    82% scalar this retires member selection from the current roster entirely: no further slot goes
    to any arrangement of members that exist today, and every remaining slot waits on a new trunk or
    new supervision.
  * <= 0.942 — the lens COSTS. Read `labelwise_dense_s2026`'s per-label row first: it is a fold-0
    member and `greedyseat_v1` (0.936) is the only entry that has ever carried it, so a drop is at
    least as likely to be that entry's other six as this one's pooling.

**HARNESS.** `run_check` on the kernel log before the slot is spent: the census must read 6 of 6 and
name `labelwise_dense_s2026`. A member whose encoder will not mount is DROPPED silently and the
entry scores as five — which is how `farseat_v1` read `dark6`'s exact number for five days.
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
        "labelwise_dense_s2026",
        "through_dense_cf",
    )
    SeatedEntry(
        names=chosen,
        # A quality-weighted share of two SYSTEMS, not `raptorseat_v1`'s 1/7 seat-count convention.
        weight=1.0 / 3.0,
    ).run()


if __name__ == "__main__":
    main()
