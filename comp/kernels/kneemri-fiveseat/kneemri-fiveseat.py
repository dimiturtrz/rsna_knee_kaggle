"""Five whole-corpus seats on ONE decode, and the fifth is a different TRUNK.

`trioseat_v1` scored **0.938** with three whole-corpus members sharing one pixel group (20 slices, band
0.1-0.9, 336). `fourseat_v1` added a fourth on that same geometry. This entry adds a fifth, and the
fifth is the only member of read A's whole-corpus pool whose mechanism is not a supervision treatment:
`base_dense_ff` is a **DINOv2-base** trunk, fitted `--fold -1 --ema` on the identical decode.

**THE TERM IT SEPARATES, WHICH NO ENTRY ON THIS BOARD HOLDS.** `raptorseat_v1` (0.943) bought its tick
by adding an architecture AND a decode at once, and `fleettwo_v1` (0.942) showed a second foreign decode
adds nothing more. So architecture distance has been measured only when it arrived bundled with a
second decode, and the two terms have never been separated. A trunk swap INSIDE one decode is that
separation, on the cheap side: if it pays, architecture diversity is a seat-axis property and the
efficiency lane is much longer than one entry; if it does not, the +0.002 `raptorseat` measured belongs
to the decode, not to the trunk.

**AND ITS QUALITY IS KNOWN, WHICH IS WHAT `fleettwo` LACKED.** `fleettwo` could not attribute its null
to the seat versus to the member, because every raptor arm is `--fold -1` and unmeasurable. This member
has a fold-0 sibling: `base_dense` read **0.8512** against `band_dense`'s 0.8578, -0.0066 — slightly
weak, and measured. So a null here reads as "a weak-but-distant trunk does not pay for a seat", which is
a statement about the trade, not an unattributable shrug.

**WHY NOT ONE OF THE OTHER THREE.** Read A's whole-corpus pool is eight members on disk (knee-dbc3), and
the three not seated here — `final_ws_e12`, `grade_ladder_ff`, `through_dense_ff` — are each the SAME
supervision treatment as a member already seated, differing only in fit. Expect them to behave like
raptor's `v4_full` x `v4_full_swa` pair, which agreed at 0.9955 because it was one model twice. They are
not spent on a slot before an agreement matrix collapses them.

**ONE DECODE, FIVE MEMBERS.** Unchanged from `trioseat` and `fourseat`: all five share `(canonical, 20
slices, tie=first, 336)`, so the pixel pass is paid once and each member is a forward pass. This is the
Efficiency Track entry as much as the score entry.

**PRE-REGISTRATION, written before the number.** Base is `fourseat_v1` on the same geometry, and
`trioseat_v1` = 0.938 at three seats. The champion pin is `raptorseat_v1` = 0.943 across ~4 decodes.

- **0.944 or above** — the trunk swap pays inside one decode, and it pays MORE than `raptorseat`'s
  bundled architecture-plus-decode did. Architecture diversity is a seat axis. Re-pin, and the next
  slots go to further trunks inside read A rather than to further decodes.
- **0.941-0.943** — it pays, and lands level with the three-decode champion at a third of the runtime.
  A clear efficiency-track result and a flat score result; keep the pin, ship this to the efficiency
  board, and seat a second trunk only if a slot is idle.
- **0.938-0.940** — flat against `fourseat`/`trioseat`. The trunk swap adds nothing a supervision
  treatment did not already add, which says `raptorseat`'s +0.002 was the DECODE and not the trunk.
  Closes the cheap-architecture lane; do not seat another trunk.
- **Below 0.938** — the fifth seat DILUTES. Read with `fourseat`'s own number: if `fourseat` also sat
  below `trioseat`, the seat term is spent at three and this says nothing extra about trunks; if
  `fourseat` gained, then a weak trunk specifically is what costs, and distance without quality does not
  buy a seat inside one decode either.

**WHAT IT CANNOT DO.** Its members are all `--fold -1`, so no offline score exists for this entry and
none may ever be reported for it. And it moves one seat against `fourseat`, so it cannot separate "the
fifth seat" from "this particular fifth member" any better than `fourseat` could separate its own.
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

    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - the kit is on sys.path only after boot

    # trioseat's three, plus the one read-A seat that is both whole-corpus and already packaged.
    chosen = ("final_ws_cf", "grade_ladder_cf", "through_dense_cf", "small_batch6_ff", "base_dense_ff")
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
