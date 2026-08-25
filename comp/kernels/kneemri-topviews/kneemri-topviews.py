"""The 0.925 four, served under TWO rigid views instead of one — views carried up to the best entry.

A CONDITIONAL ENTRY. It exists so that the slot after `kneemri-rateviews` can be spent the moment that
one scores, and it should only be fired if views moved the pair: this kernel varies TWO things against
0.925 (the views, and nothing else) but only ONE thing against `kneemri-rateviews` (the membership), so
its number is readable either way — what it cannot do is establish the views axis on its own.

WHY THE FOUR AND NOT THE PAIR. `kneemri-rateviews` prices the axis at count two, where it is cheap. If
it pays, the question that decides a leaderboard entry is whether it still pays on the members that
actually score 0.925, and multi-view gains are exactly the kind that shrink as an ensemble already
averages away the variance the views were adding.

THE RISK WAS NEVER THE COST. Two entries returned COMPLETE with a BLANK public score, and the reading
that blamed decode time is refuted: the visible run costs 2.9 s a study, so ~11,000 studies fit inside
the 9 h ceiling. The cause was an OOM in `SpatialViews._warp`, which warped a whole chunk in one
`grid_sample` — ~6.2 GB of transient for a 48-study chunk on a 16 GB T4 — outside the row-halving retry
that wraps the forward, so nothing caught it and the hidden rerun wrote no submission (knee-c9q4).
`WARP_CHUNK` now bounds that at ~230 MB, bit-identically. The afford-sheds-views mitigation sheds on
projected TIME, so it never protected against this and is not what makes the entry acceptable; the warp
fix is, and this entry must only be fired from a kit that carries it.
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
    from kneemri.inference.test_time import SpatialViews  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same

    chosen = ("res392_ws_s2026", "lr1e-4_ws_s2026", "calibrated_targets_s2026", "lr4e-5_ws_s2026")
    views = SpatialViews(degrees=(0.0, -6.0))
    KernelRun(names=chosen, overlap=False, serving=Serving(views=views)).run()


if __name__ == "__main__":
    main()
