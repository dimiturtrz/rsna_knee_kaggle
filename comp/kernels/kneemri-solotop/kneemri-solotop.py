"""The top four by SOLO holdout, against the top four by JOINT holdout — one member apart, on the board.

`kneemri-rate` scored **0.924** with res392 + intensity392 + calibrated + lr4e-5, and it beat the
identical-count `kneemri-chosen` (0.918) by swapping one member. That swap is the only clean
single-member A/B this competition has given us, and it moved +0.006 — almost exactly the +0.0068 the
new member scored SOLO on the fold-0 holdout. The joint-subset search, which is what actually picked
the set, predicted +0.0013: five times too small.

So the two rulers disagree about which member matters, and only one of them has ever been right on the
board. Solo scores over the eight cached same-split members:

    lr4e-5 0.8525 · res392 0.8523 · calibrated 0.8493 · lr1e-4 0.8490
    sliceattn 0.8473 · extended 0.8462 · intensity392 0.8459

`intensity392_ws_s2026` is the WEAKEST member in the pool, and the joint search put it in the winning
four anyway — on the grounds that it decorrelates. This entry is that four with `intensity392` replaced
by `lr1e-4_ws_s2026`, the second rung of the encoder-rate sweep (0.8490 solo, +0.0033 over the control,
inside the floor of 4e-5 and therefore its equal on the evidence — knee-wnkv). The result is exactly
the top four by solo score, and it is the set the joint search ranks BELOW the one that scored 0.924.

PRE-REGISTERED, AND WRITTEN AFTER `kneemri-ratepair` CAME BACK: **0.927**. That entry ran the same
swap at count two (res392 out, lr4e-5 in) and scored 0.921 against the pair's 0.919 — a +0.0002 solo
gap paying +0.002 on the board, next to the +0.0052 gap that paid +0.006. Two points, both landing
within a thousandth of the displaced-vs-candidate solo gap, so this one's +0.0031 gap
(intensity392 0.8459 -> lr1e-4 0.8490) predicts +0.003 over 0.924. A score at or below 0.924 says the
decorrelation the joint search is buying is real and the rule does not survive its third point.

WHAT EITHER ANSWER BUYS. Above 0.924: the joint-subset ruler is costing us score by keeping weak
members for a decorrelation the board does not pay for, and every future entry is picked by solo
strength. Below it: decorrelation is real, the 0.924 was the new member and not the selection rule, and
`intensity392` earns its slot despite being last on its own. Either way one variable moves.
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

    chosen = ("res392_ws_s2026", "lr1e-4_ws_s2026", "calibrated_targets_s2026", "lr4e-5_ws_s2026")
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
