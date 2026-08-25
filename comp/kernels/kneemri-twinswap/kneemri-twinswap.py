"""The first entry that REMOVES a twin instead of adding a stranger, at the count `sevenplane` won on.

WHAT IT SWAPS. `sevenplane`'s exact seven with `lr4e-5_ws_s2026` out and `aug_long_s2026` in. The count,
the kit, the vote and the other six are held fixed, so the single variable is which member sits in that
seat. Everything before this entry bought a seat by ADDING; this one asks whether a seat the pool already
holds is being spent on an ordering the pool already has.

WHY THAT SEAT. The distance ruler was built on the claim that a near-twin contributes one ordering twice
and a rank average cannot charge for it. Every entry so far has tested the claim from the ADDING side,
where a twin merely fails to help. If the claim is true it also has a subtraction side that has never been
read: replacing the most redundant SEATED member should pay more than adding anything.

AND `lr4e-5_ws` IS THAT MEMBER, WHICH IS NOT WHAT THE RAW COLUMN SAYS. Its nearest neighbour reads 0.9143,
second-highest of the seven — but nearest-neighbour correlation is 89% explained by solo score across all
23 same-split checkpoints (pearson 0.890, n = 23), because a weaker member's ordering is noisier and
decorrelates from everything. Roughly four fifths of that column is quality wearing distance's clothes.
Regressing it out (`nearest = 2.270 * solo - 1.038`, residual sd 0.0133) ranks members by how distant they
are FOR THEIR STRENGTH, and there `lr4e-5_ws` is +0.0173 — the most redundant member in the entire pool
bar one, and by a clear margin the most redundant of the seven seated.

`aug_long` IS THE OPPOSITE PROFILE, AND IT IS THE PROFILE THAT WON. Residual distance -0.0134, the second
most distant unseated member for its strength, at a solo 0.8511 that is not weak. Compare the member that
actually took 0.932: `through_dense` was -0.0123 residual at 0.8558 solo with a seat value of +0.0008.
`aug_long` is -0.0134 at 0.8511 with +0.0005 — the same shape, marginally further out and marginally
weaker. It is also the top seat value of all sixteen mounted-but-unseated members priced against the
sevenplane roster, as an add (+0.0004) and as this swap (+0.0005).

WHAT IT IS NOT. +0.0005 is a third of the +0.0015 `BOARD_RESOLVABLE` threshold, so this entry is NOT
claimed as an offline edge and the quality ruler cannot tell it from the incumbent. It is fired on the
twin argument with the quality ruler recording a tie, which is the same footing `sixfresh` was fired on
and the opposite of `sixstrong`.

READ IT AS AN INSTRUMENT WITH A REAL REFUTATION. Above 0.932 says redundancy is a cost the pool is
currently paying, which makes seat-by-seat residual distance a lever on every future roster and puts the
whole mounted pool back in play as displacements rather than additions. 0.932 is a tie and says the seven
are at their cap on this axis. **Below 0.932 refutes the subtraction side of the twin claim directly** —
`lr4e-5_ws` would then be earning its seat despite being the pool's most redundant member for its
strength, and the residual correction would be a worse ruler than the raw column it corrects, not a
better one.

AND THE RESIDUAL CORRECTION ALREADY FAILS ONE TEST, STATED BEFORE THE RESULT. On raw distance the ruler's
founding case reads `lr4e-5_ws` 0.9143 as more distant than `band_span` 0.9188, which is the ordering the
board produced (0.931 against 0.930). Correcting for strength INVERTS it: +0.0173 against +0.0127. So the
correction breaks the one paired observation the uncorrected ruler was built to reproduce, and this entry
is the test of which of the two is measuring the thing. Neither ruler orders the three tested sevenths
monotonically (`through_dense` 0.932, `band_span` 0.930, `grade_ladder` 0.931), so both are unvalidated
and one slot is what separates them.
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

    chosen = (
        "band_dense_s2026",
        "calib_span_s2026",
        "depth_regions_narrow_s2026",
        "band_dense_long_s2026",
        "base_dense_s2026",
        "aug_long_s2026",
        "through_dense_s2026",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
