"""The ordinal ruler's RE-TEST, at a wider gap than the one that passed.

WHAT IT SWAPS. `gradeswap`'s exact seven — the 0.933 champion — with `lr4e-5_ws_s2026` out and
`small_accum_s2026` in. Count, kit (`cf6a99f`), vote and the other six held fixed. One variable.

**WHY THIS SEAT.** `gradeswap` cleared its pre-registered band at 0.933, which per `knee-7d1` licenses
re-ranking every seat by ordinal concordance. Doing that over all 67 same-split checkpoints puts
`lr4e-5_ws` at 0.770 — the LOWEST of the champion's seven — and `small_accum` at 0.784, the highest in
the entire pool. The displacement `gradeswap` made was +0.011 on this same scale and it moved the board
one tick; this one is +0.014, wider than the gap that passed.

THIS IS A RE-TEST, NOT AN APPLICATION. `knee-7d1`'s own caveat, carried verbatim rather than dropped:
one trial cannot separate "the ordinal ruler transfers" from "grade_ladder is simply a good member". A
pass licenses re-ranking; it does not prove the ruler. So this entry is the second, harder trial — the
same rule, a wider margin, a different member, against a stronger incumbent.

WHAT THE OTHER TWO RULERS SAY, because a disagreement is what makes it an instrument. `small_accum` is
0.8579 solo against `lr4e-5_ws`'s 0.8525, so the SOLO ordering agrees with the ordinal one here — a
weaker test than `gradeswap`, where they disagreed. The binary seat ruler prices `small_accum` at +0.0005
as an eighth, the best add of all 63 candidates scored on 2026-08-22 and still a third of its own +0.0015
resolution: it points the same way with a number it cannot resolve.

PRE-REGISTRATION, written before the number.

- 0.934 or above — the ruler transfers. Two independent forward tests, the second at a wider gap against
  a stronger incumbent. Ordinal concordance becomes the seat rule and every remaining unseated member is
  ranked by it.
- 0.933 — a tie. The champion's floor cannot be improved by a +0.014 ordinal gap, which is a statement
  about the SEAT (all seven are past the point where displacement pays) rather than about the ruler. It
  reads together with the count plateau: greedy peaks at 8-10 over a span of 0.0007.
- 0.932 or below — the ruler does NOT transfer, and `gradeswap`'s pass was `grade_ladder` being a good
  member rather than the rule finding it. Ordinal concordance goes back in the box beside residual
  distance, which has already failed twice in opposite directions, and the project is left with no
  working seat ruler at all — which is itself worth knowing, and is the honest reading of the
  eleven-for-eleven on `knee-jk3`.

+0.001 is one board increment against a 1.5% house bar. This arbitrates a ruler; it is not a claimed
improvement and must not be planned around as a number.
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
        "grade_ladder_s2026",
        "band_dense_long_s2026",
        "base_dense_s2026",
        "small_accum_s2026",
        "through_dense_s2026",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
