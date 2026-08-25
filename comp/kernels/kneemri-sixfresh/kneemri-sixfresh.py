"""The same six-member test as `sixstrong`, with a sixth that shares NO recipe with the five.

Fired in the same hour as `kneemri-sixstrong` and designed to be read WITH it, not against it. Both are
the champion `basedense` five plus one strong sixth, same RankAverage vote, same `overlap=False`, nothing
else moved. They differ in one property of the sixth, and that property is the confound `sixstrong` names
in its own docstring.

`sixstrong`'s sixth is `band_span_s2026` (0.8565). `calib_span_s2026` — an incumbent — IS `band_span`'s
recipe with calibrated targets, so that entry adds a near-twin of a seat the mix already holds. If it
lands low, "six is past the useful count" and "the sixth was redundant" both explain it and one slot
cannot separate them.

THIS ENTRY'S SIXTH IS `lr4e-5_ws_s2026`, holdout **0.8525**. It shares no recipe with any of the five:
336 px where they read 336 at 20 slices on a 0.10-0.90 band, the 0.20-0.80 band none of the six-member
incumbents use, and the encoder-rate lever (8e-6 -> 4e-5) that knee-wnkv measured at +0.0068 and which
`res392`'s resolution credit turned out to be. Its solo score is 0.8525 against `band_span`'s 0.8565 —
0.0040 apart, under the 0.0062 floor — and above `base_dense`'s **0.8512**, the member that carried the
0.931. So the two sixths are matched on quality and separated on recipe overlap.

THE OFFLINE READING, WITHOUT ROUNDING. As a sixth it reads **0.8687** against the champion five's
**0.8690**: **-0.0003**, a twentieth of the noise floor, and one of fourteen mounted candidates priced
tonight whose whole distribution spans +0.0002 to -0.0019. Nothing clears the +0.0015 `BOARD_RESOLVABLE`
threshold. This entry is not fired on an offline edge and does not claim one.

WHAT THE PAIR BUYS THAT EITHER ALONE DOES NOT. If BOTH land at 0.931 or above, member count is not capped
at five and the lane reopens regardless of recipe overlap. If BOTH land at 0.929 or below, the count
itself is the cap — the strongest available reading, because two sixths differing in recipe, band, pixel
count and rate produced the same verdict. If they SPLIT, the count is not the cap and redundancy is:
whichever sixth is the near-twin of an incumbent is the one that lost, and the next entry is a seventh
chosen for recipe distance rather than for solo score.

THE HONEST WEAKNESS. `basedense`'s 0.931 is a single board reading and `BOARD_COST_MULTIPLE` says this
board understates a negative offline delta by about 3x while `BOARD_RESOLVABLE` (+0.0015) says it cannot
resolve a positive one this small at all. Both entries here sit inside that blind zone by construction —
they are spent on a question the holdout has already been shown to misorder (the champion scored 0.931 at
offline 0.8690, BELOW `longseat`'s 0.8691 which scored 0.930), not on a delta either ruler can see.
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
        "lr4e-5_ws_s2026",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
