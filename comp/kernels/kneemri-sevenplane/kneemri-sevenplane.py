"""The seventh seat again, with the distance moved to the INPUT PIXELS — `sevengrade`'s named follow-up.

WHAT THIS IS PAIRED WITH. `sevengrade` seats `grade_ladder_s2026`, whose distance is in the OBJECTIVE, and
its docstring names the confound it cannot resolve alone: a low reading there admits both "seven is past
the count cap" and "the severity objective does not transfer to this metric". This entry holds the
objective binary — every one of its seven members is fitted with the same loss — and varies distance at
the other end of the pipeline instead. A low reading HERE too is about the count; a recovery here is
about the ladder. The two entries separate what neither separates by itself.

THE SEVENTH IS `through_dense_s2026`, THE ONLY MEMBER WHOSE INPUT TENSOR DIFFERS. Every other member in
the pool — all seven small, one base — reads the SAME pixels, and their disagreement is recipe, band,
rate, resolution or encoder: all downstream of an identical input. This one puts slices i-1, i, i+1 in the
three channels, so the encoder is shown something different at the first op. That is the axis knee-nas5
measured this holdout as under-pricing by **2.4x** (0.0059 offline became 0.014 on the board), and that
measurement has never before applied at the INPUT rather than at the decode.

It also prices best. Against the 0.931 champion five it is **+0.0008** as an add and **+0.0010** taking
`base_dense`'s seat — the top of **twenty-four** candidates scored on 2026-08-21, mounted and unmounted,
against a distribution whose next value is +0.0005 and whose mounted best is +0.0002. Five times the best
move the rest of the mounted pool can make. It is still under the **+0.0015** `BOARD_RESOLVABLE`
threshold and is NOT claimed as an edge — but it is the strongest offline signal available, and the
ruler's structural blindness to distance (see `sevengrade`) means an add this novel is understated rather
than overstated.

ITS OWN ARM SAID NO, AND THE SEAT IS A DIFFERENT QUESTION. `through_dense` was fitted to test whether
stem inflation buys a depth derivative; it scored **-0.0020** against control, and the mechanism explains
why: the stem emits one token per patch and its three filters were pretrained to mean COLOUR, so it has
no capacity to carry "the slice at i" and "how it changes with depth" as separate quantities.
Conv2d at a window of 3 is Conv3d at a window of 3. That refutes stem inflation and leaves standing
exactly two things — a window WIDER than 3, and a stem emitting SEVERAL depth positions — which is
`knee-jucq`'s volumetric trunk. A refuted premise still produces a checkpoint, and the checkpoint is not
bound by the answer: `calib_span` (-0.0020) carried the 0.929 and `base_dense` (-0.0066) carried the
0.931. That is now three in a row.

THIS ENTRY IS GATED, AND MUST NOT BE SUBMITTED BEFORE ITS GATE READS. `through_dense_s2026` cannot be
served by the kit that produced the 0.931: `ba76fd1 through_plane`, the commit implementing the read it
was fitted with, is one of the nine the old kit lacked. It therefore requires the REFRESHED kit, and the
refresh rewrites `model.py`, `kernel_run`, `live_dataset` and `study_cache` under all seven members at
once. `sixfresh` v2 — identical code, identical members to the 0.931, refreshed kit, nothing else — was
submitted as the control that prices the refresh alone. **If v2 reads 0.931 the refresh is transparent
and this entry has one variable, the seventh member. If v2 reads anything else, this entry is
uninterpretable and must wait for the refresh delta to be understood.** knee-h6lr holds the sequence.

PRE-REGISTERED, against `sixfresh`'s 0.931 and whatever `sevengrade` reads. **0.932 or above** says
distance keeps paying past six AND that the input axis is where it is cheapest to buy — the encoder-input
lane opens, and the volumetric trunk enters as an eighth on the strength of an already-measured
under-priced axis. **0.931** ties: the seventh seat is free but not paid, and future members must
DISPLACE rather than add. **0.930 or below**, paired with a low `sevengrade`, says seven is past the
count cap regardless of where the distance sits — which makes `sixfresh`'s 0.931 this pool's ceiling and
moves the next work to member QUALITY rather than to roster composition.
"""

import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# >>> KERNEL-BOOTSTRAP
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


# <<< KERNEL-BOOTSTRAP


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    kit = find_kit()
    install_wheels(kit)
    sys.path.insert(0, str(kit))

    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - kit is on sys.path only after install_wheels

    chosen = (
        "band_dense_s2026",
        "calib_span_s2026",
        "depth_regions_narrow_s2026",
        "band_dense_long_s2026",
        "base_dense_s2026",
        "lr4e-5_ws_s2026",
        "through_dense_s2026",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
