"""SIX DINOv3-BASE ALL-DATA MEMBERS — the roster rebuilt in another family, not a seat swapped into it.

**THE OBJECT SEVEN FLAT ENTRIES LEFT UNTESTED.** dark6, dark7, allfit, lastseat, augseat, v3seat and
farseat read 0.936-0.937 while member COUNT, member IDENTITY, FOLD STATUS, REGULARISATION, ENCODER
FAMILY AT ONE SEAT (v3seat, DINOv3-base) and ARCHITECTURE FAMILY OUTSIDE the slice-then-pool trunk
(farseat, a V-JEPA2 tubelet transformer whose nearest-neighbour agreement is 0.8210 against a seated
band of 0.8801-0.9143) each moved alone. Adding the most distant available member buys what adding the
nearest one buys: nothing. That is not a diversity deficit — it says the SIX already determine the
answer and a seventh vote of any provenance is averaged into a decision that does not depend on it.

So the untested object is the six themselves. This is `final_fit_six.sh`'s six recipes, verbatim, with
`--variant v3base` on every one of them.

**WHAT ELSE MOVED, NAMED RATHER THAN BURIED.** The slice count: 12, not 20. `v3base_dense` peaks at
25.38 GB allocated at 20 slices and biohub holds 10.2 GB of this 32.6 GB card by arrangement, so 20 was
denied while it ran. `band_span` measured the density main effect at BOTH spans (+0.0014 at 0.1-0.9,
+0.0013 at 0.2-0.8), so what 12 slices forfeits is bounded at +0.0013 — a fifth of the 0.0062 holdout
noise floor, against a family term this roster exists to measure. Attribution is family + density, and
density is the small, measured half.

One recipe is not a copy. `base_dense_ff` was distinguished by `--variant base` (DINOv2-base); under a
family swap that axis collapses into the swap and would duplicate `v3_ws_ff`. It is replaced by
`--slice-pooling attend`, the slice combiner every other member holds at `max`.

**NO OFFLINE NUMBER IS REPORTED AND NONE CAN BE.** All six carry `--fold -1` and have seen every one of
the 909 holdout studies. Any joint score would be a memorisation reading. The board is the only ruler.

PRE-REGISTRATION, written before the number, against `dark6`'s 0.937.

  * **>= 0.945** — the family of the SIX is the plateau, and the next roster is built in whichever
    family wins rather than by seating dissenters into this one.
  * **0.940-0.944** — the family carries but does not close the gap to 0.950. A MIXED six, three of
    each, is then the arm that prices whether the gain is FAMILY or simply CHANGE.
  * **0.933-0.939 FLAT** — the expected reading. The plateau does not live in the encoder family at
    all. The ensemble axis closes in its strong form: neither the marginal seat NOR the whole roster's
    family moves this board, and every remaining path runs through data or labels.
  * **<= 0.932** — DINOv3-base does not survive all-data fitting at this depth. The harness is the
    first suspect, not the family: confirm `--variant v3base` reached the checkpoint config on all six
    before concluding anything.
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
        "v3_calib_ff",
        "v3_grade_ff",
        "v3_through_ff",
        "v3_ws_ff",
        "v3_attend_ff",
        "v3_focal_ff",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
