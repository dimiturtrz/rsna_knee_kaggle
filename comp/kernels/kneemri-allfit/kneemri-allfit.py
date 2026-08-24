"""Six of seven seats trained on ALL 4,406 studies — the one axis the offline holdout cannot see.

THIS IS AN INSTRUMENT WITH A CHANCE OF BEING A CHAMPION, which is rarer here than either alone.

**WHY THIS AXIS AND NOT ANOTHER.** `board_transfer` was refitted on 2026-08-24 against the `weakseven`
anchor and reads +2.14 board points per holdout point, not the 4.67 this project priced everything at
since 2026-08-16. Under the new constant, 0.934 -> 0.950 needs +0.0075 offline against a measured offline
ceiling of ~0.872 (+0.0026 over the champion). Every offline family is short by ~3x. But the same refit
showed the holdout is DIRECTIONALLY PERFECT where it can resolve at all — 0 inversions of 33 resolved
pairs — so the remaining hope is not a ruler that lies, it is a gain the ruler is STRUCTURALLY BLIND to.
There is exactly one: a `--fold -1` member holds out nothing, has seen every holdout study, and scores
`nan` forever. Data volume is the only axis whose gain no offline number can price.

**THE SINGLE VARIABLE.** `kneemri-finalfit` (0.934, the champion) is one all-data member and six fold-0
members. This is six all-data members and one fold-0. Same seven seats, same `RankAverage`, same kit,
same serving config; each entering member is its outgoing member's own flags plus `--fold -1 --ema` and
nothing else. `band_dense_long_ff` was never trained — `final_fit_six.sh` gates its whole chain on
finalfit reading 0.936 and it read 0.934, so the chain was stopped after member 5 — and its seat is held
by the fold-0 `band_dense_long_s2026`. Six of seven, stated so the composition is not read as an oversight.

**NO OFFLINE NUMBER IS REPORTED FOR THIS ENTRY AND NONE CAN BE.** Six of its members have seen the
entire 909-study holdout. A joint score would be a memorisation reading. The board is the only ruler
that can read this, which is the whole reason it is worth a slot.

PRE-REGISTRATION, written before the number, against finalfit's 0.934. Its one swap bought +0.001.

- **0.940 or above** — the data-volume gain COMPOUNDS across seats. Data is a live axis while every
  architectural family is closed, `band_dense_long_ff` becomes worth its 7.3 h after all, and the
  remaining fold-0 seats should follow.
- **0.936-0.939** — compounds SUB-LINEARLY. Six swaps buy roughly twice what one bought, so there is a
  shared ceiling: the members are drinking from the same extra 909 studies and largely agreeing about
  what they learned from them.
- **0.933-0.935** — a ONE-MEMBER effect. finalfit's +0.001 was the whole of it, and all-data training
  raises a single member without raising an ensemble. Data volume becomes a per-member ceiling rather
  than an additive axis, and the last blind spot closes with the rest.
- **0.932 or below** — the informative loss, and the one worth the slot on its own. All-data members
  are trained on the SAME 4,406 studies; fold-0 members differ partly by WHICH studies each held out.
  A regression here says a real share of this ensemble's diversity was FOLD diversity — accidental,
  and of the kind this project has ruled out as a lever on principle. That would relocate the whole
  diversity account, and no offline experiment can produce it.

There is no outcome on which this teaches nothing, and unlike `weakseven` there is one on which it wins.
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
        "final_ws_e12",
        "calib_span_ff",
        "grade_ladder_ff",
        "band_dense_long_s2026",
        "base_dense_ff",
        "lr4e-5_ws_ff",
        "through_dense_ff",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
