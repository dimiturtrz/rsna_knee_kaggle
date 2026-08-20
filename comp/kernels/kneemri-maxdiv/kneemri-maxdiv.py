"""Every diversity axis this project owns, in one entry, against an offline ruler that scores it last.

Six members: the `longseat` four that hold the board maximum (`band_dense`, `calib_span`,
`depth_regions_narrow`, `band_dense_long`), plus `dinov3_base_lora_s2026` — the only DINOv3-base member —
plus `crop80_tissue_s2026`, the only member that reads a DIFFERENT PHYSICAL CROP. Same RankAverage vote,
same `overlap=False`. This is `kneemri-crossenc` plus one member, so the two entries are NESTED and their
difference is the crop axis alone.

THE OFFLINE READING, STATED AGAINST THE ENTRY. Rank-averaged over the 909-study fold-0 holdout these six
read **0.8668**, against **0.8691** for the four (-0.0023) and **0.8679** for the pinned champion
`greedy4_v1` (-0.0011). Greedy forward selection over the seventeen mounted members takes neither of the
two added here: `crop80_tissue` is picked SIXTEENTH of seventeen and the DINOv3 members last. Nothing in
this file claims the holdout likes this entry. It does not.

THE ARGUMENT IS THAT THE HOLDOUT IS BIASED ON PRECISELY THIS QUANTITY. Collapsing three members into one
pixel group cost 0.0059 here and 0.014 on the board — a 2.4x under-pricing of diversity by our own ruler
(knee-nas5, knee-7d1). The greedy that rejects these two members is a search run ENTIRELY under that
ruler, so "greedy takes them last" is not independent evidence against them; it is the same biased
measurement quoted twice. -0.0023 is a third of the 0.0062 three-seed noise floor.

WHY SIX AND NOT FIVE — THE DOSE IS THE POINT. `crossenc` adds one different member and is a minimal
probe. If diversity transfers at 2.4x, the effect should be MONOTONE in how much of it the entry carries,
and this entry carries both axes the mounted pool contains that no board entry has ever used. Count is
not the variable being tested: the count curve over the mounted pool is flat from four to seven
(0.8691 / 0.8692 / 0.8692 / 0.8691), so moving from four to six costs nothing by itself and every part of
the -0.0023 is attributable to WHICH members, not HOW MANY.

PRE-REGISTERED, REFUTATION FIRST. **0.929 or below refutes the diversity-mispricing hypothesis at its
strongest dose**, and read together with `crossenc` that is close to decisive: two entries, five and six
members, two independent axes, both under the champion would say the 2.4x is a fact about the decode axis
alone and that no recombination of the mounted pool is worth a further slot — the ledger's "exhausted"
verdict promoted from a claim about MOVES to a claim about the POOL. **0.932 or above** says diversity is
a board lever the offline ruler cannot see, and the roster's whole selection procedure has been fitted to
the wrong objective. **0.930-0.931 with `crossenc` also at 0.930-0.931** says the added members are free
rather than valuable, which still buys something: it means a STRONG member of a different family is worth
the card, because a weak one already costs nothing.

WHAT WOULD MAKE THIS ENTRY UNREADABLE, NAMED BEFORE THE FACT. If it lands below 0.929 while `crossenc`
lands at or above 0.930, the difference is `crop80_tissue` and the encoder verdict survives; the reverse
ordering isolates the encoder. If both land below 0.929 the pair is refuted together and no per-axis
attribution is available or needed.
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
        "dinov3_base_lora_s2026",
        "crop80_tissue_s2026",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
