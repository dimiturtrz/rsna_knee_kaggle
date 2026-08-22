"""The ordinal ruler's pre-registered forward test, and the severity displacement the ledger has owed.

WHAT IT SWAPS. `sevenplane`'s exact seven with `depth_regions_narrow_s2026` out and `grade_ladder_s2026`
in. Count, kit (`cf6a99f`), vote and the other six held fixed. One variable.

IT SETTLES THREE OPEN QUESTIONS WITH ONE SLOT, which is why this seat and not another.

**ONE — THE ORDINAL FORWARD TEST, UNRUN SINCE 2026-08-20.** `ordinal_signal` reads severity concordance
over 334,942 cross-level pairs without opening the labels file, and on the four-occupant seat experiment
it ordered the board EXACTLY (calib_span 0.759 > depth_regions_narrow 0.743 > band_span 0.736 against
0.929 / 0.928 / 0.927) where the binary ruler read alone ordered them backwards. A forward test was
pre-registered that day — take the next single-member swap where the rulers DISAGREE and decide it by the
ordinal one — and `knee-7d1` records it as unrun rather than refuted, because no two candidates in the
pool were separated by a margin either ruler could see. Scanning all 23 same-split members closes that:
`grade_ladder` reads 0.750 / 0.780 and `depth_regions_narrow` 0.743 / 0.769, a gap of +0.007 over labels
and +0.011 over pairs, against a seat experiment whose resolvable spread was 0.007 to 0.016. The margin
is finally inside what the ruler resolves.

`grade_ladder` is the HIGHEST-ordinal unseated member in the pool, and it beats two of the seven seated.

**TWO — THE BINARY RULER NAMES A DIFFERENT SEAT FOR THE SAME MEMBER**, which is the disagreement the
pre-registration required. Priced against this roster, `grade_ladder`'s best seat by seat value is
`lr4e-5_ws` at +0.0004, not this one; as a plain eighth it is +0.0001. Both are a quarter of the +0.0015
`BOARD_RESOLVABLE` threshold, so the binary ruler is not merely outvoted here — it cannot resolve the
question at all, and it points elsewhere with what little it has.

**THREE — IT IS THE SEVERITY DISPLACEMENT `sevengrade` WAS OWED, and it reads clean where that one could
not.** `sevengrade` seated `grade_ladder` as a SEVENTH and tied at 0.931, and its own ledger section names
the confound: at the time, a seventh seat of any kind was worth 0.000, so the tie could not separate "the
severity objective does not transfer" from "no seventh seat adds anything". `sevenplane` then refuted the
second half by moving 0.932 on a seventh, which narrowed the tie to evidence about the objective — and a
DISPLACEMENT removes the remaining confound entirely, because the count never changes. This is the entry
that section asked for.

**IT ALSO FIRES AGAINST THE RESIDUAL-DISTANCE RULER, DELIBERATELY.** Nearest-neighbour correlation is 89%
explained by solo score (pearson 0.890, n = 23); correcting for that, `grade_ladder` is +0.0187 — the most
redundant member of all 23 for its strength. The residual ruler therefore predicts this entry LOSES a
tick. `kneemri-twinswap`, built the same morning on that ruler's recommendation (`aug_long` for
`lr4e-5_ws`), is held unfired precisely because the ordinal ruler contradicts it: `lr4e-5_ws` reads 0.748
against `aug_long`'s 0.738, so the ordinal ruler says keep the incumbent. Two entries, opposite
recommendations, and the pre-registration from 2026-08-20 says which one to fire.

**PRE-REGISTRATION, WRITTEN BEFORE THE RESULT.** Above 0.932 confirms the ordinal ruler as the one that
transfers, licenses re-ranking every seat by it, and independently revives the severity axis the owner
kept alive after its own arm failed to clear. 0.932 is a tie and says a displacement cannot be resolved at
this count either. **Below 0.932 refutes the ordinal forward test on its first honest trial** — the
2026-08-20 ordering would then be the 1-in-6 coincidence its own interpretation warned it might be, and
the residual-distance ruler, which called `grade_ladder` the pool's most redundant member, would be the
one left standing.

+0.001 is one board increment against a 1.5% house bar. This is an INSTRUMENT that arbitrates three
rulers, not a claimed improvement, and it must not be planned around as a number.
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
        "grade_ladder_s2026",
        "band_dense_long_s2026",
        "base_dense_s2026",
        "lr4e-5_ws_s2026",
        "through_dense_s2026",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
