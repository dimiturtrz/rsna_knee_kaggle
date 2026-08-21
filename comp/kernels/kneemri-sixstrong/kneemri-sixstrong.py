"""The champion five plus a SIXTH STRONG member — the count axis, with quality held high.

The board has now priced member COUNT three times and every reading is confounded by member QUALITY.
`longseat`'s four strong members scored **0.930**. `crossenc` added a fifth whose solo holdout is
**0.8268** and scored **0.926**; `maxdiv` added a sixth at **0.8345** and scored **0.925**. Then
`basedense` added a fifth at **0.8512** — the first non-weak addition — and scored **0.931**, the board
maximum. So every count increase that lost also lowered the pool's floor, and the one that did not lose
did not lower it. Nothing on this board separates "six members is past the useful count" from "the two
sixths we tried were weak".

THIS ENTRY IS THE CHAMPION `basedense` FIVE WITH ONE MEMBER ADDED AND NOTHING ELSE MOVED — same
RankAverage vote, same `overlap=False`, same five incumbents, so its difference is attributable to the
sixth alone. The sixth is `band_span_s2026`, holdout **0.8565**: the third-best solo checkpoint the
mounted pool holds, 0.030 above `dinov3_base_lora` and 0.022 above `crop80_tissue`, and the member the
23-checkpoint greedy curve takes SECOND with the largest single addition in that curve (+0.0062).

THE OFFLINE READING, STATED FIRST AND WITHOUT ROUNDING. Rank-averaged over the 909-study fold-0 holdout
the six read **0.8692** against the champion five's **0.8690** — **+0.0002**, a thirtieth of the 0.0062
three-seed noise floor and well under the +0.0015 `BOARD_RESOLVABLE` threshold under which `seat_value`
has never been shown to pick a board tick. It is also the BEST of twenty candidates: every one of the
fourteen mounted members not already named was priced as an add and as a swap tonight, and the whole
distribution spans +0.0002 to -0.0019. The offline ruler is exhausted on this pool and says so.

IT IS FIRED ANYWAY BECAUSE THE OFFLINE RULER HAS ALREADY BEEN SHOWN BLIND HERE. The champion itself
scored 0.931 at offline **0.8690**, BELOW `longseat`'s **0.8691** which scored 0.930 — the board and the
holdout disagreed in ORDER at exactly this margin, on exactly this question. A slot spent to move a
number the holdout can already see would be wasted; this one is spent on the question the holdout has
demonstrably mispriced.

PRE-REGISTERED. **0.931 or above** says the useful count is not capped at five when the sixth is strong,
which reopens adding members as a lane and makes the next question the SEVENTH — and, read against
`crossenc`/`maxdiv`, isolates member QUALITY rather than COUNT as what those two entries actually
measured. **0.930** is a tie with `longseat` and says a strong sixth is free but not paid — the lane is
neutral and swaps are the only remaining move. **0.929 or below** closes the lane properly: six members
costs even at the pool's third-best quality, the count itself is the cap, and every future entry is a
five-seat swap or a new recipe. All three outcomes retire an open question; that is the reason this is
worth a slot when no offline candidate clears the bar.

THE CONFOUND, NAMED BEFORE THE FACT. `calib_span` IS `band_span`'s recipe with calibrated targets, and it
holds a seat in the incumbent five. So this entry does not add a wholly unseen pixel group — it adds the
UNCALIBRATED reading of a group the mix already sees once. If it lands at 0.929 or below, the honest
reading is not "six is too many" outright but "six is too many when the sixth is a near-twin of an
incumbent", and the count axis would want one more test with a sixth that shares no recipe with the five
before it is called closed.
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
        "band_span_s2026",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
