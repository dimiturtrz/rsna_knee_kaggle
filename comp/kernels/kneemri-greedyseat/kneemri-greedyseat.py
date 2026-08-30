"""The seven seats a GREEDY SEARCH named over 111 members, shipped to buy back the offline ruler.

**THIS ENTRY IS AN INSTRUMENT, and its purpose is not the score.** `comp/champion.json` records that
two pins in a row bought board position by spending the ability to measure: six of `raptorseat_v1`'s
seven seats are `--fold -1`, so the pinned champion has no offline number by construction and
`board.jsonl` writes its offline as null. `kneemri.kaggle.entry_offline` states the consequence from the
other side and calls it structural rather than a backlog: *"The compression ABOVE 0.937 stays unmeasured
and structurally so - that is the finding, not a gap to fill later."* `priceable_entries` names the
highest FULLY scoreable entry as `gradeswap_v1` at 0.933. Every roster decision above that is made blind,
against a board that answers five times a day.

Every one of these seven seats is fold-0 and carries a `HoldoutRecord`, so this entry can be
seat-scored, curve-scored and distance-scored - which the champion cannot. If it lands where predicted it
is the first PRICEABLE entry above 0.937, and it converts future seat decisions from a slot each into a
free offline read.

**IT ALSO TESTS A MECHANISM THAT HAS NEVER BEEN ON THE BOARD.** Every entry this project has shipped was
hand-picked. These seats were chosen by the greedy in `logs/greedy_named.log`, which searched 111 members
and reached 0.8768 at seven.

PRE-REGISTRATION (knee-jsgb), written before the number, from two fold-0 anchors in `board.jsonl` mapped
at `board_transfer`'s top-band compression of +0.89 to +0.97 board points per holdout point. The
seven-member anchor 0.8712 -> 0.933 gives 0.938; the top anchor 0.8751 -> 0.937 gives 0.9385. The two
agree, and the fold-0 dose they carry is the one `priceable_entries` measured as saturated at about
+0.003.

- **0.943 or above** - procedural selection matches hand-picking AND the fold-0 penalty is smaller than
  believed. Two results at once; re-read the all-data boundary.
- **0.936-0.942** - AS PREDICTED. The anchor lands, the ruler is bought, and no score claim is made.
  This is the expected outcome and it is a success for the stated purpose.
- **0.930-0.935** - the anchor still lands and is still worth having, but the fold-0 dose is larger at
  the top than the +0.003 already measured, which is itself a recorded finding.
- **Below 0.930** - procedural selection is WORSE than hand-picking by more than the dose explains.
  Suspect the greedy overfitting its own in-sample selection over 111 members before blaming the seats.

**WHAT IT DOES NOT LICENSE.** The greedy run is IN-SAMPLE, and it seats the pool's two WORST members at
10 and 11 inside a plateau where +0.0024 buys twenty-five seats. A good number here does not turn that
order into a ranking. It does not touch the pin: `raptorseat_v1` at 0.943 stays champion.
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

    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - same
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - the kit is on sys.path only after boot

    # The greedy's first seven over 111 members (logs/greedy_named.log), named rather than ranked so the
    # entry stays quotable. All seven are fold-0 and carry holdout records, which is the point of it.
    chosen = (
        "calib_dense_hold_s2026",
        "merged_span_cf",
        "labelwise_dense_s2026",
        "blendhold_s2026",
        "finding_calib_dense_s2026",
        "margin_seat_s2026",
        "warm_calib_dense_s2026",
    )
    # LogitAverage, not the RankAverage default: the champion serves it, and holding the vote rule is
    # what leaves SELECTION as the only moving variable. Version 1 of this kernel served the default
    # and was rerun before any slot was spent.
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage())).run()


if __name__ == "__main__":
    main()
