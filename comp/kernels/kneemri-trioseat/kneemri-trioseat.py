"""The champion's three seats that SHARE A PIXEL GROUP — three members for one decode.

**THIS ENTRY IS FOR THE EFFICIENCY TRACK, and it is the first one built for that board on purpose.**
`efficiency_read --live` on 2026-08-29 put us at efficiency rank 16 of 2,516 at 0.925 — the row
`soloseat` bought by serving `final_ws_e12` alone — and reported `0 teams rank above us on efficiency
while scoring BELOW us`. On 2026-08-25 that same line read 68. Runtime has stopped binding us;
score-at-this-runtime binds. The top 20 spans 0.925 to 0.948 with a median of 0.935, so we sit on that
band's floor and the only way up is a cheap entry that scores higher.

**WHY THREE SEATS ARE NEARLY AS CHEAP AS ONE.** Decode dominates a kernel here — a fleet pass measured
the card at 0% while pydicom worked — and `inference/ensemble.py` states the unit: *"the unit of
decoding is (scheme, slices, crop), and pixel groups remain the unit of [scheduling]"*. Members inside
one pixel group are decoded ONCE and then forwarded separately. `experiments/README.md:4743` records
that the champion six are already mixed across three groups, and that **`final_ws_cf`,
`grade_ladder_cf` and `through_dense_cf` all read 20 slices over band 0.1-0.9** — one group, one
decode. So this entry costs `soloseat`'s decode plus two extra forward passes on a card that was idle
during the decode anyway.

**WHAT IS HELD.** These are the champion's OWN seats, unmodified: same checkpoints, same
`--calibrated-targets` refit, same `overlap=False`. Nothing is retrained and no weight is tuned.

CORRECTED 2026-08-30 (knee-rxwc): this paragraph claimed `LogitAverage` was held too, and it was not.
`main` passes no `serving`, so this entry served the `RankAverage` default while the champion
`kneemri-raptorseat` passes `Serving(vote=LogitAverage())`. The 0.938 stands — `vote_ceiling1.log`
prices the two rules 0.8720 against 0.8710, about one tick and under the floor — but it is a
measurement of three champion seats under a DIFFERENT vote, and must not be quoted as the champion's.

The entry differs from `calibsix` by dropping the three seats that would each add a DECODE, and from
`soloseat` by adding two that do not.

PRE-REGISTRATION, written before the number, against `soloseat` = 0.925 (1 seat), `onepair_v1` = 0.936
(2 seats, one of them a foreign decode) and `calibsix_v1` = 0.941 (6 seats, three decodes).

- **0.938 or above** — three same-group seats reach `onepair` without paying for a foreign decode. The
  efficiency entry to pin, and it says seat COUNT buys score even at zero decode diversity.
- **0.930-0.937** — a real gain over `soloseat` for approximately one decode. The best cheap entry this
  project holds; pin it for the efficiency track and stop serving one member alone.
- **0.925-0.929** — seats that share a pixel group add almost nothing. That is a CONFIRMATION rather
  than a null: `knee-nas5` measured decode-path diversity as the only ensemble axis with a board
  transfer (0.0059 offline arriving as 0.014 on the board), and this entry has zero of it by
  construction. It is also the honest prior for this arm.
- **Below 0.925** — dilution; two calibrated seats can cost score when they see the same pixels.

**WHAT THIS ENTRY CANNOT DO.** All three members are `--fold -1`, so no offline score exists or can
exist for any of them and none may ever enter a pool that reports an offline mean. The board is the
only ruler this entry has. It also cannot separate "three seats" from "these three seats" — a different
trio inside the same group would need its own slot.
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

    # The champion's three 20-slice / 0.1-0.9 seats and nothing else. Named rather than ranked, so the
    # entry stays quotable: `solo_v4` used `members=1` and its identity is gone.
    chosen = ("final_ws_cf", "grade_ladder_cf", "through_dense_cf")
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
