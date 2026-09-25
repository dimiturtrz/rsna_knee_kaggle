"""One of ours and one of theirs, at half a seat each — does distance substitute for count?

**WHAT IS BEING PRICED.** Not a champion attempt. This entry is an INSTRUMENT, and its expected board
score is BELOW the pinned 0.943. What it buys is the one number that decides where every remaining slot
goes: whether an ensemble's bonus comes from HOW MANY members it has or from HOW FAR APART they are.

Three of the four cells of that 2x2 are already on the board, all measured on the same ruler within
thirty hours of each other:

| entry | roster | board |
| --- | --- | --- |
| `calibsolo` (55828490) | `final_ws_cf` alone | 0.923 |
| `raptor_solo` (55829913) | `raptor_v4_swa` alone | 0.923 |
| `calibsix` (55810756) | six of ours, one architecture | 0.941 |
| `raptorseat_v1` (55831424) | those six + raptor at 1/7 | 0.943 |
| **this entry** | **`final_ws_cf` + raptor, 1/2 each** | **?** |

The two solos land on the SAME number, which is what makes the fourth cell readable: the pair holds
member quality fixed at exactly the value the six-member roster's own best member has, and varies only
the count (6 → 2) and the distance (one architecture → two). `calibsix`'s +0.018 over `calibsolo` is
therefore the count-and-recipe term measured at n=6, and this entry is the distance term measured at
n=2. If the second is anywhere near the first, six near-identical members were never the point.

**WHY THE BOARD AND NOT A HOLDOUT.** The raptor arm is `--fold -1`: every study of our holdout sits
inside its training set, so no offline pairing with it exists and none can be manufactured. This is the
same constraint that forced `raptorseat` onto a slot, and it is why the weight is NAMED rather than fit.

**WHY 1/2 AND NOT A FITTED WEIGHT.** Two members, one seat each. The two solos are equal on the board to
three decimals, so there is no quality argument for tilting it either way, and a tilt fitted after the
number is the thing the `raptorseat` pre-registration already refused. Half is what "one seat each" means.

**RANK SPACE.** Unchanged from `kneemri-raptorseat`, and load-bearing here rather than incidental: two
tables, two unrelated calibrations, equal weight. In probability space the more confident head would take
the blend outright and the result would report a difference in confidence as a difference in diversity.

PRE-REGISTRATION, written before the number, against `BOARD_RESOLVABLE = 0.0015`, `calibsolo` = 0.923 and
`calibsix` = 0.941.

- **0.941 or above** — two distant members match or beat six near-identical ones. Distance dominates count
  outright, the monoculture is the entire plateau, and every remaining slot and every remaining card-hour
  goes to ACQUIRING distant trunks rather than to reseating this pool. This is the outcome that reroutes
  the project.
- **0.933-0.940** — distance is worth substantially more per seat than count, without fully replacing it.
  Licenses a foreign-heavy roster (`RaptorSeat.fleet` over the distance-selected families) as the next
  build, and demotes further all-ours member fitting.
- **0.926-0.932** — an ordinary two-member bonus. Distance is real but not special at low n; `raptorseat`'s
  1/7 seating was already the right way to spend this member, and the roster stays as it is.
- **0.918-0.925** — no pairing bonus at all. Says `calibsix`'s +0.018 was bought by COUNT, that the twelve
  flat readings were measuring a saturated count term rather than a saturated diversity term, and that a
  foreign member is worth one seat and no more.
- **Below 0.918** — the pair is worse than either member alone. That is not a diversity verdict; it is a
  harness suspicion. The join, the ranking, or the metric's behaviour under a two-way average is wrong,
  and the reading is void until `RaptorSeat.combined`'s study join is re-verified against this entry's
  own tables.

**WHAT THIS ENTRY CANNOT DO.** There is no all-ours two-member board entry, so the count term at n=2 is
inferred from the n=1 and n=6 endpoints rather than measured. A reading in the 0.926-0.932 band is
therefore consistent with both "distance is ordinary" and "our own pairs would do this too", and telling
those apart would cost a further slot on `final_ws_cf` + one more of ours. That slot is not being spent
now, and this docstring says so before the number rather than after it.
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

    from kneemri.kaggle.seated_entry import SeatedEntry  # noqa: PLC0415 - the kit is on sys.path only after boot

    # `final_ws_cf` and nothing else: the member `calibsolo` served alone for 0.923, so the pair's
    # our-side quality is a MEASURED board number rather than an assumption about the roster's average.
    chosen = ("final_ws_cf",)
    SeatedEntry(
        names=chosen,
        # One seat each. Two members, equal on the board to three decimals; see the module docstring.
        weight=0.5,
    ).run()


if __name__ == "__main__":
    main()
