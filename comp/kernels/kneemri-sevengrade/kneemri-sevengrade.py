"""The SEVENTH seat, chosen for recipe distance — the move `sixfresh` pre-registered before it scored.

WHAT THE PAIR MEASURED, AND WHY THIS IS THE FOLLOW-ON IT NAMED. `sixstrong` and `sixfresh` were fired in
one hour as the champion five plus one strong sixth, identical in every other respect and differing only
in whether the sixth's recipe was already seated. They SPLIT: `sixfresh`'s recipe-distant sixth
(`lr4e-5_ws`, solo 0.8525) scored **0.931**, the board maximum, while `sixstrong`'s near-twin of the
incumbent `calib_span` (`band_span`, solo 0.8565 — the BETTER checkpoint by 0.0040) scored **0.930**.
Member count is not the cap; redundancy is. `sixfresh`'s own docstring, written before either score
existed, says what comes next: *"the next entry is a seventh chosen for recipe distance rather than for
solo score."* This is that entry, and it is `sixfresh`'s exact six plus one, nothing else moved.

THE SEVENTH IS `grade_ladder_s2026`, AND ITS DISTANCE IS IN THE OBJECTIVE. Every seated member — all six
— is fitted with the same binary loss and differs downstream of it: band, rate, resolution, schedule,
calibration, encoder. This one is fitted on the SEVERITY LADDER, so its ordering is produced by a
different objective rather than by a differently-tuned path to the same one. Among the axes a member can
vary, the loss is the one no incumbent varies at all, and the ranking a different objective produces is
genuinely a different ordering rather than a rescaling of the seated one. It has never been mounted.

Solo it is **0.8565** — the pool's best solo, and 0.0040 ABOVE the 0.8525 sixth that just held the board,
so this entry does not trade quality for distance and a low reading cannot be blamed on a weak seventh.

WHY NOT `through_dense`, WHICH IS MORE DISTANT STILL. It is the only member whose input PIXELS differ
(slices i-1, i, i+1 in the three channels), and it prices best offline (+0.0008 as an add to the
champion five, the top of twenty-four candidates). It was this entry's first choice and was changed for a
plumbing reason, recorded here so it is not re-argued: the MOUNTED kit is nine commits behind, and one of
the nine (`ba76fd1 through_plane`) is the commit that implements the read `through_dense` was fitted
with. Shipping it means refreshing the kit, and the refresh touches `model.py`, `kernel_run`,
`live_dataset` and `study_cache` — the serve path of all seven members — so a 0.930 reading could not be
told apart from a code change. `grade_ladder` needs NONE of the nine, so this entry runs on the same kit
(e3c884e) that produced the 0.931. That keeps the single variable single. `through_dense` is queued for a
slot AFTER a kit refresh is itself verified against a known score.

THE OFFLINE READING, STATED AND THEN SET ASIDE. `grade_ladder` prices **+0.0001** as an add to the
champion five — inside the noise, well under the **+0.0015** `BOARD_RESOLVABLE` threshold, and not a
reason to fire. The reason it is set aside is stronger than the usual noise caveat: TONIGHT'S PAIR SHOWED
THE OFFLINE RULER POINTING THE WRONG WAY ON EXACTLY THIS AXIS. `lr4e-5_ws`, the sixth that took 0.931,
prices **-0.0003**; `band_span`, the sixth that cost a tick, prices **+0.0002**. Rank-averaging a
near-twin contributes one ordering twice, so the holdout barely moves whether an addition is redundant or
novel — it orders members by QUALITY and is structurally blind to DISTANCE, which is the property that
decides a seat. That is a ruler defect, not a resolution limit, and no number of seeds repairs it.

PRE-REGISTERED. **0.932 or above** says recipe distance STACKS past six, the pool is not exhausted, seats
are still buyable, and `knee-jucq`'s volume-pretrained trunk enters as an EIGHTH rather than having to
displace anyone. New champion, and the severity axis has earned a board result for the first time.
**0.931** ties `sixfresh` and the champion: a seventh distant seat is free but not paid, the add-a-member
lane has stopped producing ticks, and every future member must earn its place by DISPLACING an incumbent
— a materially harder bar that changes how the volumetric member is judged. **0.930 or below** says
seven is past the count cap even when the seventh is distant, which partially REOPENS count as the cap
and makes `sixfresh`'s 0.931 the ceiling of this pool rather than a step in a series.

THE CONFOUND, NAMED BEFORE THE FACT. The severity ladder has never scored on the board, so a low reading
admits a second explanation — "the OBJECTIVE does not transfer to this metric" — alongside "seven is too
many". The two are separable by the follow-up rather than by this entry: `through_dense` after a kit
refresh varies distance while holding the objective binary, so a low reading there too is about the
count, and a recovery there is about the ladder. Do not collapse them on one number.

Precedent for spending a slot on a member its own arm refused: `calib_span` (-0.0020 against control)
carried the 0.929 entry, and `base_dense` (-0.0066, outside the floor) carried the 0.931. An arm's
verdict is about a knob; a seat is about a mix, and this repository has now measured them apart twice.
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
        "grade_ladder_s2026",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
