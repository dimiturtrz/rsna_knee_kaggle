"""relabel7 — the champion six plus the one member trained on a CORRECTED label table.

**WHAT THIS IS, and why it is not a reseating.** Seven board entries have varied WHO sits and HOW MUCH
each seat counts inside one fixed pool, and all seven read flat inside 0.937-0.944. This adds a member
the board has never seen, and the thing that makes it different is not its architecture, its resolution
or its pooling — every one of those is `band_dense`'s. It is its SUPERVISION. `errors_relabel_s2026` is
the only checkpoint this project owns that was fitted against a label table with 1,932 cells corrected
at the net-benefit-optimal budget of a confident-learning detector (Northcutt, Jiang and Chuang, JAIR
70:1373, 2021). Nothing else in the pool disagrees with the pool about what the TRUTH is.

**WHY IT IS BEING SHIPPED AFTER BEING CLOSED.** It reads 0.8522 solo against `band_dense`'s 0.8578 and
was declined on that -0.0056. That decision used the 909-study holdout, and on 2026-09-05 that ruler
failed its only external check: `colmcl` replaced the MCL column with a constant and read 0.904 against
the champion's 0.944, which under the metric `colnull` settled puts MCL's BOARD AUC at 0.980 (interval
0.968-0.992). MCL is our WORST column offline at 0.8095 and nearly the board's BEST. The offline map is
not a noisy board, it is a differently-biased ruler, and the bias has a measured direction: over the 58
radiologist-annotated studies the reader posts 108 false positives against 32 misses, all confident YES.

**THE MECHANISM, stated so it can be wrong.** Our holdout labels carry the SAME false-positive-heavy
noise as our training labels. A member fitted on corrected labels is therefore scored by a ruler that
still rewards the reader's over-calling, and its -0.0056 is the sign the mechanism PREDICTS rather than
evidence against it. If the board's labels are cleaner — and the MCL read says they are — the corrected
member is worth more there than here. That claim is unfalsifiable offline by construction, which is
precisely why it costs a slot and not an arm.

**WHAT MOVES AND WHAT DOES NOT.** The champion's six, in the same order, same `overlap=False`, same
`LogitAverage`, same kit, same raptor checkpoint at the same 1/3 weight. One name is APPENDED to
`chosen`. It is an addition and not a swap on purpose: a swap prices the corrected member against a
member whose own board contribution is unknown, and there is no offline number for any `_cf` seat to
anchor one, so the delta would be unattributable in both directions.

PRE-REGISTRATION, written before the number, against the champion's 0.944 and BOARD_RESOLVABLE = 0.0015.

- **0.9455 or above** — the corrected-label member is worth a seat the offline ruler said it was not.
  The supervision axis is OPEN and it is the first axis to move the board since the foreign seat.
  Re-pin `comp/champion.json`, and the next slot ships `errors_rank` (the same detector used to PRUNE
  rather than to correct) to separate correction from removal.
- **0.9455 > s >= 0.9425** — a gain at or above the resolution floor. Directional, not a result under
  the 1.5% bar, and it licenses ONE further slot on the axis rather than a re-fit of every member.
- **0.9425 > s >= 0.9425 - 0.0015, i.e. 0.941 <= s < 0.9425** — FLAT. Adding the corrected member
  changes nothing the board can see. That does NOT close the supervision axis, because a seventh seat
  diluted to 1/7 of our side can hide a real member; it closes THIS reading of it, and the follow-up
  that survives is a corrected-label member trained at `--fold -1` and shipped SOLO.
- **below 0.941** — the corrected member DILUTES. Our labels' false-positive bias is either not present
  on the board's labels or is not what limits us, and the whole supervision lane closes with one slot
  instead of the twelve a per-column read would have cost.

The refutation this cannot make: it cannot separate "corrected labels help" from "any seventh member of
this quality helps", because no seventh seat of comparable solo score has ever been added to exactly
this six. `dark6` is the nearest control and it removed a seat rather than adding one.
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

    chosen = (
        "final_ws_cf",
        "calib_span_ff",
        "grade_ladder_cf",
        "merged_blend_cf",
        "lr4e-5_ws_cf",
        "through_dense_cf",
        # The seventh, and the only one whose TRUTH differs from the other six: fitted against a label
        # table with 1,932 cells corrected by a confident-learning detector. Appended rather than
        # substituted, so the delta is this member's own contribution and not a difference against a
        # seat with no offline number of its own.
        "errors_relabel_s2026",
    )
    SeatedEntry(
        names=chosen,
        # A quality-weighted share of two SYSTEMS, not `raptorseat_v1`'s 1/7 seat-count convention.
        weight=1.0 / 3.0,
    ).run()


if __name__ == "__main__":
    main()
