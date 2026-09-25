"""spanseat — the same seat holdswap vacated, filled with a different KIND of member.

**WHY THIS ENTRY EXISTS AT ALL.** `holdswap_v1` read 0.944 against the champion's 0.944 hours before this
was written: exactly zero, to the board's full precision. It vacated `lr4e-5_ws_cf` — the seat ten of ten
proxy candidates unanimously named as the weakest — and filled it with `calib_dense_hold_ff`, the project's
best solo member, priced top-of-table at +0.0022 offline. Nothing moved. That is the THIRD flat on the
seating axis: `raptorweight_v1` moved the foreign share by 2.3x for one unresolvable tick, `vjepaseat_v1`
ADDED the most architecturally distant member the project owns for -0.001, and `holdswap_v1` EXCHANGED a
seat for the best-priced alternative for 0.000. Reweight, add, swap — no resolvable move on any of them.

**SO THIS IS NOT ANOTHER SWAP OF THE SAME KIND, AND IT MUST NOT BE READ AS ONE.** What the three flats
establish is that the ARRANGEMENT of near-equal seats is not a lever. Every one of them moved a member
inside the pool's own quality band and inside its own pixel group: `calib_dense_hold` sits at nearest
agreement 0.9011+ like all fourteen strong members and has a measured twin. This entry moves the SAME SEAT
holdswap vacated and fills it with a member from OUTSIDE that group in two directions at once.

**THE ENTERING MEMBER, AND THE CONJUNCTION THAT MAKES IT DIFFERENT.** `merged_span_ff` is the blind
full-fit of `merged_span_cf`, and it is the only candidate this project owns that is simultaneously:

  * EXTERNAL-CORPUS. Fitted on `merged_blend_v1.csv` — the blend carrying external expert image reads
    (Rijeka) — not on the report labels every other champion seat learned from.
  * IN AN UNOCCUPIED DECODE CELL. 20 slices at band 0.2-0.8. No champion seat reads the stack that way.
  * AT CHAMPION QUALITY. Its fold-0 parent scores 0.8618, within a whisker of the best solo the project
    owns, and is greedy seat TWO of the 111-member curve with the largest single jump anywhere in it
    (0.8634 -> 0.8709, +0.0075).

That conjunction is exactly what `experiments/README.md` names as the route past the member-quality
ceiling: "a seat that is BOTH high-quality and in a pixel group nothing else occupies — which is external
data and a different decode path in the same member, not another term on an existing seat." No entry in 78
recorded submissions has tested it.

**WHY THE THREE FLATS DO NOT ALREADY ANSWER IT.** `vjepaseat_v1` is the closest prior and it is the one to
argue against: it added the most distant member owned and read flat. But that member was QUALITY-BOUND at
0.8337 solo, and the row's own conclusion was a SIZE — a member at that correlation needs roughly +0.017 to
+0.020 solo over the lane's 0.8331 to buy a seat rather than sit inert in one. `merged_span_cf` clears that
by construction at 0.8618. So `vjepaseat` established that distance without strength does not pay; it said
nothing about distance WITH strength, because the project had never shipped one.

**WHAT THIS ENTRY DOES NOT ISOLATE, stated before the number.** The champion already holds `merged_blend_cf`,
so this makes TWO merged-corpus seats out of six rather than introducing the corpus. The external-data term
is therefore a DOSE increase, not a first exposure, and the cleanly novel term is the decode cell. A
positive read here cannot be attributed to the corpus alone, and the honest description of the moving
variable is "external-corpus dose 1 -> 2, plus one decode cell the roster did not occupy". The matched pair
that makes it readable at all is `holdswap_v1`: the same seat, vacated the same way, filled by a member
from inside the pool's group instead of outside it.

**ONE MEMBER MOVES AND NOTHING ELSE DOES.** `lr4e-5_ws_cf` leaves, `merged_span_ff` enters, in the same slot
of the same tuple. Same six-then-blend shape, same `overlap=False`, same `LogitAverage`, same
`FOREIGN_WEIGHT = 1/3`, same raptor checkpoint. The entrant is `--fold -1 --ema`, so it is BLIND like every
other seat here — a fold-0 member on a champion-shaped entry is the mistake `greedyseat` already paid for,
0.936 against this same champion's 0.944.

**THE HONEST LIMIT.** `merged_span_ff` has no holdout number and can never have one; its parent's 0.8618 is
a fold-0 number belonging to a fold-0 member and may not be compared across the `--fold -1` line. So this
entry is fired on a MECHANISM and a matched control, not on an offline prediction. `farseat_v1` is the
standing warning that a board number can come from an entry that was never assembled — `run_check --log`
runs BETWEEN the kernel completing and this slot being counted, for that reason.

PRE-REGISTERED AS DELTAS AGAINST A NAMED CONTROL, which is the fix for the bug this project has now made
twice. `holdswap_v1`'s bands were written against "the champion's 0.944" in prose and anchored at 0.941, so
its "real gain over the champion" band opened at 0.9425 — BELOW the control — and a score identical to the
control landed inside a band labelled as a gain. `raptorweight_v1` made the same error with an edge at
0.9435, a score the board cannot express because it quantises to 0.001. Bands here are DELTAS; the absolute
edges are the control plus the delta, and every edge is a whole number of board ticks away from it.

CONTROL: `raptorweight_v1`, ref 55875622, board 0.944. `BOARD_RESOLVABLE = 0.0015`, board quantum 0.001.

- **delta >= +0.002 (s >= 0.946)** — the conjunction is REAL and above the resolution floor, and it is the
  first resolvable move on the seating axis in four entries. Re-pin `comp/champion.json`. The lane becomes
  corpus x decode, and the next arms are more cells of it (`merged_labelwise_cf` is already queued) rather
  than more member quality.
- **delta == +0.001 (s == 0.945)** — one board tick, UNDER `BOARD_RESOLVABLE` and a sixth of the 0.0062
  holdout floor. DIRECTIONAL ONLY, not a result and not a lane. It would license one confirming entry on a
  second cell of the same conjunction, and nothing else. Do not re-pin on the strength of the mechanism.
- **delta == 0.000 (s == 0.944)** — FLAT, and this is the branch that decides the most. Four operations on
  the seating (reweight, add, swap-near, swap-far-and-strong) would then have produced no resolvable move,
  which retires the SEAT as a lever entirely — including for the conjunction the route statement names. The
  consequence is specific and must be taken: "a different decode path in the same member" is refuted as a
  seat-level mechanism, and what remains is a genuinely different EXTERNAL CORPUS (MRNet, `knee-ombx`) or a
  roster-wide change of the kind `calibratedsix` made, which is the only operation that has ever moved this
  board off a plateau.
- **delta <= -0.001 (s <= 0.943)** — the swap COSTS. Suspect the BLIND REFIT before the cell: the parent is
  fold-0 and this is its first full-fit, so a training failure is the cheaper explanation than the seat
  being harmful. Check the checkpoint's own loss curve and its `witness_check` before concluding anything
  about corpus or decode. Keep the champion pinned.
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
        "merged_span_ff",
        "through_dense_cf",
    )
    SeatedEntry(
        names=chosen,
        # A quality-weighted share of two SYSTEMS, not `raptorseat_v1`'s 1/7 seat-count convention.
        weight=1.0 / 3.0,
    ).run()


if __name__ == "__main__":
    main()
