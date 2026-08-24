"""SEVEN dark members — `kneemri-dark6` plus one, and the only variable is the count.

**THIS IS THE UPPER RUNG OF THE NESTED COUNT CURVE `kneemri-dark6` opens.** It is that entry's exact six
names in that exact order, `RankAverage` unchanged, kit unchanged, serving config unchanged, plus
`small_batch6_ff` appended. Nothing is swapped, reordered or dropped. Ship `dark6` FIRST: without its
reading this entry has no control and its number means nothing.

**WHY A SEVENTH DARK MEMBER IS THE ONE UNTESTED VARIABLE.** Count-of-dark-members has been held fixed at
six by every board entry since `finalfit`, for the mundane reason that exactly six existed and no entry
had a seventh to offer. `small_batch6_ff` is the first — `small_batch6`'s own flags plus `--fold -1
--ema`, which is the strongest member this project owns (0.8598, greedy seat 1 of 67) and the member that
held the fold-0 seat in `lastseat` when it read 0.937, so its behaviour inside this ensemble is a
measured control rather than a new unknown.

**WHY THE OFFLINE POOL CANNOT ANSWER THIS AND NO NUMBER IS REPORTED HERE.** Greedy ranks members by a
holdout record; a `--fold -1` member holds out nothing and scores `nan` forever. Not one dark member has
ever appeared on the greedy curve, so the 2026-08-24 measurement that the curve is FLAT across 32 seats
(0.8731 peak, 0.0016 spread, 0.0062 noise floor) says nothing whatever about this axis. That flatness
retired member SELECTION as a lever. It cannot retire member COUNT on the one member type it cannot see.

**PRE-REGISTERED, against `dark6`'s reading and not against a champion:**
- **dark6 + 0.002 or more** — the all-data dose is still paying at the seventh seat, the curve has not
  flattened, and building an eighth dark member is licensed with a number under it.
- **within 0.002 of dark6** — the dose is spent. The full-fit axis closes on a MEASUREMENT rather than on
  the extrapolation that has stood in for one since 2026-08-22, and no further dark member is worth card
  time. This is the most likely branch and it is the deliverable either way.
- **dark6 − 0.002 or worse** — a seventh member of this type actively costs. That indicts `--ema` or the
  all-data recipe rather than the count, and the harness is the first suspect, not the idea.

**STALE-CONSTANT WARNING, since this file was cloned from `dark6` and that entry's header carries it:**
`board_transfer` reads **+1.82** board points per holdout point as of 2026-08-24, not the +2.14 quoted
below and not the 4.67 this project priced against for three months. It is fitted entirely on entries at
or below 0.933, so any figure derived from it at the top of the board is a floor, not an estimate.

The inherited discussion below is `dark6`'s and is kept because its reasoning about the blind axis is what
licensed this entry; read its arithmetic against +1.82.

--- inherited from kneemri-dark6 ---

Six of seven seats trained on ALL 4,406 studies — the one axis the offline holdout cannot see.

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

    # THE BOTTOM RUNG OF A STRICTLY NESTED COUNT CURVE. These are the SIX dark members and nothing
    # else. `allfit` is these six plus `band_dense_long_s2026`; `count8` is those seven plus
    # `small_batch6_s2026`. So the three entries are 6 ⊂ 7 ⊂ 8, each step ADDING one fold-0 member to
    # six seats that are byte-identical across all three. Nothing is exchanged and nothing is
    # reordered — the only thing that moves is how many fold-0 members ride along.
    #
    # WHY THIS AND NOT ANOTHER SWAP. Four board entries now read 0.936 / 0.937 / 0.937 / 0.932, and
    # three of them were seat EXCHANGES at seven members (on identity, on offline quality, and on
    # regularisation type). All three landed inside one tick, so the seat is not the variable. Member
    # COUNT at a fixed all-data dose is the one structural axis never moved, and until now it could not
    # be: reading it upward needs a seventh dark checkpoint that does not exist. Reading it DOWNWARD
    # needs no checkpoint at all — it needs a kernel that names six of the members already mounted.
    # This is the cheapest unasked question on the board, and it costs a slot and no card time.
    #
    # WHY `peak8` DOES NOT ALREADY ANSWER IT. It scored 0.932 with eight members and was read by some
    # as evidence against count. It is not: all eight of its seats are fold-0, so its all-data dose is
    # 0/8 against `allfit`'s 6/7. Count and dose moved together, and −0.005 is the size the dose gap
    # already had. This ladder holds dose fixed at six dark seats precisely so count moves alone.
    #
    # PRE-REGISTRATION, written before the number, against `allfit`'s measured 0.937 at seven.
    #
    #   COUNT IS LOAD-BEARING DOWNWARD — ≤ 0.933. Removing one fold-0 member costs at least four ticks,
    #     so the fold-0 riders are contributing real diversity and the ladder should be climbed, not
    #     descended: the seventh dark checkpoint becomes the best-priced card work we have.
    #   FLAT — 0.935–0.938. Six, seven and eight members are the same entry to this board. That closes
    #     the count axis with a number instead of an extrapolation, and it says something sharper than
    #     "flat": the six dark members already carry the whole entry, and every fold-0 rider we have
    #     mounted since is decoration. If `count8` also reads flat, the axis is closed from both sides
    #     by a nested triple and member-side work should stop entirely.
    #   SIX IS BETTER — ≥ 0.940. The fold-0 riders are actively COSTING us, which would invert a rule
    #     this project has followed all week, and the first suspect is the harness: confirm the kernel
    #     named six members and that all six mounted before reading it as a result.
    #
    # The bands are ~one board tick against a board that has returned 0.936–0.937 four times. That is
    # deliberate. This entry is not trying to clear the 1.5% bar alone; it is buying the third point on
    # the only curve we can still draw, and a curve is what tells us whether to spend card time on a
    # seventh dark checkpoint or to stop building members altogether.
    chosen = (
        "final_ws_e12",
        "calib_span_ff",
        "grade_ladder_ff",
        "base_dense_ff",
        "lr4e-5_ws_ff",
        "through_dense_ff",
        "small_batch6_ff",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
