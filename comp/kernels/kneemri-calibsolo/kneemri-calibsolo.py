"""`--calibrated-targets` priced on ONE member, as the only single-variable pair a slot can still buy.

**THE PAIR, AND IT IS ONE FLAG.** `comp/board.jsonl` holds `soloseat_v1` 0.925 (2026-08-26), which served
`final_ws_e12` ALONE through this same kernel. `comp/members.roster` describes this entry's member in one
line: *"final_ws_cf.pt — DARK. final_ws_e12's flags plus --calibrated-targets"*. Same recipe, same fit
regime (`--fold -1 --ema`), same serving, same kit. The only variable between 0.925 and whatever this
scores is the calibration flag.

**WHAT THE PAIR SEPARATES, AND WHY NOTHING OFFLINE CAN.** The roster's calibsix block states the champion's
own mechanism as a hypothesis: seat pricing measured a new member's solo quality passing into a seat at
only 23%, while `--calibrated-targets` measured a pass-through ABOVE one — 147% on a trio, 126% on a pair —
so it *"buys ensemble AMBIGUITY as well as member quality"*. Those two halves have never been separated,
because every calibration read to date was taken at roster scale, where the seating pass-through sits in
the way. Serving one member removes seating entirely and prices the member half directly. It cannot be
done offline at any cost: `final_ws_cf` is `--fold -1`, so it has seen every holdout study and no honest
offline score exists or can exist for it.

**SERVING IS THE CHAMPION'S, MINUS THE VOTE.** `overlap=False` and the same kit. `LogitAverage` over one
member is the identity, so no vote rule is passed and none is implied. Against `soloseat_v1` the only
change is the member; against `kneemri-calibsix` (0.941) the only change is COUNT, six to one.

PRE-REGISTRATION, written before the number, against the 0.925 anchor and `BOARD_RESOLVABLE = 0.0015`.

- **0.930 or above** — `--calibrated-targets` is a MEMBER-QUALITY treatment that survives without an
  ensemble to carry it. Every `knee-wbwq` warm-start arm then carries the flag by default, and a
  warm-start member becomes judgeable solo on the board rather than only in a roster.
- **0.927-0.929** — directional, and consistent with `calibsix`'s +0.002 at roster scale. No reordering.
- **0.925 or below** — calibration buys a single member NOTHING, so the 147% pass-through is ensemble
  ambiguity in full. That is a mechanism claim rather than a null: the treatment would belong to the
  ROSTER — apply it to every seat of whatever pool exists — and not to the member, and it re-opens what
  `calibsix`'s climb from 0.938 actually was.

All three branches change what the card does when it returns, which is the bar a read must clear when it
cannot buy rank. A low read costs no rank: the public leaderboard keeps the best entry, and `calibsix`
stands at 0.941.

**WHAT THIS ENTRY CANNOT DO.** It cannot move us up. Seven seatings now span 0.937-0.941 against a 0.0053
Hanley-McNeil floor, and `2026-08-27_every-no-card-axis-is-priced-and-flat.md` prices every remaining seat
permutation between -0.004 and +0.001. No zero-training play has a mechanism that could deliver the +0.009
that 0.941 to 0.950 needs. Stated here rather than discovered when the number arrives.
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

    # Named, not ranked, and the name is the whole experiment: `soloseat_v1` served `final_ws_e12`
    # through this kernel, and this is that member's `--calibrated-targets` twin. One flag apart.
    chosen = ("final_ws_cf",)
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
