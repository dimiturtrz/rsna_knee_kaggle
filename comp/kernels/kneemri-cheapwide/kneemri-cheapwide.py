"""EVERY cheap-corner member at once — 224 px, 12 slices, fp16, logit vote, TEN seats.

**THIS IS AN EFFICIENCY-TRACK INSTRUMENT AND IT MOVES ONE THING: THE SEAT COUNT AT A FIXED DECODE.**
`cheapdecode_v1` served three members of this exact group on 2026-09-05 and read **0.925**, which was
0.008 BELOW its pre-registered 0.933-0.939 band. That number disqualified the cheap corner on ACCURACY,
not on speed: `efficiency_read --live` puts the top-20 efficiency ranks between 0.927 and 0.952 with a
median of 0.940, so 0.925 is under the floor a cheap build has to clear. This entry asks whether the
shortfall was the corner or the seat count.

**THE MECHANISM, AND IT IS A RUNTIME CLAIM.** Members that share a decode corner share the decode.
`comp/logs/cheapdecode/kneemri-cheapdecode.log` reads `studies 1-3: decode 5s at [224], elapsed 5s` —
the decode IS the elapsed time at three seats; the ten forward passes this entry adds are the only
new work, and they run on a card the fleet pass measured at 0% while pydicom worked. So 3.7x the seats
should cost far less than 3.7x the wall clock. Nothing in this project has ever measured that, and it
decides the whole efficiency lane: if seats are near-free at a fixed decode, the lane is "hold ~0.940
accuracy with many cheap members" rather than "give up accuracy for speed".

**THE SEATS ARE CHOSEN BY RULE, NOT BY RANK.** Every roster member whose recipe reads 224 px / 12 slices
/ band 0.2-0.8, and nothing else. No score enters the tuple, so this is not a selection or a reseating —
the same rule run tomorrow returns the same set plus whatever has since been trained at that corner.
Two members are missing for a stated reason, and Kaggle told us which: `blendemalabel224_s2026`
(0.8591, the group's strongest) and `blendema224_s2026` (0.8567) both finished tonight, after the
members package was last built, so `MemberPackage.named` refuses them. Version 1 of this kernel died on
exactly that refusal, which is the behaviour we want -- `KernelRun` drops a member it cannot MOUNT, but
a member the package does not HOLD raises.

    blendhold_s2026 0.8575 · ema224_s2026 0.8544 · blendlabels_s2026 0.8537
    labelwise224_s2026 0.8517 · sided224_s2026 0.8512 · warm224_long_s2026 0.8512
    base224_s2026 0.8509 · through224_s2026 0.8505 · mixup224_strong_s2026 0.8516 · native224_s2026 0.8502

**PRE-REGISTERED, BEFORE THE NUMBER.** Control is `cheapdecode_v1` = 0.925, three seats, same corner,
same vote, same precision:

- **0.933-0.940** — the expectation. Ten seats at one decode recover the cheap corner onto the top-20
  efficiency floor. The lane opens: hold accuracy, cut runtime, and the entry is a live efficiency row.
- **0.926-0.933** — seats help but not enough. The cheap corner is short of the 0.927 floor on merit and
  the efficiency lane needs a better corner, not more seats.
- **within 0.002 of 0.925** — seat count buys NOTHING at a fixed decode corner. That would be the eighth
  board flat on the member axis and would close it at the cheap corner too, which is worth knowing: it
  would mean the 0.019 `cheapdecode_v1` gave up is the corner's price and not the group's.
- **below 0.920** — read `run_check` on the log BEFORE reading the score. `KernelRun` drops a member it
  cannot mount and keeps going; ten seats is the most this project has shipped at once and it is the
  most exposed to a silent drop.

**THE RUNTIME NUMBER IS THE POINT AND IT IS READ FROM THE LOG, NOT THE BOARD.** Whatever the score, the
deliverable is the elapsed-per-study line beside `cheapdecode_v1`'s. A near-flat runtime at 3.7x the
seats is the finding; the accuracy is what says whether the lane is worth walking.

**WHAT THIS ENTRY IS NOT.** Not a candidate for the accuracy board: the champion is `raptorweight_v1` at
0.944 and nothing here approaches it.
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

    import torch  # noqa: PLC0415 - the kit is on sys.path only after boot

    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - same
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same

    # EVERY roster member at the cheap decode corner - 224 px, 12 slices, band 0.2-0.8 - and nothing
    # else, minus blendemalabel224_s2026, which finished after the members package was last built and
    # is therefore not mountable today. Chosen by RULE, not by rank: no score enters this tuple, so the entry is not a selection.
    # The question is runtime. Members at one decode corner share the decode, which is what a kernel
    # here actually spends (the cheapdecode log reads decode 5s of 5s elapsed for three studies), so
    # four times the members should cost far less than four times the wall clock.
    chosen = (
        "blendhold_s2026",
        "ema224_s2026",
        "blendlabels_s2026",
        "labelwise224_s2026",
        "sided224_s2026",
        "warm224_long_s2026",
        "base224_s2026",
        "through224_s2026",
        "mixup224_strong_s2026",
        "native224_s2026",
    )
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), precision=torch.float16)).run()


if __name__ == "__main__":
    main()
