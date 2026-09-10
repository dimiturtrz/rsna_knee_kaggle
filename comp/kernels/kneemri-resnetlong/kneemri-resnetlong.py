"""A MATCHED PAIR, one variable moved: the same resnet50d recipe at 48 epochs instead of 12.

WHAT IT MEASURES, AND IT IS NOT THE CNN FAMILY. `cnnsolo_v1` already put `resnet_224_short_s2026` on
the board alone (ref 56142195, board 0.915, offline 0.8527 midpoint). This ships its SIBLING —
`resnet_224_long_s2026`, offline 0.8460 midpoint / 0.8764 ledger75, best epoch 12 of 48 — and the two
differ in the epoch budget and in nothing else: same encoder, same 224 px geometry, same 2.5D
through-plane RGB, same 20-slice max pooling, same labels, same fold. Every architecture, family and
reading term cancels in the pair. What is left is a single number: 0.0067 of offline, ONE THIRD of this
project's own noise bar, and the question of whether the board can see it.

WHY THAT QUESTION IS WORTH THE LAST SLOT OF THE DAY. `CLAUDE.md` legislates that a delta near 0.005 is
noise and must be reported as directional only. That rule was fitted on our 909-study HOLDOUT, and it
has been used all week to decide what gets card time. Nothing has ever tested whether it describes the
BOARD, whose test set is larger and is not our data. The two readings imply opposite schedules: if a
0.0067 offline gap transfers at the fitted slope it is worth ~0.010 of board, which is larger than the
gap to the leader, and small-delta arms are the cheapest lever this project has. If it does not
transfer, a whole class of knob arms is dead as a source of board movement and card time belongs
somewhere else entirely. `matchpair_v1` (2026-09-10, board 0.933) asked the same question at a 0.0003
separation and could not answer it; this is the same instrument at twenty times the separation, at a
different offline level, and inside one recipe rather than across two.

PRE-REGISTERED, before the slot, against `cnnsolo_v1`'s 0.915 and the +1.55 board_transfer slope, which
puts the point estimate at 0.905:

  * **<= 0.908** — a sub-noise offline delta DOES transfer, at roughly the fitted slope. Then the noise
    bar is a statement about our holdout's RESOLUTION and not about what is real, and arms whose
    mechanism promises half a percent become worth running and ranking.
  * **0.909-0.918** — the board cannot resolve 0.0067 either. The noise bar holds on both rulers, and
    small-delta arms are closed as a source of board movement for good.
  * **>= 0.919** — the longer budget WINS on the board while losing on the holdout. That is a defect
    finding about our own ruler at within-family scale, and the most consequential of the three: every
    member ranking this project has ever made is fold-0 holdout order.

THE MEMBER LOST ITS OWN TRAINING BRANCH AND IS SHIPPED ANYWAY. `after_resnetlong.sh` pre-registered
`long - short >= 0.015` and it came back -0.0067. That closes the epoch budget as a lever for this
recipe; it does not make the checkpoint uninformative, and the owner's standing rule is that a member
which does not beat our previous ones is still worth having available. Here it is not being ensembled
at all — it is a ruler test that happens to be made of a member.

NOT A SEATING, NOT A SELECTION, NOT A WEIGHTING, NOT A RE-LENSING. A set of one, served under the
champion's own vote, chosen before the pool was looked at because its SIBLING is already on the board.

Our weights, our data layer, our submission path. The member was fitted by `kneemri.training.train` on
the report-derived labels, selected on a holdout split by REPORT-TEXT HASH so the template reports that
recur across studies cannot straddle the divide. The same fingerprint check guards it as guards every
published member: a member run under a different reading than it was fitted under loads cleanly and
predicts from the wrong image, and nothing else in the run would say so.

Everything below the bootstrap lives in `kneemri.kaggle.kernel_run`. Only the bootstrap — finding the
mounted kit and installing its wheels — has to be here, because it runs BEFORE the package it is
finding can be imported.
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

    from kneemri.inference.budget import (  # noqa: PLC0415 - the kit is on sys.path only after boot
        KERNEL_SECONDS,
        KERNEL_SECONDS,
        Budget,
    )
    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - same
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same

    # A set of ONE, exactly as `cnnsolo` is. `LogitAverage` over a single member is the identity, named
    # rather than special-cased so the vote matches the entry this is paired against — the ONLY thing
    # that differs from that file is which of the two siblings is named here.
    chosen = ("resnet_224_long_s2026",)

    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
