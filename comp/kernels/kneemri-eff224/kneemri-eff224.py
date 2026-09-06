"""The champion's six members, READ AT 224 px — final B, the efficiency selection.

**THIS ENTRY IS PRICED BY THE EFFICIENCY FORMULA, NOT BY THE ACCURACY BOARD.** The competition scores
`AUC/(0.5 - 0.95) + Runtime/32400`, minimised, so one AUC point buys 72,000 s and one second per study
over the 1,322 test studies is 0.0184 AUC-equivalent. Measured offline on the six fold-0 twins of the
champion's dark members (the champion's own five are `--fold -1` and cannot be scored here):

    native 0.8706 · 280 px 0.8687 (-0.0019) · 224 px 0.8637 (-0.0069)

and the forced size costs -1.08 s/study, so 224 nets -0.029 score units against 280's -0.017. The
accuracy final stays `raptorweight_v1` at 0.944; this entry gives up nothing that entry holds.

**WHAT A FORCED READ SIZE ACTUALLY BUYS, AND IT IS LESS THAN THE SLOPE SAYS.** `Member.resized`
overrides `train_size`, never `img`, and `Ensemble.reads` groups members WITHOUT resolution, so the
corpus is still decoded once at the shared cache size — `SlotReader.cropped` stops before the resize
and one read serves every size. The saving is FORWARD time only: 2.37 s of the ~4.11 s at six seats is
decode and this entry cannot touch it. Six seats go from ~1.74 s to ~0.66 s of forward; the entry is
~3.03 s/study, a 26% cut, not the 42% the naive slope implied.

**SERVED IN fp16.** Measured 2026-09-05: fp16 serving ties 17% of the corpus against fp32 and the vote
does not care. It is free speed on a T4 and it is the owner's standing serving directive.

**NO RAPTOR SEAT.** The champion blends a seventh foreign pass at 1/3 for +0.002 board. A whole extra
model's forward is the most expensive thing this entry could carry per point of AUC, so final B drops it.

**PRE-REGISTERED, BEFORE THE NUMBER.** Control is the champion's 0.944 on the accuracy board, and the
transfer constant `board_transfer` is +1.55 on our offline deltas.

- **0.930-0.938** — the expectation: the offline -0.0069 through the transfer, minus the raptor seat's
  +0.002. The efficiency lane is live and the runtime term is the thing to spend on next.
- **above 0.938** — 224 costs less on Kaggle's studies than on our 909; the resolution axis has more room.
- **below 0.925** — read `run_check` on the log BEFORE the score: six seats is where a silent member drop
  has hidden before (`farseat_v1` shipped six of seven and read its control's exact number for five days).

**THE DELIVERABLE IS THE WALL CLOCK.** Every runtime number this project owns is a smoke-harness slope.
This entry's log gives the first real kernel s/study at a known seat count and a known resolution, and
that number replaces the slope everywhere it is quoted.
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

    # The champion's own six, unchanged and in its order, so the ONLY variables against `raptorweight_v1`
    # are the read size, the serving precision and the dropped raptor seat - each of them priced above.
    chosen = (
        "final_ws_cf",
        "calib_span_ff",
        "grade_ladder_cf",
        "merged_blend_cf",
        "lr4e-5_ws_cf",
        "through_dense_cf",
    )
    serving = Serving(vote=LogitAverage(), precision=torch.float16, read_size=224)
    KernelRun(names=chosen, overlap=False, serving=serving).run()


if __name__ == "__main__":
    main()
