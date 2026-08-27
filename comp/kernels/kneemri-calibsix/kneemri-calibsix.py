"""Every seat calibrated at once, instead of one seat given a better member.

**THE UNIFORM-LIFT ENTRY.** This is the champion `kneemri-blendseat` (0.939, six members under
`LogitAverage` with `overlap=False`) with five of its six seats replaced by their own recipe plus
exactly one flag: `--calibrated-targets`. The sixth seat, `calib_span_ff`, already carries that flag
and is reused unchanged. Nothing else moves — not the vote rule, not the serving config, not the
order, not the kit.

**WHY A UNIFORM LIFT AND NOT A BETTER MEMBER.** On 2026-08-26 seat pricing measured the pass-through
of a NEW member's solo quality into an ensemble seat at **23%**: `calib_dense_hold` is +0.0056 solo
and +0.0013 in a seat. That reprices the whole add-a-better-member lane down to a ~0.943 ceiling. The
ambiguity decomposition says why — `ensemble error = mean member error − ambiguity`, and a new member
moves both terms so they cancel. Lifting EVERY seat with the same flag moves the first term for
certain and may also move the second.

**AND IT DOES.** Measured on records already on disk, six members joined over the same 909 shared
studies. Solo, over three matched pairs: +0.0061, +0.0072, −0.0020, mean **+0.0038**. Ensembled by
the shipped rank-average rule: calibrated trio 0.8685 vs uncalibrated twins 0.8629 (**+0.0056**
against a mean solo of +0.0038), calibrated pair 0.8679 vs uncalibrated pair 0.8595 (**+0.0084**
against +0.00665). Pass-through **147% and 126%** — above one, which a pure error-term lift cannot
produce. The calibrated members disagree with each other MORE than their uncalibrated twins do.

**THE PREDICTION, arithmetic stated before the number and taken at the LOW end.** Five of six seats ×
1.0 pass-through (not 1.3) × +0.0038 per seat = +0.0038 offline; × `board_transfer` +1.72 = +0.0065
board. **0.939 → 0.946**, with 0.949 the optimistic end if 1.3 holds at five seats. Not 0.950 on its
own under either reading, and this entry does not claim it.

**THE HONEST SIZE.** Every solo delta above is inside the 0.0062 three-seed floor, the negative one
included; only the +0.0084 pair clears it. What licenses the entry is not any one pair but that the
pooled solo estimate and both ensemble readings agree in SIGN across two nested pools. A directional
bet, priced as one.

**THE OFFLINE GATE THIS ENTRY PASSED.** Four of the five new members are `--fold -1` and can never be
scored offline. `merged_blend_cf` is fold-0 like its original and is the arm's only offline read; it
is the sole evidence that the flag transfers to corners outside the two it was measured at, and the
arm's own rule was to ship nothing if it read negative.

PRE-REGISTRATION, written before the board number, against `blendseat`'s 0.939.

- **0.946 or above** — the uniform lift transfers at the size the decomposition predicted. Calibration
  stops being a member flag and becomes the default for every future seat; refit the remaining corners
  and re-serve.
- **0.942-0.945** — transfers, under the prediction. Keep it, and the honest reading is that
  pass-through at five seats is nearer 1.0 than the 1.3 the small pools showed.
- **0.940-0.941** — one tick, inside board scatter. The lift is real offline and does not survive the
  test distribution at a size worth another slot.
- **0.936-0.939** — FLAT. The uniform-lift route is refuted at the operating point and the ensemble
  axis closes for good rather than just for marginal members. First suspect is the confound the arm
  named: of the three matched pairs, both positives carry `--hold-annotations` and the negative does
  not, while none of these five seats do. The follow-up is one matched no-hold pair, not a verdict.
- **0.935 or below** — a real regression from five simultaneous single-flag changes the offline ruler
  called an improvement, which would say the flag interacts with the corners it was never measured at.

COST: one slot, and the 7.1 h of otherwise-idle overnight card the arm already spent.
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

    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - the kit is on sys.path only after boot
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same

    # The champion's order, seat for seat. Five names gain the `_cf` suffix and `calib_span_ff` does
    # not, because it already carried the flag — so the diff against `kneemri-blendseat` is five lines
    # and reads as one change applied five times.
    chosen = (
        "final_ws_cf",
        "calib_span_ff",
        "grade_ladder_cf",
        "merged_blend_cf",
        "lr4e-5_ws_cf",
        "through_dense_cf",
    )
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage())).run()


if __name__ == "__main__":
    main()
