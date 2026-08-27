"""Eleven seats, two supervision regimes — the first ensemble built on a MEASURED difference in error.

**WHAT MOVES.** The UNION of the last two champions' rosters: `kneemri-calibsix` (0.941, six
`--calibrated-targets` members under `LogitAverage` with `overlap=False`) and the seating it replaced,
`kneemri-blendseat` (0.939). `calib_span_ff` sits in both, so the union is ELEVEN members, not twelve.
Nothing else moves: same vote rule, same serving config, same kit, same order within each half. The
single variable is that both supervision regimes are in the vote at once instead of one replacing the
other.

**WHY THIS IS NOT THE COUNT AXIS AGAIN, WHICH IS CLOSED.** `dark6` (6) and `dark7` (7) both read 0.937,
`peak8` (8) read 0.932, and the ledger closed member count on that evidence. Every one of those
entries added a member that differed by RECIPE — a rate, a resolution, an encoder family — and the
project has never had a way to say whether two members actually make different MISTAKES. It does now.

`kneemri.evaluation.noise_binding` (2026-08-27, zero slots) measures how strongly a member is bound to
the label reader's confident false positives: the reader over-calls 108 of the 684 annotated cells, and
the statistic asks where those disputed cells fall between the two poles where reader and radiologist
AGREE. Over all four held checkpoints:

    band_span_hold   +0.683 [0.600, 0.762]        calib_span_hold   +0.538 [0.467, 0.608]
    band_dense_hold  +0.700 [0.627, 0.773]        calib_dense_hold  +0.550 [0.470, 0.630]

`--calibrated-targets` cuts the binding by **-0.145** at the span corner and **-0.150** at the dense
corner — the same size at two independent corners, roughly 2x the CI half-width. That is a measured,
reproduced statement that the two halves of this entry are wrong about DIFFERENT cells: the `_ff` half
inherits the reader's invented positives, the `_cf` half is a fifth of the way out of them. No other
pair of members this project owns has that reading, and error diversity — not recipe distance — is the
term the ambiguity decomposition actually says an ensemble is paid for.

**WHAT ARGUES AGAINST IT, STATED FIRST.** The `_cf` half is the better half: it took the board from
0.939 to 0.941 by itself. Putting the uncalibrated twins back into the vote re-admits members that ARE
more bound to the reader's errors, and a plain average over eleven gives them nearly half the weight. If error
diversity is worth less than mean member quality here, this entry gives back the champion's tick. It is
a bet on the ambiguity term, and the mean-error term points the other way.

PRE-REGISTRATION, written before the board number, against `calibsix` 0.941.

- **0.944 or above** — error-structure diversity is a live lever and the count ceiling was a statement
  about TWIN-ness, not about count. The seat search reopens on one rule: pair members by measured
  binding, never by recipe. Next slots go to widening the spread rather than raising the floor.
- **0.942-0.943** — directional, in the right direction, under the noise floor. Keep the eleven and
  re-test; do not rewrite the count principle on it.
- **0.939-0.941** — FLAT. The extra six neither help nor hurt, which says the vote is already averaging
  over the same errors and the -0.15 binding gap is not enough separation to matter at serving. The
  count principle survives and the honest next move is a LARGER binding gap, not more members.
- **0.938 or below** — mean member quality dominates the ambiguity term at this separation. Uniform
  calibration is the right call and mixing regimes DILUTES; the champion stands and this axis closes.

**WHAT IT CANNOT SAY.** noise_binding reads 58 studies and 108 over-called cells. A ratio on that
denominator is DIRECTIONAL — it separates 1 from 0 and cannot resolve 0.6 from 0.7 — so -0.15 is
"roughly a fifth", never a constant, and this entry is not a calibrated prediction of anything. It also
cannot attribute a move to diversity alone: eleven members at two regimes changes the COUNT as well,
and only the 0.944+ branch is large enough that the closed count axis cannot explain it.

COST: one slot, zero card hours — every one of the eleven checkpoints is already on the roster and
already mounted, so this is a kernel-only change.
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

    # The UNION of the two champions' rosters: calibsix's six, then blendseat's six in their own order.
    # `calib_span_ff` sits in both and is named once, so the union is eleven names rather than twelve.
    # The diff against `kneemri-calibsix` is five added lines and nothing else, which is what makes the
    # two entries a matched pair on the one axis this entry moves.
    chosen = (
        "final_ws_cf",
        "calib_span_ff",
        "grade_ladder_cf",
        "merged_blend_cf",
        "lr4e-5_ws_cf",
        "through_dense_cf",
        "final_ws_e12",
        "grade_ladder_ff",
        "merged_blend_s2026",
        "lr4e-5_ws_ff",
        "through_dense_ff",
    )
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage())).run()


if __name__ == "__main__":
    main()
