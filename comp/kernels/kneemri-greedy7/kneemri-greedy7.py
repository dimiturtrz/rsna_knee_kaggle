"""The greedy rule, re-run on a pool that grew by five members since it last chose a roster.

WHAT IT CHANGES. Four of the champion's seven seats. `band_dense`, `grade_ladder`, `base_dense` and
`lr4e-5_ws` leave; `band_dense_long`, `through_dense`, `depth_regions_narrow` and `calibrated_targets`
take their places. Count (7), kit (`cf6a99f`) and vote are held fixed. This is a WHOLESALE exchange, not
a one-for-one swap — which is exactly why it is not the move the ledger banned.

**WHY THIS ROSTER.** `record_mix --greedy` over the 28 members currently packaged in `kneemri-members`
peaks at **0.8712 at seven** on the 909-study fold-0 holdout, against the champion seven's **0.8694**.
+0.0018 is the first offline gap in this entire search above the seat ruler's +0.0015 resolution.

**THIS IS THE STANDING RULE, NOT A NEW BET.** `greedy4` ran this exact search over the then-mounted
sixteen, was pre-registered to land *at or below* 0.928 — and came back **0.930 as a new champion**,
refuting the one-swap rule it was built to test. This ledger's own conclusion was to make every future
entry greedy-optimal by rule. The pool has since gained five members, so the rule is re-run, not re-argued.

**IT IS NOT THE BANNED MOVE.** `accumseat` retired one-for-one member swaps justified by a RULER
(residual distance, binary seat value, ordinal concordance — all three dead). Nothing here is chosen by a
ruler: four seats move on the joint ensemble metric itself, which is the quantity the board scores.

PRE-REGISTRATION, written before the number.

- 0.936 or above — greedy transfers at full roster width. Every future entry is greedy-optimal over the
  mounted pool by rule, and the remaining budget goes to widening the POOL rather than searching it.
- 0.934-0.935 — directional. Below the house bar, above one board increment; greedy is kept as the
  default constructor but no axis is closed on it.
- 0.933 — a third tie. The board is decisively flat to WHOLESALE exchange within this pool, not merely to
  one-for-one swaps. Roster search is retired entirely and the budget belongs to new axes only.
- 0.932 or below — selection bias, and the mechanism is named in advance: this roster was chosen by
  maximising the very number it is scored on, over 28 candidates on 909 studies. An offline +0.0018 that
  turns into a board loss is overfit to the holdout, and it prices every future offline greedy number.

+0.0018 offline at `knee-7d1`'s median +4.67 board points per holdout point prices at about +0.008. I do
not expect the median: the compression is a median over a spread and this edge is a quarter of the 0.0062
noise floor. Silent on data volume — `final_fit` answers that axis separately.
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
        "small_batch6_s2026",
        "band_dense_long_s2026",
        "calib_span_s2026",
        "through_dense_s2026",
        "aprior_dense_s2026",
        "depth_regions_narrow_s2026",
        "calibrated_targets_s2026",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
