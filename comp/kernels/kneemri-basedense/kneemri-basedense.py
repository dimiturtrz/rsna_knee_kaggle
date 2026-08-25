"""The board maximum plus one member from a LARGER encoder — the size axis no entry has ever varied.

Nine entries have scored on this board and every member any of them named is DINOv2-**small**. This is
`longseat`'s four (0.930, the joint board maximum) with `base_dense_s2026` added as a FIFTH: DINOv2-base,
768 wide against small's 384, twelve layers against twelve at three times the width, on the same settled
band recipe with nothing else moved. Same RankAverage vote, same `overlap=False`, same four incumbents, so
the entry is a ONE-MEMBER delta off a known 0.930 and its difference is attributable to that member alone.

THE OFFLINE READING, STATED FIRST AND WITHOUT ROUNDING. Rank-averaged over the 909-study fold-0 holdout
the five read **0.8690** against the four's **0.8691** — one ten-thousandth, a sixtieth of the 0.0062
three-seed noise floor. Against the pinned champion `greedy4_v1` (**0.8679**) the five are **+0.0011**.
This is the first member this project has mounted whose add is not negative. Its best swap is -0.0006.

WHY IT IS NOT THE SAME BET AS `crossenc`, WHICH SCORED FOUR HOURS AGO AND LOST. `crossenc` (the same four
plus `dinov3_base_lora`) landed **0.926** and `maxdiv` (that five plus `crop80_tissue`) landed **0.925**,
against `longseat`'s 0.930. Both pre-registered ≤0.929 clauses fired. The board ordering matched the
offline ordering exactly — 0.8691 / 0.8678 / 0.8668 to 0.930 / 0.926 / 0.925 — so this holdout RANKS
correctly and errs only in magnitude, understating what a weak member COSTS by about 3x. That is the
opposite of the bias those two entries were fired on, and it is stated here because it is the reason this
entry is not the same bet.

Under that corrected reading the prediction for this entry is a TIE, and it is fired anyway for one
reason: it is the only candidate that is not weak. `dinov3_base_lora` scores 0.8268 alone, 0.032 below the
pool's best, and greedy forward selection over all eighteen mounted members takes it SIXTEENTH;
`crop80_tissue` (0.8345) fifteenth. `base_dense` scores **0.8512** and is taken **seventh**, inside the
flat top where the curve reads 0.8691 / 0.8692 / 0.8692 / 0.8692 at four through seven. Both refuted
entries named exactly this as their surviving question — *a weak member of a different encoder is priced,
so is a strong one worth the card* — and tonight's monotonicity is what makes it answerable, because it
says the board reads member quality faithfully enough to reward it.

ITS OWN ARM READS IT AS A LOSS, AND THAT IS THE POINT. Against its matched control `band_dense` (0.8578,
same recipe on DINOv2-small at batch 12) it is **-0.0066** — just outside the floor — at 921 s an epoch
against ~370, with the batch halved to 6 because 336 px at 20 slices does not fit otherwise. A member is
not bought for its solo score but for what it ADDS, and a checkpoint 0.0066 behind the control on its own
that comes back level once mixed is the signature of an encoder whose errors are not a re-parameterisation
of the other four's. `band_dense_long` is the precedent: its arm's premise was refuted and its checkpoint
carried the 0.930.

PRE-REGISTERED, WITH THE PREDICTION STATED HONESTLY AS A TIE. **0.930 or 0.931 is the expected outcome**
and it is the one that buys the most: it would say a STRONG member of a different encoder holds the board
where a weak one drops it four ticks, separating MEMBER QUALITY from MEMBER COUNT as the thing the board
prices, and it would make DINOv2-base at effective batch 12 the next thing worth card-time. **0.932 or
above** says encoder capacity is a board lever this holdout cannot see at all and the roster has been
selected against the wrong objective. **0.929 or below refutes the size axis at this operating point**,
and — read against `crossenc`'s 0.926 — would say adding ANY fifth member to these four is priced
regardless of its quality, which retires adding members entirely and leaves only swaps and new recipes.

THE CONFOUND, NAMED BEFORE THE FACT. This member trained at `--batch 6`, not the recipe's 12, because 336
px at 20 slices on a 768-wide encoder does not fit in 32 GB. `batch_teardown` is on the card right now
measuring `band_dense`'s exact recipe at batch 6 and at batch 6 with `--accumulate 2`. If it shows batch 6
costs materially on its own, a 0.929-or-below reading here is a batch verdict, not an encoder verdict, and
the axis must be re-run at effective 12 before it is called closed.
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
        "band_dense_s2026",
        "calib_span_s2026",
        "depth_regions_narrow_s2026",
        "band_dense_long_s2026",
        "base_dense_s2026",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
