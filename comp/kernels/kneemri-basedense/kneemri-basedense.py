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

WHY IT IS NOT THE SAME BET AS `crossenc`. That entry (fired the same day) varies the encoder FAMILY with a
WEAK member: `dinov3_base_lora` scores 0.8268 alone, 0.032 below the pool's best, and reads -0.0013 as a
fifth. This varies the encoder SIZE with a STRONG one: **0.8512** alone, level with the pool's own best
checkpoints (`band_dense` 0.8578, `band_dense_long` 0.8587) and 0.024 above the DINOv3 member. Both of
`crossenc`'s and `maxdiv`'s pre-registrations name this entry as the follow-up their tie outcome implies —
*a weak member of a different encoder costs nothing, so a strong one is worth the card* — and it is here
whether that tie happened or not, because the size axis is untested on the board independently of family.

ITS OWN ARM READS IT AS A LOSS, AND THAT IS THE POINT. Against its matched control `band_dense` (0.8578,
same recipe on DINOv2-small at batch 12) it is **-0.0066** — just outside the floor — at 921 s an epoch
against ~370, with the batch halved to 6 because 336 px at 20 slices does not fit otherwise. A member is
not bought for its solo score but for what it ADDS, and a checkpoint 0.0066 behind the control on its own
that comes back level once mixed is the signature of an encoder whose errors are not a re-parameterisation
of the other four's. `band_dense_long` is the precedent: its arm's premise was refuted and its checkpoint
carried the 0.930.

PRE-REGISTERED, REFUTATION NAMED FIRST. **0.929 or below refutes it** and says the size axis is worse than
free — that a member trained at half the batch on a recipe tuned for small is a liability on the board no
matter how it mixes offline, and that the halved batch (which `batch_teardown` is measuring on the card
right now) is the confound to blame before the encoder is. **0.932 or above** says encoder capacity is a
board lever this holdout cannot see at all, and the next card-time goes to DINOv2-base trained at
effective batch 12 by accumulation rather than to another small recipe. **0.930 or 0.931** is a tie at the
reporting resolution and says the size axis is free but not profitable at this dose — which still buys the
next decision, because it would make the third entry in this line (base at full effective batch, or a
base/small pair replacing two smalls) the only version of the axis left untested.

WHAT WOULD MAKE THIS UNREADABLE, NAMED BEFORE THE FACT. If `crossenc` or `maxdiv` landed at 0.932 or above
then encoder diversity is already established and this entry's marginal reading is about SIZE only; if
they landed at 0.929 or below, a tie here would be the stronger result of the two, because it would say a
STRONG different encoder holds the board where a weak one loses it.
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
