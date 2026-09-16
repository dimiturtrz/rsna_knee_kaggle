"""pricetop — the best entry this project can PRICE, shipped to give the top of the transfer curve leverage.

INSTRUMENT. It is not expected to beat the pinned `tripleseat_v2` (0.946) and must not be read as trying
to: six honest seats predict 0.940. What it buys is a reading, and the reading is the point.

WHY. `board_transfer`'s top regime — entries at or above offline 0.8695 — rests on FOUR resolved pairs and
the tool labels it DIRECTION ONLY. That slope is what converts the gap from 0.946 to the 0.955 goal into an
offline requirement (+0.0092 to +0.011), so it prices every remaining decision in the project, and it has
been quoted wrong three times (+4.67, +2.14, x1.55). The reason it is direction-only was measured today:
the entire priceable record tops out at offline 0.8751, and the band above 0.8695 holds thirteen entries
spanning 0.0056 — barely over the holdout's own 0.0053 resolution floor. Nearly every pair up there is
unresolved BY CONSTRUCTION. A slope fitted across a span the width of the floor is the exact shape of the
+4.67 error.

WHAT THIS ENTRY DOES ABOUT IT. Offline 0.8785, which is +0.0034 above the record's maximum. It widens the
band to 0.0090 and resolves against the eight band entries at or below 0.8732, taking its resolved pairs
from four to about twelve.

WHY EVERY SEAT HOLDS OUT A SPLIT. Forty-two board entries at 0.940 and above carry `offline: null`, because
their seats are `--fold -1` and unpriceable by construction (`comp/champion.json:54`). The champion is one
of them. This entry is deliberately built from members that DID hold out a split, which is what makes it an
anchor rather than another unpriceable point at the top.

PRE-REGISTRATION, against `board_transfer`'s own interval read before the slot was spent:

- **0.943+** — compression at the top is HIGHER than the tool reads, the offline requirement for 0.955
  SHRINKS, and every axis closed for delivering under +0.005 offline is re-priced. The outcome worth the slot.
- **0.940–0.941** — the ~+0.8 top slope is confirmed and tightens on triple the pairs; the +0.0092 to
  +0.011 requirement stands.
- **0.938 or below** — the curve has gone flatter still at the top, 0.955 costs MORE offline than stated,
  and the member problem is worse than the record says.

There is no null branch, which is what makes this an instrument rather than a hope. See knee-hr499.
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

    # Six seats, all of them holding out a split, priced together at offline 0.8785 by
    # `python -m kneemri.kaggle.priceable_blend --members …`. Three sit on `kneemri-members` and three on
    # `kneemri-members-2`, so BOTH shelves are mounted; a member on an unmounted shelf is dropped silently
    # and the entry scores anyway under a name for a seating it never had (`farseat_v1`, knee-nbk3).
    #
    # A seventh seat was available and is NOT here. `priceable_blend`'s greedy search offered
    # `labelwise_dense_drop40_long36_s2026` at +0.0007, and that member is on no shelf and is declined in
    # writing at `comp/members.declines:1240` — the 36-epoch horizon control that lost to its own 24-epoch
    # twin by -0.0018 against a pre-registered +0.003 bar. A decline is re-opened by naming a mechanism, not
    # by a tool re-quoting the number it was declined on (knee-k8p9i).
    chosen = (
        "labelwise_depth24_drop40_long_s2026",
        "labelwise_drop_drop40_long_s2026",
        "oof_teacher_s2026",
        "margin_seat_s2026",
        "warm_calib_lowlr_drop40_long_s2026",
        "blendhold_s2026",
    )
    # Serving is the champion's, unchanged: no overlap TTA (five members measured it flat on whole-stack
    # members), plain logit averaging, no weighting. The entry varies the MEMBER SET and nothing else, so a
    # board number it returns is about the seats rather than about a serving choice made at the same time.
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage())).run()


if __name__ == "__main__":
    main()
