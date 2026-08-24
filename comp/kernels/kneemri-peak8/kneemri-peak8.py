"""The 8-member greedy holdout peak, on the board for the first time.

**THE HOLE THIS FILLS.** The peak set — `small_batch6`, `margin_seat`, `calib_span`, `through_dense`,
`band_dense_long`, `sided_pool`, `aprior_dense`, `crop80_span` — has scored 0.8731 offline since it was
found, and until 2026-08-24 it was NOT SHIPPABLE: `margin_seat` was rostered but never pushed,
`sided_pool` appeared in no roster line at all, and `crop80_span` only inside a comment. Every gate was
green — `champion_check`, `mount_check`, `kit_check` — because each one guards the entry it was written
for, and none of them asked whether our BEST measured ensemble could be built at all. Our 0.937 board
entry is a seven-member set that excludes the two members greedy valued most on distance.

**WHY IT IS WORTH A SLOT.** `margin_seat` is solo 0.8525 — below several seated members — and greedy
took it SECOND of 65, on distance alone. `crop80_span` is the weakest member of the peak at solo 0.8466
and sits at nearest-neighbour agreement 0.8717, second most distant of the eight. Neither has ever been
on a board entry. The holdout prices diversity ~1.9-2.4x more cheaply than the board does, so the two
members this entry adds are exactly the kind the offline ruler under-reads.

PRE-REGISTRATION, written before the number, against `basedense_v1`'s 0.937.

- **0.940 or above** — the board reads the distance-bought seats the way the transfer fit predicts, the
  holdout peak IS the board peak, and member selection by `MemberDistance` is vindicated end to end.
- **0.938-0.939** — a real but sub-transfer gain; the greedy peak transfers, the multiplier does not.
- **0.935-0.937** — flat. The 0.0026 of offline headroom over the champion does not survive the board,
  and the ninth-seat saturation result extends to the board: this pool is done, full stop.
- **0.934 or below** — the informative loss. The offline greedy peak is OVERFIT TO THE 909-STUDY
  HOLDOUT, which would make `seat_value` a ruler that ranks correctly (0 inversions of 33 resolved
  pairs) while its ARGMAX is still a holdout artefact — and would move the whole member-selection
  account onto the board, where each read costs a slot.
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
        "small_batch6_s2026",
        "margin_seat_s2026",
        "calib_span_s2026",
        "through_dense_s2026",
        "band_dense_long_s2026",
        "sided_pool_s2026",
        "aprior_dense_s2026",
        "crop80_span_s2026",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
