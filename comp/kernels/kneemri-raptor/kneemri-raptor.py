"""The raptor member served SOLO, on our own decode and our own forward pass.

**WHAT IS BEING PRICED.** A published third-party member — CoAtNet at 384 over a five-slot, 64-deep
volume with 42 evaluation windows and a per-finding attention pool — against `soloseat_v1` 0.925, the
best single member this project has put on the board. Their own page reports 0.914 for this arm alone
and 0.915 for a three-arm rank-mean, so their ensembling bought ~0.001; this entry buys the MEMBER, not
their blend, which is why one arm is mounted and not three.

**NO FOREIGN CODE RUNS HERE.** `external/raptor` is two notebook exports with no LICENSE file and could
not be vendored into a kit at all. `kneemri.evaluation.raptor_feed` and `raptor_model` reimplement the
decode and the forward pass from their published prose spec, and both are held to EQUALITY rather than
agreement: 200 studies decode byte-identically to theirs, and 25 studies score identically probability
for probability on CPU/fp32. Their weights stay what they always were — a public Kaggle dataset,
CC0-1.0, mounted.

**WHY IT CANNOT BE PRICED OFFLINE.** Their arm is `--fold -1` by construction: it was fitted on the whole
public corpus, so every study of our fold-0 holdout is inside its training set and no honest offline
number for it exists. The board is the only ruler that can see it, which is what a slot is for.

PRE-REGISTRATION, written before the number, against `soloseat_v1` 0.925 and `BOARD_RESOLVABLE = 0.0015`.

- **0.945 or above** — the member is stronger than anything we have trained, and their reported 0.914 was
  a floor set by their own decode rather than by the weights. Seating it beside our six becomes the next
  entry, not a later one.
- **0.930-0.944** — a real member and a real diversity candidate: it beats our best solo while landing
  under the 0.941 roster, which is exactly the profile a seventh seat is bought for.
- **0.915-0.929** — it reproduces their published solo, no more. The port is confirmed end to end on the
  hidden set, and the member is worth a seat only if a seating read says the diversity pays.
- **Below 0.914** — OUR chain, not their member: the same weights scored 0.914 in their hands, so a lower
  number here is a defect in the mount, the slot fill or the window count, and it is a bug report rather
  than a result.

**WHAT THIS ENTRY CANNOT DO.** It cannot move us up on its own. Our board best is 0.941 and the leader is
0.952; a member landing anywhere in the bands above is a SEAT candidate, and the seating is a later entry.
Stated here rather than discovered when the number arrives.
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

    from core.mounts import Mounts  # noqa: PLC0415 - the kit is on sys.path only after boot
    from kneemri.device import VisibleCards  # noqa: PLC0415
    from kneemri.evaluation.raptor_model import RaptorMember  # noqa: PLC0415
    from kneemri.kaggle.raptor_run import RaptorEntry  # noqa: PLC0415

    mounts = Mounts()
    device = str(VisibleCards().device())
    member = RaptorMember.loaded(RaptorEntry.weights(mounts=mounts), device)
    entry = RaptorEntry(mounts.one("sample_submission.csv", "competition data"))
    entry.run(member, device)


if __name__ == "__main__":
    main()
