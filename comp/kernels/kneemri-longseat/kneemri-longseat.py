"""The champion four with `band_span` replaced by a checkpoint whose own arm was refuted.

`kneemri-greedy4` scored **0.930** and is the entry pinned in `comp/champion.json`. This is that four
with ONE member swapped: `band_span_s2026` (0.8565 solo) out, `band_dense_long_s2026` (0.8587) in.

WHERE THE CANDIDATE CAME FROM, STATED AGAINST ITSELF. `band_dense_long` is `band_dense`'s recipe run for
24 epochs instead of 12, and the arm asked one question: is the schedule a bound on the band members?
The answer is no, unambiguously. It peaked at epoch 6 (0.8587), never beat that in eighteen more epochs,
and slid to 0.8405 by epoch 23 while the training loss halved — a 0.0182 decline, 2.9x our 0.0062 noise
floor, which is memorisation rather than a schedule wanting room. Against its matched control
`band_dense` (0.8578) it is +0.0009, a seventh of the floor. As a MEMBER RESULT this arm is a refutation.

WHY IT IS IN A KERNEL ANYWAY. Priced in the seat rather than alone, it is the best move available to the
champion: rank-averaged over the 909-study fold-0 holdout this four reads **0.8691** against greedy4's
**0.8679** for the roster, +0.0012, taking `band_span`'s seat. At the moment it landed, the best swap the entire mounted
pool could offer was +0.0002 (`attend_dense`), so this is six times the alternative. The precedent is
`calib_span`, which this ledger wrote up as a failure against its own control hours before it carried the
0.929 entry; the shape here is sharper, because it is not merely the member that lost — the arm's whole
PREMISE was refuted, and the checkpoint it dropped out is still the largest improvement on the card.

WHAT THE SWAP DOES NOT CHANGE. Same count (four, per the FOUR MEMBERS principle), same vote
(RankAverage), same `overlap=False`. `band_span` reads 0.10-0.90 at 12 slices and `band_dense_long` reads
0.10-0.90 at 20, so the pixel groups differ and the decode count is unchanged from greedy4's.

PRE-REGISTERED. **0.930 or 0.931** — at or above the champion. The offline edge is +0.0012, a fifth of
the floor, and the ledger's own reading of five prior entries is that this ruler ORDERS the field at 4/5
over a 0.003 spread and that firing its maximum is the supported move (greedy4 itself fired on +0.0009
and won a tick). **Below 0.930 refutes that principle at the first opportunity it has had to fail**, and
that is the more useful outcome: it would say the ordering held for the five entries that built it and
does not extend to a candidate whose own matched control is inside the floor — i.e. that seat value needs
a member result behind it, which is exactly what `calib_span` appeared to disprove.

A THIRD OUTCOME WORTH NAMING. If it lands at 0.930 exactly, that is the reporting resolution and not
evidence either way; the entry would be a tie held at lower risk, and the next move stays with the label
axis (knee-mxj) rather than another swap in this pool.
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

    chosen = ("band_dense_s2026", "calib_span_s2026", "depth_regions_narrow_s2026", "band_dense_long_s2026")
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
