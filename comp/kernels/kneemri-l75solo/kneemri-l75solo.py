"""What is one held-out member worth on the board, with no selection step anywhere in it?

WHY THIS ENTRY EXISTS. Every all-ours row above the top band's floor is `scope: excluded` — greedily
selected ON the 909-study holdout, in-sample by construction, which is the mechanism that voided
`raptor_solo`. The band we ship in was fitted from rows that should never have been in it. A SOLO has
no selection step at all: one checkpoint, one ledger row taken on studies it never trained on, one
board number. It is the first fully clean offline<->board pair this record would own.

THE MEMBER. `labelwise_depth24_drop40_long_s2026` is the best solo the project has ever measured
honestly — **ledger75 0.9053**, 909 rows, peak epoch 18 — and it has been seated in fourteen board
rows, every one of them an ensemble. It has never been shipped alone.

THE SECOND READING, FOR FREE. `labelwise_depth24_drop40_long_ff` is this same recipe and seed refit at
`--fold -1 --ema`, and it scored **0.944 solo** (ref 56296698). The two differ in exactly one thing:
whether 909 studies were held out. So this entry also prices the FULL-FIT PREMIUM directly, a number
this project has never had and which decides whether the final deliverable should hold anything out.

THE PREDICTION, fitted on SOLO ROWS ONLY. The whole-record and top-band slopes are fitted over entries
of every seat count, and seat count is the board's strongest covariate, so neither of them is the right
line for a set of one. `comp/board.jsonl` holds seven single-member rows carrying an `offline75` on a
clean scope, and they make their own line:

    ortho_dense_s2026 0.8748 -> 0.907   resnet_224_long_s2026 0.8764 -> 0.907
    resnet_224_short_s2026 0.8842 -> 0.915   labels_v4hybrid_s2026 0.8874 -> 0.922
    calib_span_s2026 0.8895 -> 0.921   calib_dense_hold_s2026 0.8949 -> 0.928
    stackfit_s2026 0.8962 -> 0.933

OLS over those seven: `board = -0.1217 + 1.174 * offline75`, **residual sd 0.0019**, n=7. At offline75
0.9053 this member predicts **0.9414**, 2 sd band **[0.9377, 0.9452]**. The board reports at 0.001.

That fit also already bounds the thing this entry is for. `_ff` scored 0.944, which sits at the TOP EDGE
of its held-out twin's band: a full-fit premium of about **+0.0026, some 1.4 sd**. Suggestive, and not
resolved, because 1.4 sd is not a result and because the prediction has never been checked against an
actual solo shipment of this member.

PRE-REGISTERED, against 0.9414 and that 0.0019:

  * **board <= 0.937** — below 2 sd. Either the +1.174 solo transfer breaks at the top of its own range,
    where it is extrapolating past `stackfit_s2026`'s 0.8962, or a held-out member transfers worse than
    the line says. The transfer is the first suspect, since every anchor sits below this point.
  * **board 0.938-0.945** — ON the line, and the modal branch. The full-fit premium is then whatever
    `0.944 - board` comes to, read with 0.0019 in hand rather than assumed. Anything inside +/-0.004 is
    inside 2 sd and means holding 909 studies out is FREE — keep the ledger row, it costs nothing.
  * **board >= 0.946** — the held-out twin matches or beats both the full fit and the eight-seat
    champion, alone. That makes `--ema` or the absent early stop the suspect in `_ff`, and makes this
    checkpoint the deliverable rather than a member.

All three also give the solo transfer its first out-of-sample test.

THIS IS AN INSTRUMENT AND IS SUBMITTED AS ONE. A single member is not offered as a way past a 0.946
ensemble — the modal branch sits three thousandths under the pin. It costs no GPU: the checkpoint has
been on the shelf since 2026-09-15.

Our weights, our data layer, our submission path. Everything below the bootstrap lives in
`kneemri.kaggle.kernel_run`.
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

    from kneemri.inference.budget import (  # noqa: PLC0415 - the kit is on sys.path only after boot
        KERNEL_SECONDS,
        Budget,
    )
    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - same
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same

    # A set of ONE. `LogitAverage` over a single member is the identity, named rather than
    # special-cased so the vote stays the champion's convention here too. One name is also what
    # makes the reading clean: a set of two would need a selection step, and the holdout is the
    # only thing available to select on.
    chosen = ("labelwise_depth24_drop40_long_s2026",)

    # The whole session, not a share of it: no foreign producer runs first here, so there is nothing to
    # subtract. Spelled explicitly because `oofseat` subtracts and this file is read beside it.
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
