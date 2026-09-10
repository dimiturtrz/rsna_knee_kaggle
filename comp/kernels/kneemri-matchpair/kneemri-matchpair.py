"""A DIRECT measurement of the board ruler's resolution floor, using a matched-offline pair.

WHAT IS BEING MEASURED, AND WHY IT IS NOT A MEMBER ATTEMPT. Fitted on `comp/board.jsonl` over entries
from 2026-08-20 with `scope: excluded` dropped, the offline->board transfer splits by entry size:
solos (n=5) carry +0.0650 with sd 0.0032, ensembles (n=23) +0.0614 with sd 0.0026. Ranking every one of
the 230 ordered pairs of ensemble anchors by offline and asking whether the board agreed:

    delta offline 0.000-0.002    86 pairs    15 inverted = 17%
    delta offline 0.002-0.005    73 pairs     4 inverted =  5%
    delta offline 0.005-0.010    49 pairs     0 inverted
    delta offline >= 0.010       22 pairs     0 inverted

So the ruler is strictly monotone above 0.005 offline and blind below 0.002. That floor is now the
number vetoing slots: the marginal seat on the diversity curve (+0.0004), `sided_pool` (+0.0007),
LogitAverage over RankAverage (+0.0014), the seat-oracle headroom (+0.0039) and split-half per-label
top-4 (+0.0043) all sit under it, which is why five separately-measured seating results read flat on
the board and why three different member sets have now all returned exactly 0.944.

BUT THAT FLOOR IS A REGRESSION RESIDUAL, NOT A MEASUREMENT. It is inferred from the scatter of a fit
across heterogeneous entries of different sizes and families. Nothing has ever put two NEARLY IDENTICAL
entries on the board and read the gap directly. This does.

THE PAIR. `stackfit_s2026` reads offline 0.8631 midpoint; `calib_dense_hold_s2026` reads 0.8634 and
is already on the board at 0.928 (`holdsolo_v1`, 2026-09-02). Their recipes are identical on every
recorded key — variant small, 336 px, 20 slices, band 0.1-0.9, `cls_mean` / attend / max, batch 12,
accumulate 1, 12 epochs, seed 2026, fold 0, backbone lr 4e-5 — and both were selected on the same fold-0
holdout. The offline gap is 0.0003, deep inside the blind band.

PRE-REGISTERED, and the branches are about the RULER rather than about the member:

  * **0.926-0.930** — the board reproduces a matched-offline pair to within its own rounding. The 0.005
    floor is real and is a property of the transfer, not an artefact of pooling heterogeneous entries.
    Every operation under it stays closed, and the +0.0110 offline debt the ensemble owes for 0.955 is
    the correct target to attack.
  * **0.920-0.925 or 0.931-0.936** — a gap of 0.003-0.008 opens between two models that agree offline to
    0.0003. Then offline is NOT sufficient to predict the board even at fixed recipe, the fitted residual
    understates the true scatter, and every board delta this project has read at this size is weaker
    evidence than it was recorded as.
  * **outside 0.920-0.936** — a gap larger than the entire measured residual range between two members
    that differ in nothing the ledger records. That impeaches the transfer framework itself, and the
    first suspect is the offline number rather than the board.

THIS IS AN INSTRUMENT AND IS SUBMITTED AS ONE. It predicts no gain over the champion and is not offered
as one: `stackfit_s2026` is the second-best solo the project owns and would land near 0.928, sixteen
thousandths under the pinned 0.944. It is NOT a seating, not a selection, not a re-weighting and not a
re-lensing — the operations the board has closed five times. It is a single member, alone, chosen
because its offline number collides with one already on the board.

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

    # A set of ONE. `LogitAverage` over a single member is the identity, named rather than special-cased
    # so the vote is the champion's own convention here too — what differs from `holdsolo` is WHICH
    # member, and the two of them agree offline to 0.0003, which is the entire point of the entry.
    chosen = ("stackfit_s2026",)

    # The whole session, not a share of it: no foreign producer runs first here, so there is nothing to
    # subtract. Spelled explicitly because `oofseat` subtracts and this file is read beside it.
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
