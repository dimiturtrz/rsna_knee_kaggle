"""INSTRUMENT, not a candidate: K=3 against K=4 on the full-data line, one variable.

WHAT THIS IS. `basinsoup_ff_k3_s2026` is the uniform weight average of the THREE 4-epoch seed-children
(seeds 301/302/303) forked off `labelwise_ext_drop40_long_ff` at `--fold -1` with `--init-scope all`.
`basinsoupff_v1` (board 0.946, ref 56406877) shipped the K=4 average of the same three children PLUS
that parent. The two differ in exactly one term: whether the parent anchor is in the pot. Single
checkpoint, single forward pass, same serving line, same kit.

WHY THIS IS AN INSTRUMENT AND SAYS SO. It proposes no new member and cannot be expected to beat the
pin. Its whole job is to attribute the +0.001 that `basinsoupff_v1` (0.946) bought over the same parent
shipped solo as `extffsolo_v1` (0.945):
to the CHILDREN'S AVERAGE, or to the parent anchor that K=4 keeps and K=3 drops. Every slot this week
has been priced against a soup family whose next step nobody can name, and this is the reading that
names it.

WHY IT CANNOT BE READ OFFLINE. Both products are `--fold -1`: full data, nothing held out, so no
holdout record for either exists or can be made. On the fold-0 line the same comparison has now been run
TWICE - `basinsoup_k3`/`k4` and `basinsoup_dmlb_k3`/`k4` - and both pairs read IDENTICAL to four
decimals, the direct measurement that the children never left the parent's point. If that carries to the
full-data line, this entry ties 0.946 and the anchor is inert. The board is the only ruler that can say
so, and it costs one slot.

PRE-REGISTERED, against `basinsoupff_v1`'s 0.946 and the 0.001 board resolution:

  * **0.946** - K=3 and K=4 agree on the board as they agreed twice offline. The parent anchor
    contributes nothing, the gain belongs to the children's average, and the soup is genuinely a
    property of averaging seed-children rather than of perturbing a known-good point. Ship either.
  * **0.945** - the anchor is load-bearing: dropping it costs exactly the +0.001 the soup bought, which
    makes the operation a small perturbation of the parent rather than an averaging effect, and predicts
    that iterating it saturates.
  * **0.947 or above** - the children's average is BETTER than the anchored one, the anchor was holding
    the product back, and K=3 becomes the shipping form of every future soup.

WHAT IT DOES NOT DO. It does not attack the 0.011 gap to first place. Nothing on the shelf is priced
within a factor of six of that gap tonight, and the pacing rule's own instruction when nothing
mechanism-backed can beat the champion is to spend the slot on a declared instrument instead of a
reseating. This is that instrument, and the submission message says so.
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
    # special-cased so the vote stays the champion's convention here too. The member is DARK:
    # it holds out nothing, so no offline number for it exists or can be made, and the board
    # number this entry returns is the only reading it will ever have.
    chosen = ("basinsoup_ff_k3_s2026",)

    # The whole session, not a share of it: no foreign producer runs first here, so there is nothing to
    # subtract. Spelled explicitly because `oofseat` subtracts and this file is read beside it.
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
