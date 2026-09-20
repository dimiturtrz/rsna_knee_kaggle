"""Is the full-corpus refit term a CONSTANT, or does it depend on the member it is applied to?

WHY THIS ENTRY EXISTS. Every `--fold -1` refit this project ships is priced by adding a single number
to the honest twin's board score, and that number has been read exactly twice:

| member | honest | refit `_ff` | term |
|---|---|---|---|
| `labelwise_depth24_drop40_long` | 0.941 (`l75solo_v1`, 56309162) | 0.944 (`ffsolo_v1`, 56296698) | **+0.003** |
| `labelwise_ext_drop40_long` | 0.937 (`extsolo_v1`, 56315030) | 0.945 (`extffsolo_v1`, 56312235) | **+0.008** |

Two readings that differ by 0.005 — more than twice the top band's residual sd of 0.0022 — are not a
constant and are not yet a shape. Two points cannot separate "the term is noisy around +0.005" from
"the term is larger where the honest twin scored lower". This member is the third point, and it is
placed where the two candidate readings disagree most: `headlr_ship_s2026` is an honest 0.937, the
SAME abscissa as the ext pair, so under "depends on the honest level" it predicts ~0.945 and under
"constant near the mean" it predicts ~0.942.

THE MEMBER. `headlr_ship_ff` is `headlr_ship_s2026`'s line with `--fold -1 --ema`: it trains on all
4407 studies and holds nothing back. Converged, epoch 23 of 24 recorded. It is DARK by construction
(knee-s8kim) — every holdout study is in its training set, so no ledger row can price it, none is
owed, and its `offline` cell on the board record is null rather than a number. Its only ruler is the
board.

THIS IS AN INSTRUMENT AND IS SUBMITTED AS ONE. The pinned champion is `tripleseat_v2` at 0.946.
Carried off a 0.937 honest twin, the two known terms predict **0.940-0.945**, so P(this beats the
champion) is about 0.10 and this entry is not a champion attempt. The standing entry is not at risk:
the board keeps the best.

PRE-REGISTERED, on the TERM rather than on the member:

  * **board 0.944-0.945** — the term tracks the honest level, the ext pair was not a fluke, and every
    future refit of a low-scoring honest member is worth ~+0.008 rather than +0.003. That doubles the
    value of the refit step on exactly the members we would otherwise not ship.
  * **board 0.940-0.943** — the term is roughly constant near +0.005 with real scatter, and the ext
    pair's +0.008 was the high tail. Refits stay worth shipping, priced at +0.005 with a wide band.
  * **board <= 0.939** — the term is not reliably positive at all, and both prior reads were scatter
    around something near zero. That retires `--fold -1` as a free board gain and makes every entry
    built on one unpriceable, which is the single most useful thing this slot can return.
  * **board >= 0.946** — the term is larger than either reading and the refit alone reaches the
    champion. Then the next member to ship is a refit, not a new recipe.

WHAT IT COSTS: no GPU. The checkpoint was fitted by an arm that finished 2026-09-18T02:16Z and its
encoder already mounts in every entry we ship.
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

    # A set of ONE, and it has to be: the reading is about the refit TERM on one member, and a second
    # name would add a seating effect on top of the term this entry exists to measure.
    chosen = ("headlr_ship_ff",)

    # The whole session, not a share of it: no foreign producer runs first here, so there is nothing to
    # subtract. Spelled explicitly because `oofseat` subtracts and this file is read beside it.
    KernelRun(
        names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS))
    ).run()


if __name__ == "__main__":
    main()
