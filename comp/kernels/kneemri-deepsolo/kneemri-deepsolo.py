"""Does serving DEEPER than the fitted depth buy board on a lone member? The one ruler that can say.

WHAT THIS MEASURES, STATED FIRST. `extffsolo_v1` is `labelwise_ext_drop40_long_ff` served at its fitted
20 slices and it boards **0.945** — the best SOLO this project owns, and one thousandth under the
eight-seat champion `tripleseat_v2`. This entry is that member, byte for byte, served at **32**. Nothing
else moves: the band stays the fitted 0.1-0.9, so DENSITY alone changes and never coverage. The whole
entry is one constant.

WHY DEPTH AND WHY NOW. The depth ladder is monotone increasing where it has been measured -
`holdout_backfill --served-slices` on `band_dense_s2026`, band held, reads **0.8859 / 0.8877 / 0.8888
ledger75 at 12 / 20 / 24**, about +0.0024 per twelve slices. `band_dense_s2026` was itself fitted at 20,
so two of those three points are at or past the fitted depth: serving past the depth a member was fitted
at is not an extrapolation this project has to guess about — it has been measured once and it was
positive.

AND THAT OFFLINE RULER HAS ALREADY BEEN CHECKED AGAINST THE BOARD, ON THIS AXIS. `onegroup_v1` served
the champion six as one read group at 12 slices, boarded **0.942 against 0.944**, and its own board row
records that "THE OFFLINE TWIN PREDICTED IT EXACTLY". So the quantity this entry extrapolates is not an
offline number hoping to transfer - it is an offline number that HAS transferred, at full size, on a
depth change, once.

WHICH MAKES THIS THE FIRST UPWARD POINT. Every depth entry the board holds moves depth DOWN and every
one of them cost: `onegroup_v1` 0.942 at 12 slices, `cheapdecode_v1` 0.925 and `cheapwide_v1` 0.928 at
the 224 px / 12 slice corner. The board has therefore measured the descending half of this ladder three
times and the ascending half never, while the offline ruler says the ladder keeps climbing past 20. That
asymmetry is the gap this slot buys.

THE ONE TRANSFER ASSUMPTION, NAMED. The ladder was measured on `band_dense_s2026`, a `canonical`-scheme
member, and this one is `extended`. Both are 20 slices over band 0.1-0.9 and both are the same trunk and
pooling, so the assumption is that the per-slice information curve does not depend on the sampling
scheme. That is an assumption, it is unmeasured, and it is the most likely reason for a flat reading
that is not "depth stops paying".

WHY 32 AND NOT 24. Because 24 is unreadable. The measured step from 20 to 24 is +0.0011 and the board
reports at 0.001, so the modal outcome of shipping 24 is a tie that says nothing. 32 is the smallest
step that puts the projected move (+0.0024 offline, ~1.1 transfer, so ~+0.0026) clear of the board's own
resolution. The risk it buys is real and is the point: slot pooling was fitted over twenty slots and
thirty-two is further out of that distribution than anything measured, so this can come back negative.

WHY THE BOARD IS THE ONLY RULER, WHICH IS WHAT MAKES THIS AN INSTRUMENT. Two separate reasons, and
either alone is sufficient. The member is DARK - `--fold -1`, it holds out nothing, so no offline number
for it exists or can be made. And its fold-0 twin cannot be regrouped offline either: it reads the
`_extended` cache family, which exists on disk at 20 slices only, so a 32-slice reading of it would have
to be decoded first. That is hours of card for a number the board returns for a slot.

PRE-REGISTERED, against `extffsolo_v1`'s 0.945 and the board's 0.001 resolution:

  * **0.947 or above** - serve-depth extrapolation PAYS, by more than one resolution step. It is then a
    free term on every solo this project ships, it costs no training, and the next question is where the
    ladder turns over (40, 48) rather than whether it climbs.
  * **0.946** - it pays by one step and ties the champion with a SINGLE member. Real but unresolvable
    against scatter at n=1; re-read it on a second member before quoting a term.
  * **0.945** - FLAT. Depth past the fitted point is free but buys nothing, and the offline ladder's
    +0.0024 does not transfer. The axis closes and no future solo pays the decode.
  * **0.944 or below** - serving out of distribution COSTS, and the fitted depth is the right one. This
    is the branch that would retire the whole idea, and it is not the unlikely one.

THE ONE WAY THIS FAILS BADLY, AND WHY IT CANNOT HAPPEN HERE. `Ensemble` mitigates an overrun by giving
up the weakest member per chunk, and it stops at `len(surviving) <= 1` - a SOLO is never dropped. So a
solo that projects past the ceiling is not trimmed, it is killed at nine hours having written nothing,
and the slot scores nothing at all. That is the failure to rule out before shipping, not after. It is
ruled out structurally: `tripleseat_v2` serves THREE read groups of 20 slices - sixty slice-decodes a
study - plus eight member forwards, and it finishes. This entry is thirty-two slice-decodes and ONE
forward, under a third of the champion's bill on both axes.

NOT A RESEATING, WHICH IS WHY IT MAY HAVE THE SLOT. `submit_check` refuses it on SET equality against
`extffsolo_v1` and that refusal is correct and expected: the member set is IDENTICAL BY DESIGN. The
variable is the serving, and a check that guards member sets cannot see it. The `--anyway` says so.

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

SERVED_SLICES = 32
"""The one variable. The member was fitted at 20 and is served here at 32, band unchanged.

Not 24, which the offline ladder puts +0.0011 above 20 — under the board's own 0.001 reporting step, so
it would return a tie whatever the truth was. 32 projects +0.0024, which the board can resolve.

A SOLO IS WHY THIS IS AFFORDABLE AT ALL. Depth is the decode bill, and a lone member pays it once: the
champion serves eight seats out of three groups for 56 slice-reads a study, where this entry reads 32
for its one. The budget headroom that makes a 1.6x decode unremarkable here is exactly the headroom an
ensemble does not have, which is why this question has gone unasked while the pool grew.
"""

SERVED_BAND = (0.1, 0.9)
"""The member's own fitted band, held so DENSITY is the only thing that moves.

`SharedRead` spreads its indices across the band it is given (`SlotReader.band_indices`). Moving the
band as well is what charged band loss to depth in the retracted knee-uy8z0 ladder, and the offline
ladder this entry extrapolates from (0.8859 / 0.8877 / 0.8888) was itself measured band-held.
"""


def main() -> None:
    boot()

    from kneemri.inference.budget import (  # noqa: PLC0415 - the kit is on sys.path only after boot
        KERNEL_SECONDS,
        Budget,
    )
    from kneemri.inference.ensemble import Serving, SharedRead  # noqa: PLC0415 - same
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same

    # A set of ONE, identical to `extffsolo_v1`'s. `LogitAverage` over a single member is the identity,
    # named rather than special-cased so the vote stays the champion's convention here too.
    chosen = ("labelwise_ext_drop40_long_ff",)

    # `SharedRead` over a set of one is not a collapse — there is nothing to collapse — it is the only
    # surface this repo has for serving a fitted member at another depth, and the kernel and
    # `holdout_backfill --served-slices` reach the same object so a regroup has one definition.
    KernelRun(
        names=chosen,
        overlap=False,
        serving=Serving(
            vote=LogitAverage(),
            budget=Budget(seconds=KERNEL_SECONDS),
            group=SharedRead(SERVED_SLICES, SERVED_BAND),
        ),
    ).run()


if __name__ == "__main__":
    main()
