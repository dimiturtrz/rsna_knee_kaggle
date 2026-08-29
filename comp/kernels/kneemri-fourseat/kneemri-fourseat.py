"""A FOURTH seat on the trio's single decode — the seat term, measured where it is free.

`trioseat_v1` scored **0.938** with three whole-corpus members sharing one pixel group (20 slices, band
0.1-0.9, 336). This entry adds ONE member on that same geometry and changes nothing else: same decode,
same `overlap=False`, same vote, same three incumbents. A fourth forward pass on a card that was idle
while pydicom worked.

**WHY THIS MEMBER, AND WHY IT COSTS NO CARD TIME.** `decode_entry` (new 2026-08-29) searched 37 members
under a ceiling on decodes and reported the best one-decode roster it could build: seven seats, all on
this same read A, 0.8736 offline against `honest12`'s 0.8750 across THREE decodes. Read A beat read B by
+0.0067, the only separation measured that day that clears the 0.0062 floor. `small_batch6` is the
walk's second pick on that read, and it is the only one of the four extra seats that already exists as a
whole-corpus checkpoint (`small_batch6_ff`, `--fold -1 --ema`) AND is already inside
`dimiturnt/kneemri-members`. So this is a kernel edit: no fit, no push, no decode.

**WHY IT IS NOT THE SEVEN-SEAT ENTRY.** The other three seats exist only as `_s2026` — fold 0 — and that
experiment is already on the board. `greedy7_v1` served seven fold-0 members for **0.933**, five
thousandths BELOW this entry's own three-seat base. Fold-0 seats dilute an all-whole-corpus roster
faster than the seat count repays, so the seven-seat version needs three refits at `--fold -1` (about
three hours of card) rather than a slot. THIS ENTRY IS THE INSTRUMENT THAT DECIDES WHETHER TO SPEND
THEM.

**PRE-REGISTRATION, written before the number.** Base is `trioseat_v1` = 0.938 at three seats / one
decode. `calibsix_v1` = 0.941 and `raptorseat_v1` = 0.943, both across three or more decodes.

- **0.941 or above** — a fourth free seat reaches the three-decode entries at a third of their runtime.
  The seat term on a paid read is real and large, the three `_cf` refits are licensed immediately, and
  this is the efficiency entry to pin.
- **0.939-0.940** — a directional gain of one to two thousandths, inside the 0.0053 Hanley-McNeil floor
  and NOT a result on its own. It still licenses the refits, because four seats moving the number at all
  is what the seven-seat version scales.
- **0.936-0.938** — the seat term is spent at three. `trioseat`'s +0.013 over `soloseat` came from the
  first two extra seats and saturates there; the refits are NOT run and the efficiency lane's next move
  is a better single member, not more of them.
- **Below 0.936** — a fourth seat DILUTES even at whole-corpus fit. That would contradict `trioseat`'s
  own reading and should be read as a question about `small_batch6_ff` specifically (it is the one
  member here fitted at batch 6) rather than about the seat axis.

**WHAT IT CANNOT DO.** One board read on one hidden test set, and it cannot separate "a fourth seat"
from "this fourth seat" — the same limit `trioseat` recorded. Its members are all `--fold -1`, so no
offline score exists for this entry and none may ever be reported for it.
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

    # trioseat's three, plus the one read-A seat that is both whole-corpus and already packaged.
    chosen = ("final_ws_cf", "grade_ladder_cf", "through_dense_cf", "small_batch6_ff")
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
