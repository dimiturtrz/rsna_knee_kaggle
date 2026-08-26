"""The champion's six, with the one decode-path axis it does not vary.

**ONE SEAT MOVES AND NOTHING ELSE.** This is `kneemri-blendseat` — the champion at 0.939, six members
served under `LogitAverage` with `overlap=False` — with `lr4e-5_ws_ff` replaced by `res392_ws_s2026`.
Every other seat, the vote rule, the kit and the serving config are untouched, so the diff is one line.

**WHY THIS AXIS.** All six champion seats read the same 336 px input. The only ensemble diversity this
project has measured TRANSFERRING to the board is decode-path diversity, and it transfers at 2.4x its
offline size (collapsing three members onto one pixel group cost 0.0059 offline and 0.014 on the board;
knee-nas5). `res392_ws_s2026` reads 392 px. It is the only resolution the champion does not contain.

**THE SUBSTITUTION IS QUALITY-NEUTRAL BY CONSTRUCTION.** The outgoing seat's fold-0 twin
`lr4e-5_ws_s2026` reads 0.8525 solo; the entering `res392_ws_s2026` reads 0.8523. Two ticks apart. So
the seat's quality is held approximately fixed and its decode path is what changes — which is the only
way to buy diversity without paying for it in member strength.

**THE OFFLINE RULER IS BLIND HERE, AND THIS ENTRY SAYS SO FIRST.** Scored on the champion's fold-0
twin six (0.8704 base), eleven seat variations were measured on 2026-08-26: the entire span was
0.8687-0.8705, against a three-seed noise floor of 0.0062. This swap read 0.8705, the joint best, at
+0.0001 — which is not evidence and is not quoted as any. No candidate available tonight has offline
evidence of beating 0.939; that was established before this kernel was written, not after it scored.
This entry is therefore an explicit BET on the one mechanism (decode-path diversity) that the board has
priced higher than the holdout does, chosen because the offline ruler cannot separate the field.

**COUNT IS HELD AT SIX, DELIBERATELY.** Adding `band140_s2026` as a seventh tied this swap offline at
0.8705. It was rejected on board evidence rather than offline: every seven-member entry this project has
fired caps at 0.937 (five of them), while both 0.939 entries are six-member. A fold-0 seat carries no
such penalty — `merged_blend_s2026` is fold-0 and sits in both 0.939 entries — so the second fold-0 seat
this swap introduces is the cheaper of the two structural costs.

PRE-REGISTRATION, written before the number, against blendseat's 0.939.

- **0.941 or above** — decode-path diversity transfers at or above its measured 2.4x even when the
  entering member is no stronger than the one it replaces. Resolution becomes a seat-design axis: build
  392 px counterparts for two more seats and re-serve.
- **0.940** — the wall breaks by one tick, which is the outcome this entry was fired for. Honest read:
  a one-tick move at the board's own scatter cannot distinguish the mechanism from luck, so it licenses
  ONE confirming entry on the resolution axis and nothing larger.
- **0.937-0.939** — flat. The most likely outcome. Either decode-path diversity does not extend to
  resolution (as opposed to crop and field, where it was measured), or the second fold-0 seat ate the
  gain. Neither is separable from one entry, and the axis closes without another slot.
- **0.936 or below** — the swap cost real points at equal member quality, which would mean the seat was
  carrying something other than its solo score. That retires resolution as a diversity axis and puts the
  second-fold-0-seat cost back on the table as a live confound for every future swap.

COST: one slot, zero card time, nothing retrained. Both members already exist and are already mounted.
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

    # `res392_ws_s2026` stands where `lr4e-5_ws_ff` stood. The order is the champion's, unchanged,
    # so the diff against `kneemri-blendseat` is one line and reads as one.
    chosen = (
        "final_ws_e12",
        "calib_span_ff",
        "grade_ladder_ff",
        "merged_blend_s2026",
        "res392_ws_s2026",
        "through_dense_ff",
    )
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage())).run()


if __name__ == "__main__":
    main()
