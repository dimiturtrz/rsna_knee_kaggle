"""What does a runtime cut ALONE move on the Efficiency board, when the score does not move with it?

THE QUESTION. `efficiency_read --live` (2026-09-20T04:02Z) puts us at efficiency rank 17 at 0.945 and
says of the teams ahead: **6 teams rank above us on efficiency while scoring BELOW us - runtime alone**.
So the thing separating us from them is not accuracy, and no better member can answer it. This entry
changes runtime and nothing else, and it is the only way to learn the exchange rate that board runs on.
The efficiency track is ADDITIVE: it cannot touch the standing accuracy entry, which the board keeps.

THE MEMBER IS UNCHANGED - the SAME single checkpoint `extffsolo_v1` shipped at board 0.945 (ref
56312235), the same weights, the same band, the same 336 px, the same vote. The ONE difference is the
depth it is READ at: 20 slices becomes 12, through `SharedRead`, which is the only serving knob that
moves DECODE rather than forward time. `SlotReader.cropped` decodes exactly one file per sampled index
(`slot_reader.py:365`), so 12 of 20 is 40% of the DICOM reads removed - and decode is "the I/O-bound
half on a Kaggle host with four CPUs". `read_size` cannot do this: sizes collapse inside a read group
because the crop stops before the resize, so resolution buys forward time only.

`submit_check` will refuse this on SET equality with `extffsolo_v1` and it is RIGHT to: the member set
is identical. It ships with `--anyway` because the variable here is the reading, not the roster.

THE PRICE, MEASURED ON THE OBJECT THAT SHIPS. An earlier version of this file carried a `member_curve
--slices` ladder pricing 16 slices at -0.0008. That ladder was RETRACTED before it was ever submitted:
`member_curve --slices` centres a window on the cached slab, which narrows the band (0.1-0.9 to about
0.26-0.74 at 12) as well as the depth, and `slot_reader.py` prices that band narrowing at 0.0040 on its
own - the same size as the effect. Every rung was measuring two things at once.

Re-measured with `holdout_backfill --served-slices 12 --served-band 0.1 0.9`, which builds the
`Serving(group=SharedRead(slices=12, band=(0.1, 0.9)))` this kernel actually runs and resamples 12
indices across the SAME band. Fixed corpus, fixed band, fixed pixels, fixed epoch, population eb3ba199,
909 studies, `ledger75`:

| member | head | native | 12 slices | cost |
|---|---|---|---|---|
| `base_dense_s2026` | dense | 0.8812 | 0.8778 | -0.0034 |
| `labelwise_dense_s2026` | labelwise | 0.8964 | 0.8925 | -0.0039 |
| `labelwise_depth20_b10_s2026` | labelwise | 0.8970 | 0.8974 | +0.0004 |
| `labelwise_depth24_drop40_long_s2026` | labelwise | 0.9053 | 0.9038 | **-0.0015** |

plus, independently and already on disk, `read_groups.py`: the champion's five fold-0 twins read 0.8706
native against 0.8684 at 12 slices / band 0.1-0.9, a cost of 0.0022. Five readings, two heads, all
between 0.0000 and 0.0039 - every one inside the 0.0053 paired-holdout floor. This retires the worry
that our labelwise head, with per-label slots, would spend slices a max-pooling head could not: it does
not, and its mean cost is SMALLER than dense's.

THE LAST ROW IS THE CORNER ITSELF, and it is why the transfer is no longer the weak part of this entry.
`labelwise_depth24_drop40_long_s2026` is the drop40_long corner this member lives in - and it was read
at 24 -> 12, a 2x cut, where the shipping member runs 20 -> 12. The harsher cut on the nearer member is
the CHEAPEST reading of the four, which is the opposite of slice-hungry.

WHAT IS STILL ASSUMED, stated in the submission message. None of the five IS this entry's member, and
it cannot be: `labelwise_ext_drop40_long_ff` is a `fold -1` full fit, so it held nothing out and has no
holdout to re-derive - correct behaviour, not a gap. Its fold-0 twin `labelwise_ext_drop40_long_s2026`
carries `scheme: extended`, and the corpus cache suffix comes from THAT key, so it wants a 12-slice
extended corpus that is not on disk (only the 20-slice one it trained on). Reading it would cost a full
decode, which is why the corner was read on its canonical sibling instead.

PRE-REGISTERED, on the EXCHANGE RATE rather than on the score. At the top-band `ledger75` slope of
+1.09 to +1.17, the corner's 0.0015 offline cost predicts board **0.943** against this member's own
0.945; the three-member transfer's wider 0.003 predicts 0.941-0.942. Either is inside the branch below.

  * **board 0.941-0.942 and efficiency rank improves** - the transfer holds and runtime is purchasable
    at about 0.001 of board per 13% of decode. Then the next efficiency entry caps harder, not smarter.
  * **board 0.941-0.942 and efficiency rank does NOT improve** - the metric does not weight decode the
    way the 6-teams-above-us reading implies, and slice count is closed as an efficiency knob. That is
    the most useful thing this slot can return, because it retires the whole axis.
  * **board <= 0.939** - now the UNLIKELY branch, and a sharper finding than it was. It would mean the
    `scheme: extended` corpus, the ONE axis separating this member from the corner read above, is what
    spends slices - which no reading has ever isolated, and which would make the 12-slice extended
    decode worth paying for.

Every reading above is a FLOOR in the honest direction: this member was TRAINED at 20, so a capped read
pays a domain shift on top of the information it drops. A member trained at 12 should beat it.
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
    from kneemri.inference.read_groups import SharedRead  # noqa: PLC0415 - same
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same

    # The SAME member extffsolo_v1 shipped at 0.945. Named identically on purpose: this entry's
    # variable is the reading, so the roster must not move underneath it.
    chosen = ("labelwise_ext_drop40_long_ff",)

    # Depth only. The band is the member's own recorded [0.1, 0.9], passed unchanged so the regroup
    # moves ONE axis — a band change would decode different files too and confound the reading with it.
    lean = SharedRead(slices=12, band=(0.1, 0.9))

    KernelRun(
        names=chosen,
        overlap=False,
        serving=Serving(vote=LogitAverage(), budget=Budget(seconds=KERNEL_SECONDS), group=lean),
    ).run()


if __name__ == "__main__":
    main()
