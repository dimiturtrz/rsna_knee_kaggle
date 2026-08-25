"""allfit's seventh seat given to the pool's most DISTANT member instead of its second-best one.

**THE SINGLE VARIABLE, AND WHY IT IS CLEANER THAN THE PRECEDENT AGAINST IT.** `kneemri-allfit` scored
0.937 with six dark members plus `band_dense_long_s2026` (0.8587, DINOv2-small, a decode-axis sibling of
four seats already in the room). This entry holds all six dark seats byte-identical, holds the count at
seven, holds `RankAverage`, holds `overlap=False`, and moves ONLY who sits seventh — to
`lora_r8_v3small_224_s2026` (0.8215), the pool's only non-DINOv2 member.

The ledger's precedent points DOWN and it must be stated first: `crossenc` and `maxdiv` each ADDED an
individually weak member (0.8268, 0.8345) and gave back four board ticks. But both of those moved member
COUNT and member QUALITY together, which is the exact confound `comp/champion.json` names when it retires
the old FOUR MEMBERS principle. A swap at fixed count does not have that confound. Quality moves alone
here, against distance, which is the question those two entries were read as answering and did not.

**WHY DISTANCE IS THE AXIS WORTH THE SLOT.** The pin's own strongest surviving selection rule is
`THE SEAT COUNT IS NOT THE VARIABLE - THE SEVENTH MEMBER'S TYPE OF DISTANCE IS`: five entries held six
members fixed and moved only the seventh, and a near-twin scored 0.930, a recipe-distant member 0.931, an
objective-distant member 0.931, and a PIXEL-distant member 0.932. Distance type ordered the board where
solo quality did not. This member is distant on the one axis none of those five varied — the TRUNK. It is
the only checkpoint in the package whose encoder family differs from every seat around it, and
`member_distance` measures it at nearest 0.8455 against a seated spread of 0.88-0.89, which is that
tool's own definition of genuinely new information rather than a re-parameterisation of what is already
seated.

**TWO THINGS THIS ENTRY IS NOT.** It is not a distance-RANKED selection: the pin retires residual distance
as a selection rule after it failed twice in one day in opposite directions, and nothing here ranks
candidates by it. Distance is the reason this ONE swap is interesting, not the rule that picked it. And it
is not a claim about DINOv3: `v3base_dense` shows the same encoder fully adapted reads 0.8455 against this
member's LoRA-over-frozen 0.8268, so 0.8215 prices an adaptation REGIME, not an architecture.

**IT CANNOT BE PRICED OFFLINE, AT ALL.** Six seats are `--fold -1` and have seen the entire 909-study
holdout, so no seat value, curve or joint mean may be computed for this roster in either direction. That
is not an economy; it is the standing cost of the all-data axis, and it is why this is a board bet.

PRE-REGISTRATION, written before the number, against `allfit`'s measured 0.937 on the identical six.

- **>= 0.940** — trunk distance pays where decode distance is spent, and it pays enough to cover a 0.037
  solo deficit. The pool's whole diversity account moves to the encoder family, and the next card work is
  a STRONG non-DINOv2 member (`v3base_own_recipe` is written and waiting) rather than another decode knob.
- **0.935-0.938** — flat. The seventh seat is insensitive to a 0.037 solo spread AND to a trunk change,
  which together say the seat is saturated rather than mis-filled, and member-side work should stop.
- **<= 0.934** — quality is load-bearing at fixed count, which is the reading `crossenc`/`maxdiv` were
  given without earning it. That retires trunk diversity as a cheap seat and makes member STRENGTH the
  constraint, so the pool needs a strong distant member built, not a weak one seated.

Every band is about one board tick against a board that has returned 0.936-0.937 four times, and the
house bar is 0.02. So no outcome here is a RESULT. What each outcome does is tell the next card-hour
which way to point, and that is the whole of what a slot buys at this saturation.
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

    chosen = (
        "final_ws_e12",
        "calib_span_ff",
        "grade_ladder_ff",
        "lora_r8_v3small_224_s2026",
        "base_dense_ff",
        "lr4e-5_ws_ff",
        "through_dense_ff",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
