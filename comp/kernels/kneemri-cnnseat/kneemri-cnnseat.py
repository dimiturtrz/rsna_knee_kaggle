"""cnnseat — the champion seven, plus the first non-transformer member this project has ever fitted.

**WHAT IS BEING PRICED.** An ARCHITECTURE FAMILY, not a seating. Every one of the 122 members on the
shipped roster is a vision transformer or a hybrid; twelve of the thirteen registry encoders are. The one
convolutional line, `radnet121`, closed EMPTY at 0.6716 — and that number is a reading about RadImageNet's
weights, not about convolution. No ImageNet CNN had ever been fitted here, so "CNNs are weak on this task"
was a belief the ledger could not support in either direction. `resnet_224_short` makes it a number:
resnet50d at 224 px, 2.5D through-plane RGB, max over 20 slices, full-trunk finetune at 2e-4, twelve
epochs — the budget every other seat on the ledger was fitted under. It peaked at **0.8527 on the 909 at
epoch 8** and converged, which TIES the best single member the project owns (`res392_ws` 0.8523).

**WHY THIS IS NOT AN EIGHTH FLAT RESEATING.** knee-urvf's standing result is that every operation on the
champion's SEATING is inert — seven board flats, and every one of them a permutation, a weight, a count or
a refit WITHIN the ViT family. This entry does not move who sits or how they are weighted. It adds one
seat from a family the board has never held: the only prior non-ViT seat is `raptor_v4_swa`, which is
FOREIGN (someone else's checkpoint, mounted, not ours) and is already in this entry unchanged at 1/3.

**THE OFFLINE RULER SAYS IT CANNOT TELL, AND THAT IS THE POINT.** `seat_value --against honest12 --ruler
ledger75` reads **+0.0007** as a thirteenth seat and **+0.0020** as the best swap (for `mae_matched`) —
both under the +0.0040 under which this ruler has never been shown to pick a board tick. So no score is
claimed from it. What the member axes DO say is that it is the odd one out: nearest neighbour 0.8581
against the roster's own 0.8956 boundary, js 0.0188, maha 4.654 — the DISTINCT cell, not the NOISY one.
D277's finding is the reason that disagreement is worth a slot rather than a shrug: method ranking does
not reliably survive a cohort change, our holdout ordering is an in-distribution statement about 909
studies, and the two members that have actually MOVED this board were both out-of-distribution. An
in-distribution rank ruler is the exact instrument that cannot answer this question.

**IT IS ALSO THE MOUNT'S FIRST EXECUTION.** `dimiturnt/resnet50d-224` has never run on Kaggle, and neither
has any convolutional encoder path in the kit. `farseat_v1` is what that costs when it is not checked: a
COMPLETE kernel, a real `submission.csv`, a board score, and one member silently dropped for five days.
`run_check` is run on this kernel's own log before the slot is counted, and the census must read 7 of 7.

**EVERYTHING ELSE IS THE CHAMPION, UNCHANGED.** Same six, same order, same `overlap=False`, same
`LogitAverage`, same raptor checkpoint at weight 1/3, same rank-space join by study.

PRE-REGISTRATION, written before the number, against the champion's 0.944 and `BOARD_RESOLVABLE = 0.0015`.

- **0.946 and above** — architecture family is a live axis and the offline ruler is blind to it exactly as
  D277 predicts. Re-pin `comp/champion.json`, and the 48-epoch arm plus a second CNN family (convnextv2,
  already on the registry) take the next slots ahead of everything else.
- **0.9455 to 0.9459** — a real gain, one resolution step clear. The family is worth a seat; continue with
  the long arm before widening.
- **0.9425 to 0.9455** — FLAT. Then the offline +0.0007 transferred honestly and a tied-solo CNN adds
  nothing a ViT of the same solo would not. This does NOT close the family: `resnet_224_long` (48 epochs,
  his recipe's real ask) is on the card as this ships, and a member fitted to a quarter of its budget is
  the weakest possible test of the axis. What it DOES close is the cheap version of the question.
- **Below 0.9425** — the seat is actively harmful, which for an equal-weight average of seven would be the
  first such reading this project has taken. Check `run_check`'s census FIRST: a dropped member and a
  harmful member look identical from the board.
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

    from kneemri.kaggle.seated_entry import SeatedEntry  # noqa: PLC0415 - the kit is on sys.path only after boot

    chosen = (
        "final_ws_cf",
        "calib_span_ff",
        "grade_ladder_cf",
        "stack_increments_ff",
        "lr4e-5_ws_cf",
        "through_dense_cf",
        # THE ONLY LINE THAT DIFFERS FROM THE CHAMPION. It sits last so a diff of the two kernels is one
        # added row: the six above are in the champion's own order, unchanged, and nothing about how they
        # are weighted or joined moves. This member lives on the SECOND shelf
        # (`dimiturnt/kneemri-members-2`), which is why this kernel mounts one dataset the champion does
        # not, and its encoder mounts from `dimiturnt/resnet50d-224` — neither has ever run on Kaggle.
        "resnet_224_short_s2026",
    )
    SeatedEntry(
        names=chosen,
        # A quality-weighted share of two SYSTEMS, not `raptorseat_v1`'s 1/7 seat-count convention.
        weight=1.0 / 3.0,
    ).run()


if __name__ == "__main__":
    main()
