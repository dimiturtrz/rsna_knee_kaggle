"""The champion line plus ONE member from a different encoder family — the axis no entry has ever varied.

Seven entries have scored on this board and every one of them is 100% DINOv2-small. This is `longseat`'s
four (0.930, the current joint maximum) with `dinov3_base_lora_s2026` added as a FIFTH: DINOv3-base/16
under rank-8 adapters, 768 wide against small's 384, patch 16 against 14, distilled from a 7B teacher.
Nothing else about the entry moves — same RankAverage vote, same `overlap=False`, same four incumbents.

THE OFFLINE RULER REJECTS THIS MEMBER AND THE NUMBER IS STATED. Rank-averaged over the 909-study fold-0
holdout the five read **0.8678** against the four's **0.8691** (-0.0013) and against the pinned champion
`greedy4_v1`'s **0.8679** (-0.0001). Its best available swap is worse still, -0.0025. It scores 0.8268
alone, the second-weakest checkpoint in the mounted pool.

WHY FIRE IT ANYWAY — THE RULER IS BIASED, NOT MERELY NOISY. Collapsing three members into a single pixel
group cost 0.0059 on this holdout and 0.014 on the board: our own ruler under-prices DIVERSITY by 2.4x
(knee-nas5, knee-7d1). An encoder change is a strictly larger diversity step than the decode change that
measurement was taken on — different pretraining corpus, different teacher, different patch grid, so the
errors it makes are not a re-parameterisation of the errors the other four make. A -0.0013 reading is a
fifth of the 0.0062 three-seed noise floor and sits well inside the region where this ruler has twice
been shown to point the wrong way. `calib_span` is the precedent: mounted at -0.0020 offline, it carried
the 0.929 entry.

AND THE POOL IS OUT OF ANYTHING ELSE. `longseat` spent the largest seat move the mounted pool contained
(+0.0012, six times the runner-up, measured over all 56 same-split checkpoints) and tied the board. The
ledger's conclusion was that the mounted pool is exhausted as a source of board movement and that the
argument is now for an axis the pool does not contain. This is that axis.

PRE-REGISTERED, WITH THE REFUTATION NAMED FIRST. **0.932 or above** says encoder diversity transfers and
that the 2.4x under-pricing is a property of DIVERSITY rather than of the decode axis it was measured on;
the next card-time then goes to a second encoder trained properly rather than to another DINOv2 recipe.
**0.929 or below refutes it**, and that is the outcome this entry exists to make possible: it would say
the offline under-pricing does NOT extend past decode, that a member 0.032 below the pool's best cannot
be bought back by being different, and that the DINOv3 line is dead as a member source — closing an axis
that has cost this project four arms and never once shipped. **0.930 or 0.931** is the reporting
resolution around a tie: encoder diversity is then free but not profitable, and the entry is evidence
that a WEAK member of a different family costs nothing, which makes a STRONG one worth the card.

PAIRED WITH `kneemri-maxdiv`, deliberately and in the same window. That entry is this one plus
`crop80_tissue_s2026`, the pool's only different physical crop. The two are nested, so their difference
isolates the crop axis while this one isolates the encoder axis, and the pair brackets the question: if
both tie, diversity is not a board lever at any dose and recombination stops for good.
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
        "band_dense_s2026",
        "calib_span_s2026",
        "depth_regions_narrow_s2026",
        "band_dense_long_s2026",
        "dinov3_base_lora_s2026",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
