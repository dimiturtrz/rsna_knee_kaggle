"""The encoder family varied with the STRONG reading of it, for the first time on this board.

**THE SINGLE VARIABLE, and it sits in a seat with two matched controls at the same number.** `allfit`
(0.937) is six `--fold -1` members plus the fold-0 seat `band_dense_long_s2026`. `lastseat` (0.937) is
the same six with `small_batch6_s2026` in that seat instead. So seat four has already been exchanged
once between two DINOv2-small members and the board did not move. This entry exchanges it a second time,
for `v3base_dense_s2026` — a DINOv3-BASE trunk. Count (7), vote (`RankAverage`), the all-data dose (six),
kit and serving are all held. Only the ENCODER FAMILY of the seventh seat moves, and the two controls
mean the seat itself is measured rather than assumed.

**WHY THE STRONG READING IS THE NEW PART.** Every board entry that has ever varied the encoder family
varied it with `dinov3_base_lora_s2026`, 0.8268. That number is not a fact about DINOv3: it is
LoRA-over-frozen measured against full adaptation on the same trunk, which reads **0.8455**, +0.0187 for
the adaptation alone. It is one of the two inherited-hyperparameter defects the 2026-08-24 audit found
written into the ledger as facts about architectures. The package has held only the weak end since
2026-08-20; `v3base_dense_s2026` was mounted 2026-08-25 00:47Z and this is its first entry.

**WHAT COSTS WHAT.** One slot, ZERO card — the member is on disk and converged (12 epochs asked, 12 run,
peak before the last), and its trunk weights have been mounted at `dimiturnt/dinov3-hf/vitb16` since
2026-08-12. The card is biohub's until ~02:00Z and this run does not want it.

**THIS SEAT HAS AN OFFLINE NUMBER AND IT IS NOT POSITIVE.** `2026-08-24_architecture_diversity_is_closed_
and_the_empty_cell_is_strong_and_distant.md` measures `v3base_dense`'s seat value with `record_mix` over
the 909-study holdout, roster fixed: **-0.0007 as a sixth, -0.0005 as a swap.** A tie. The note closes the
architecture axis on that plus `cnx_dense` (-0.0042) and `vjepa_warm` (+0.0002), and its mechanism reading
is sharp: every one of those arms holds the slice sampling, the 2D-slice-then-pool architecture, the
targets, the objective and the head fixed, so a trunk swap changes how features are SPELLED and not what
the model is optimised toward — and the errors an ensemble needs uncorrelated are made downstream of all
of it. It further re-reads `base_dense`, the one paying encoder seat, as bought by its 768 width rather
than by family, which removes the last data point that looked like architecture diversity paying.

**SO THIS ENTRY IS NOT SOLD AS AN IMPROVEMENT, AND IT IS NOT ARGUING WITH THAT NOTE.** It is spending a
slot on the one thing the note could not do: the axis is closed on the OFFLINE ruler, and this session has
two board entries saying that ruler is anti-correlated at the top (`peak8` 0.8731 -> 0.932 and `greedy7`
0.8712 -> 0.933, both BELOW `lastseat`/`allfit` 0.937 at lower offline). -0.0005 offline is -0.001 board
at the measured transfer, which is a sixth of the noise floor: the offline reading predicts a TIE, not a
loss, so the downside is bounded and the question is whether the board sees something the holdout cannot.

**WHY IT COULD BE WORTH MORE THAN THE OFFLINE NUMBER SAYS.** The holdout under-prices diversity by about
2.4x on this pool — collapsing three members into one pixel group cost 0.0059 offline and 0.014 on the
board (knee-nas5, knee-7d1) — and an encoder change is a strictly larger diversity step than a decode
change. `calib_span` is the precedent: mounted at -0.0020 offline, it carried the 0.929 entry. Against
that, the offline ruler is measured SATURATED at the top of our range (0.8712 -> 0.933, 0.8731 -> 0.932),
so it is not a ruler this entry can be pre-judged on either way.

**NO OFFLINE NUMBER IS REPORTED AND NONE CAN BE.** Six of these seven have seen the entire 909-study
holdout, so any pooled offline mean is a fit statistic and not a score.

PRE-REGISTRATION, written before the number, against the 0.937 that BOTH controls read.

- **0.940 or above** — the offline seat ruler is not merely noisy but WRONG-SIGNED on the family axis, and
  encoder-family diversity pays at the champion operating point when the foreign trunk is strong. That
  re-opens a family the interpretations closed, and it does so with the only instrument entitled to: two
  more trunks (RadDINO, V-JEPA) were entered into the ledger from CRASH readings and would be next.
- **0.936-0.939** — FLAT, which is what BOTH the two board controls and the -0.0005 offline seat value
  predict, from opposite directions. Three rulers agreeing is a closure rather than a null: the
  architecture axis is then closed on the board as well as offline, the diversity-under-pricing argument
  has its first clean board refutation rather than an inference from a pixel-group collapse, and seat
  search closes for good. This is the expected outcome.
- **0.933-0.935** — the swap COSTS, which given the 0.0187 the member gains over its family twin means
  what the vote wants from that seat is agreement with the other six, not distance from them. That is a
  reading about the vote, not about DINOv3, and it argues for LogitAverage (`kneemri-votefit`) next.
- **0.932 or below** — indicts the mount or the serving path for this trunk, not the member. First
  suspect is the kernel loading vitb16 differently than training wrote it; `witness_check` covers exactly
  that and must be re-run before the number is believed.

`board_transfer` currently reads **+1.72** board points per holdout point (2026-08-27, fitted on the
completed `comp/board.jsonl` — the +1.82 it read before was fitted on 34 of 58 submissions). Any
inherited prose in this directory quoting 4.67, 2.14, 1.92 or 1.82 is stale; `board_check --live` is the
only place to read it from.
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
        "v3base_dense_s2026",
        "base_dense_ff",
        "lr4e-5_ws_ff",
        "through_dense_ff",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
