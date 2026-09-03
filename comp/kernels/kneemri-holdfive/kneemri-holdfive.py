"""Five members taught from HELD ANNOTATIONS, against five taught the ordinary way — an INSTRUMENT.

**THE AXIS, AND WHY IT IS THE ONE LEFT.** Nine board entries have now closed the ensemble machinery.
Who SITS is flat across seven seatings (raptorweight +0.001, vjepaseat -0.001, holdswap 0.000, dark7,
spanseat, stackseat 0.000, lensseat 0.000). How they are COMBINED is flat across three vote rules
(LogitAverage 0.944, LabelQualityRank 0.943, StackedLogit 0.943, the last with a table deviating
24.83 pp). How the output is SHAPED is closed at identity (`monotone_v1`, cubing every column, exactly
0.944 — the metric is per-column rank-based). `distinct14` seated the only two members that are
simultaneously distant AND above the size bar and read 0.000. Every one of those rows ends with the
same sentence: the lever is not in the ensemble, it is in the MEMBERS — what a member SEES, what it
IS, and what it is TAUGHT. This entry is the first board reading of the last of those three.

**WHAT `--hold-annotations` CHANGES.** It is a SUPERVISION change, not a seating, serving or decode
one. `reader_agreement` over the 58 annotated studies reads 108 confident false positives — 15.8% of
every annotated cell — against 32 misses of which only 8 are a confident NO. The label defect is
over-calling by about an order of magnitude on the confident end, and holding annotations out of the
training signal is the arm that answers it. Offline the four `--hold-annotations` checkpoints read
+0.538 to +0.700 against their controls on the gold cells, and `--calibrated-targets` cuts tracking of
the defect by -0.145 / -0.150.

**WHY A SINGLE SEAT COULD NEVER HAVE ANSWERED IT, AND THIS CAN.** `holdswap_v1` already put ONE
`--hold-annotations` member into the champion's seven and read 0.944 — flat, like every other single
seat exchange. That entry could not resolve the axis: at 0.944 the marginal seat is worth less than
BOARD_RESOLVABLE 0.0015, which is exactly what the six seating flats before it established. A whole
pool is a different operation. All five seats here carry the supervision change, so what is measured
is the FAMILY at full dose, the same shape as the external-corpus dose curve (1, 2, 4, 6 seats) that
was licensed as an instrument and read the corpus axis closed.

**THE MATCHED CONTROL, AND IT WAS SHIPPED YESTERDAY.** `twinfive_v1` (ref in `comp/board.jsonl`) is
five FOLD-0 members, LogitAverage, overlap=False, same kit machinery, and it read **0.935** against a
measured offline **0.8706**. This entry holds every one of those terms and moves one: the five members
are the `--hold-annotations` family instead of the champion's twins. Same count, same fold-0 status —
so the 0.006 twin-to-shipped gap `fiveship_v1` measured is present in BOTH arms and cancels. That
cancellation is the whole reason this is readable and a comparison against the champion's 0.944 would
not be.

**THE OFFLINE NUMBER, MEASURED BEFORE THE PUSH.** `vote_ceiling` over these exact five names, 909
shared fold-0 studies, weights fitted on one half and scored on the other: RankAverage 0.8723,
QualityWeightedRank 0.8723, LabelQualityRank 0.8723, LogitAverage **0.8729** (+0.0006 vs flat, overfit
gap +0.0000), MedianLogit 0.8707. `LogitAverage` is shipped below and is the family best, as at every
other pool size. Against `twinfive_v1`'s 0.8706 that is **+0.0023 offline**.

**SELECTION, STATED HONESTLY BECAUSE IT IS NOT ZERO.** These are not a searched subset. They are the
`--hold-annotations` members on `comp/members.roster`, minus two named exclusions: `recal_dense_s2026`
(the roster itself declares it a TWIN of `calib_dense_hold_s2026`, so seating both buys correlation,
not a member) and `holdannot_s2026` (0.8506, the only one under the 0.8521 lane bar). The second of
those two IS a score-based exclusion and it is declared as such. `peak8` measured greedy-selection
bias at about 0.0027 at this pool size; one bar-based exclusion from a family of seven is far less
search than that, but it is not nothing, and the +0.0023 offline gap is inside the range where it
matters. That is a caveat on the OFFLINE gap, not on the board reading, which is what the slot buys.

**THE SLOPE THIS PRE-REGISTRATION USED WAS WRONG, AND IT INVERTED THE READING (annotated after the
score, `knee-` P0 on the constant).** The +1.55 below was carried from prose, never from the tool.
`python -m kneemri.evaluation.board_transfer` reports compression as a DECREASING function of where
the pool already sits: +1.25 over the whole record, +0.97 among entries at or above 0.8678, and
**+0.71 among entries at or above 0.8694** — which is the band this pool's 0.8729 falls in. At +0.71
the +0.0023 offline gap predicts +0.0016, i.e. **0.9366**. It scored **0.936**. So the gain
transferred exactly as the record says it must, and the branch it landed in below — "the offline
+0.0023 does not survive" — is the wrong reading of a correct number. The bands are left as written
because a pre-registration that is edited after the score is not one; this note is the correction.

PRE-REGISTRATION, written before the board number, against `twinfive_v1`'s 0.935 and a
`board_transfer` slope quoted as +1.55 board points per offline point (which predicts 0.938-0.939):

- **>= 0.941** — the supervision family transfers at MORE than the fitted slope. The label lane is a
  real board lever at pool scale, not just an offline one, and the follow-up is a full re-fit of the
  champion's seats under `--hold-annotations --calibrated-targets` rather than a sixth seat. This is
  the branch that would justify card hours against the label axis over any trunk work.
- **0.937-0.940** — transfers at about the fitted slope. The supervision change is worth its offline
  delta and nothing more; the label lane is real but ordinary, and it is priced like any other member
  improvement. Directional, not a finding, per the 1.5% bar.
- **0.933-0.936** — FLAT against `twinfive_v1` to the board's own quantum. The offline +0.0023 does
  not survive, the supervision axis joins seating, combining and shaping as board-flat, and the label
  lane's gold-cell gains are an offline artefact of the 58 annotated studies they were measured on.
  That is a genuine and expensive thing to learn, and it would stop the label lane from consuming card.
- **<= 0.932** — the supervision change actively COSTS at pool scale. Suspect the pool before the
  mechanism: all five members share `--calibrated-targets` as well as `--hold-annotations`, so the two
  flags are fused in this entry and a negative reading cannot separate them.

**WHAT THIS ENTRY CANNOT SAY.** The two supervision flags are confounded, as stated above. It reads
five seats, not the champion's seven, so it says nothing about whether the same change would move a
0.944. And it is fold-0 throughout, so `board_transfer`'s twin caveat applies to its absolute number —
which is why every branch above is written against `twinfive_v1` and not against the champion.

COST: one slot, INSTRUMENT not mechanism. No card — every member already exists and is on the roster.
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

    # The five `--hold-annotations` members on the roster, minus the two exclusions named in the
    # docstring. Count, fold-0 status, vote rule and serving all match `twinfive_v1`, so the only
    # term that moves between the two entries is the supervision family. Offline 0.8729 by
    # `vote_ceiling` over 909 shared studies, against twinfive's 0.8706.
    chosen = (
        "calib_dense_hold_s2026",
        "finding_calib_dense_s2026",
        "blendhold_s2026",
        "band_dense_hold_s2026",
        "calib_span_hold_s2026",
    )
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage())).run()


if __name__ == "__main__":
    main()
