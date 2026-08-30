"""holdswap — the champion six with its weakest seat replaced by the blind full-fit of our best solo.

**ONE MEMBER MOVES AND NOTHING ELSE DOES.** `lr4e-5_ws_cf` leaves, `calib_dense_hold_ff` enters, in the
same slot of the same tuple. Same six-then-blend shape, same `overlap=False`, same `LogitAverage`, same
`FOREIGN_WEIGHT = 1/3`, same raptor checkpoint, same kit. The champion `raptorweight_v1` scored 0.944 with
this exact tuple minus that one name, so whatever this reads is attributable to the seat.

**WHY THIS SEAT AND NOT ANOTHER.** The champion's six are all `--fold -1` full-fits and therefore carry no
holdout record — none of them can be priced offline. Their fold-0 analogues can. `record_mix --roster` held
the five that HAVE analogues (`calib_span`, `grade_ladder`, `merged_blend`, `lr4e-5_ws`, `through_dense`)
and priced ten strong pool members against them. Ten of ten named the SAME incumbent as the one to
displace: `lr4e-5_ws`, the weakest of the five at 0.8525 solo. Five of the ten cleared `BOARD_RESOLVABLE
= +0.0015` on that swap. Unanimity across ten candidates is a SIGN result, and the sign is the only thing a
member swap has ever been shown to transfer (`knee-7d1`; the ordering broke on its third point). An earlier
and independent run recorded the same seat on the roster line for `finding_calib_dense` — `+0.0017 as a
swap for lr4e-5_ws on the champion shadow` — so this is a reproduction, not one draw.

**WHY `calib_dense_hold_ff` IS THE REPLACEMENT.** The ordering does NOT transfer, so the top row of that
table (`labelwise_dense`, +0.0024) is not what picks the entrant. What picks it is the intersection of two
constraints. First, the entrant must be BLIND — a fold-0 member on a champion-shaped entry is the mistake
`greedyseat` already paid for, scoring 0.936 against this same champion's 0.944, because a member fitted on
a fold gives up about eight thousandths on the board. Second, it must be in the group the proxy could not
separate. `calib_dense_hold_ff` is the only member satisfying both: it is `--fold -1 --ema
--calibrated-targets`, its fold-0 analogue `calib_dense_hold_s2026` is the BEST SOLO MEMBER THIS PROJECT
OWNS at 0.8634, and that analogue read +0.0013 marginal and +0.0018 as the swap, inside the band of the
five that cleared the bar. It has never been seated on the board in 81 recorded submissions.

**WHAT THIS IS NOT.** It is not a diversity entry. `pool_census` (2026-08-30) measured all 117 members and
every one of the fourteen strong sits at nearest agreement 0.9011 or above; `calib_dense_hold` is inside
that band and has a measured twin (`recal_dense`, 0.9496 agreement, 0.0065 bits divergence). The far-lane
seat run the same day priced eight distant members against a champion-shaped roster and the deliberately
included TWIN CONTROL beat all eight, `0 of 9 clear +0.0015`. So the far lane is not what this spends the
slot on. This entry buys member QUALITY in the one seat the proxy says is weakest, which is the same
mechanism the champion's own history is made of: `res392_ws` out, `band_dense` in, 0.925 -> 0.928.

**THE HONEST LIMIT.** The proxy is fold-0 and the entry is blind; five of six seats are analogues rather
than the members themselves, and there is no fold-0 `final_ws` at all, so the sixth was never in the
comparison. A swap that reads positive in proxy space can read flat on the board, and `farseat_v1` is the
standing warning that a board number can come from an entry that was never assembled — `run_check --log`
runs BETWEEN the kernel completing and this slot being counted, for that reason.

PRE-REGISTERED, before the number, against the champion's 0.944 and `BOARD_RESOLVABLE = 0.0015`.

- **s >= 0.946** — the swap is real AND it reaches the top-ten floor. Re-pin `comp/champion.json`, and the
  next slot takes the SECOND seat the same table names, since the mechanism would then be shown to carry.
- **0.946 > s >= 0.9425** — a real gain over the champion, above the resolution floor. The fold-0 proxy
  ORDERS the blind seats, which is a new and reusable instrument: it means the champion's remaining five
  seats can be ranked offline without spending a slot each. Re-pin, then price the next-weakest seat.
- **0.9425 > s >= 0.9415** — FLAT. The swap is inside the board's own resolution. Then the fold-0 proxy
  does NOT order blind seats — it separates members it cannot separate on the board — and the correct
  reading is that the champion six are interchangeable at this quality level. No further slot goes to a
  single-seat swap; the remaining levers are corpus, decode and trunk.
- **< 0.9415** — the swap COSTS. `lr4e-5_ws_cf` was carrying something the proxy cannot see, most likely
  decorrelation rather than quality (it is the weakest of the five and yet the champion kept it through
  four entries). Suspect the proxy's DENOMINATOR first: it priced over 909 shared studies in fold-0 space,
  and the board's test set is hidden and not ours. Keep the champion pinned, and do not re-run this axis
  without a blind-space ruler.
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

FOREIGN_WEIGHT = 1.0 / 3.0
"""The foreign side's share of the rank blend. `raptorseat_v1` shipped 1/7 = 0.1429 as a convention about
SEAT COUNT; this is a quality-weighted share of two SYSTEMS, named before the number and never fitted.

Below the bootstrap block rather than above it: the block opens with imports, and a constant in front of
them makes every one of them E402. The kernel that RAN carried it above and the ruff gate caught it after
the fact — the number is identical either way.
"""


def main() -> None:
    boot()

    import time  # noqa: PLC0415 - the bootstrap block above is byte-identical across kernels

    import pandas as pd  # noqa: PLC0415 - the kit is on sys.path only after boot
    import torch  # noqa: PLC0415 - same

    from core.mounts import Mounts  # noqa: PLC0415 - same
    from kneemri.device import VisibleCards  # noqa: PLC0415 - same
    from kneemri.evaluation.raptor_model import RaptorMember  # noqa: PLC0415 - same
    from kneemri.inference.budget import KERNEL_SECONDS, Budget  # noqa: PLC0415 - same
    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - same
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same
    from kneemri.kaggle.raptor_run import WORKING, RaptorEntry  # noqa: PLC0415 - same
    from kneemri.kaggle.raptor_seat import RaptorSeat  # noqa: PLC0415 - same

    # The raptor member runs FIRST and into a scratch directory, because both producers write a file
    # called `submission.csv` and the one that runs last is the one Kaggle would otherwise score.
    scratch = WORKING / "raptor"
    scratch.mkdir(parents=True, exist_ok=True)
    started = time.time()
    mounts = Mounts()
    device = str(VisibleCards().device())
    member = RaptorMember.loaded(RaptorEntry.weights(mounts=mounts), device)
    theirs = RaptorEntry(mounts.one("sample_submission.csv", "competition data"), working=scratch).run(member, device)
    # The six that follow load onto the same T4. Dropping the reference is not enough on its own —
    # torch keeps the freed blocks in its caching allocator, and a seventh model's worth of them is
    # exactly the margin an OOM would eat, in a kernel whose whole point is running both.
    del member
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    chosen = (
        "final_ws_cf",
        "calib_span_ff",
        "grade_ladder_cf",
        "merged_blend_cf",
        "calib_dense_hold_ff",
        "through_dense_cf",
    )
    # What the raptor pass cost, handed to the six so their nine-hour budget is the SESSION's and not
    # a fresh one. Without it both producers aim at the same ceiling and the kernel is killed at the
    # real one with no submission — a budget defeated by its own assumption that it starts at zero.
    spent = time.time() - started
    logging.getLogger(__name__).info(f"the raptor pass cost {spent:.0f}s; the six get what is left")
    left = Budget(seconds=max(KERNEL_SECONDS - spent, 0.0))
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), budget=left)).run()
    ours = pd.read_csv(WORKING / "submission.csv")

    RaptorSeat(weight=FOREIGN_WEIGHT).blended(ours, theirs).to_csv(WORKING / "submission.csv", index=False)


if __name__ == "__main__":
    main()
