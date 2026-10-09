"""mirrortilt — mirrorseat with our seven's share set PER FINDING around 1/2 against nartaa's CoAtNet alone.

Same seven, same foreign seat, same serving as `kneemri-mirrorseat` (0.953, ref 57008651). The level stays
at mirrorseat's 1/2; only the per-finding PROFILE moves, by one step either side. Two holdout instruments
(2026-09 and `interpretations/kneemri/2026-10-09_no-stratum-flips-the-blend-share.md`, ledger75, 896
studies) agree that ours leads raptor-recipe replicas by >= 0.03 on ACL, MCL, Lateral Meniscus, PF OA and
Synovitis, ties on Effusion, Baker's, Contusion and Lateral OA, and loses only Fracture. Only the SIGN of a
per-finding holdout delta has ever transferred, so the tilt is one step (1/2 -> 2/3 or 1/3), never the 0.9
the holdout's own optimum asks for. Read it AFTER the mirrortwo/mirrorthird bracket: if the level moves,
re-centre this profile on the winning level before spending a slot on it.

PRE-REGISTRATION (mirrorseat 0.953, board resolution 0.001, top-band residual sd ~0.002):
- **>= 0.954** — the per-finding profile transfers against the real nartaa arm; tilt is the new floor.
- **0.952-0.953** — flat; the profile is below board resolution. Keep mirrorseat.
- **<= 0.951** — the replica-based profile does not hold against nartaa; per-finding shares are closed here.
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


HALF, TWO_THIRDS, THIRD = 1.0 / 2.0, 2.0 / 3.0, 1.0 / 3.0
OURS = {
    "ACL": TWO_THIRDS,
    "MCL": TWO_THIRDS,
    "Lateral Meniscus": TWO_THIRDS,
    "PF OA": TWO_THIRDS,
    "Synovitis": TWO_THIRDS,
    "Medial Meniscus": HALF,
    "Medial OA": HALF,
    "Lateral OA": HALF,
    "Effusion": HALF,
    "Baker's": HALF,
    "Contusion": HALF,
    "Fracture": THIRD,
}
"""Our seven's share of the final blend, per finding. See the module docstring."""


def main() -> None:
    boot()

    import shutil  # noqa: PLC0415
    import time  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    from core.findings import TARGETS  # noqa: PLC0415 - the kit is on sys.path only after boot
    from core.mounts import Mounts  # noqa: PLC0415
    from kneemri.inference.budget import KERNEL_SECONDS, Budget  # noqa: PLC0415
    from kneemri.inference.ensemble import Serving  # noqa: PLC0415
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415
    from kneemri.kaggle.raptor_seat import RaptorSeat  # noqa: PLC0415

    chosen = (
        "final_ws_cf",
        "calib_span_ff",
        "stack_increments_ff",
        "through_dense_cf",
        "labelwise_depth24_drop40_long_s2026",
        "labelwise_ext_drop40_long_s2026",
        "labelwise_dense_drop40_long_s2026",
    )
    if set(OURS) != set(TARGETS):
        message = f"shares name {sorted(OURS)}, the board scores {sorted(TARGETS)}"
        raise ValueError(message)
    working = Path("/kaggle/working")
    started = time.time()
    mounts = Mounts()
    script = mounts.one("nartaa_mirror.py", "nartaa's scoring source") / "nartaa_mirror.py"
    subprocess.run([sys.executable, str(script)], check=True, cwd=working)  # noqa: S603 - our mounted copy, fixed argv
    foreign = working / "nartaa_seat.csv"
    shutil.move(working / "submission.csv", foreign)
    spent = time.time() - started
    logging.getLogger(__name__).info("nartaa's pass cost %.0fs", spent)

    KernelRun(
        names=chosen,
        overlap=False,
        serving=Serving(vote=LogitAverage(), budget=Budget(seconds=max(KERNEL_SECONDS - spent, 0.0))),
        mounts=mounts,
    ).run()
    ours = pd.read_csv(working / "submission.csv")
    theirs = pd.read_csv(foreign, dtype={"StudyInstanceUID": str})
    ours["StudyInstanceUID"] = ours["StudyInstanceUID"].astype(str)
    blends = {share: RaptorSeat(weight=1.0 - share).blended(ours, theirs) for share in set(OURS.values())}
    tilted = blends[HALF].copy()
    for finding, share in OURS.items():
        tilted[finding] = blends[share][finding]
    logging.getLogger(__name__).info("per-finding shares %s", OURS)
    tilted.to_csv(working / "submission.csv", index=False)


if __name__ == "__main__":
    main()
