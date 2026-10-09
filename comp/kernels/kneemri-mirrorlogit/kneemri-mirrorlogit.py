"""mirrorlogit — mirrorseat's two halves at 1/2, averaged in LOGIT space instead of rank space.

Same seven, same foreign seat, same share as `kneemri-mirrorseat` (0.953, ref 57008651); only the space
the two tables are averaged in moves. `RaptorSeat` ranks both sides because their calibrations are
unrelated, and that argument stands: a logit average lets the more confident head decide each study. The
holdout says that confidence is informative — on 896 studies (ledger75, paired bootstrap, two honest
replicas of nartaa's recipe as the foreign side) the logit average at 1/2 beats the rank average at 1/2 by
+0.0040 / +0.0041, CI [+0.0026, +0.0057], every resample positive, and the geometric mean of probabilities
agrees (+0.0045). The replicas are weaker than the real checkpoint and their confidence need not match
it, so the size will not transfer; this asks the board whether the sign does. A z-scored logit average
(scale-free like rank) gains only +0.0009, so the gain is the cross-head confidence, not the tails.

THE TAP. nartaa's notebook writes per-column percentile RANKS (`rankpct` of `arm_probs`, cell 4), so his
`submission.csv` carries no confidence at all. The mounted source is patched at run time — one anchored
insertion after his own write, refused unless the anchor occurs exactly once — to also write the
pre-rank, mirror-averaged probabilities to `nartaa_probs.csv`, and those are checked to reproduce his
ranks before they are blended. His mounted file is never modified.

PRE-REGISTRATION (mirrorseat 0.953, board resolution 0.001, top-band residual sd ~0.002):
- **>= 0.955** — confidence carries across heads; logit becomes the blend space and replaces mirrorseat.
- **0.953-0.954** — inside scatter; rank stays (it is the space every other number was built on).
- **<= 0.952** — the real checkpoint's confidence is mis-scaled against ours; rank space was right.

The foreign seat is `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` (public dataset
`nartaa/rsna-knee-publication-swa-weights-20261007`, sha256 7e5315da...), served by nartaa's OWN scoring
code: the code cell of `nartaa/rsna-knee-0949-anatomical-mirror`, whose sha256 bcd864cb... matches the
source hash that notebook publishes. It runs verbatim in a child process (so its two-T4 thread pool and
CUDA memory are gone before ours load), mounted from the private `dimiturnt/nartaa-mirror-src`. Fetch:
`kaggle kernels pull nartaa/rsna-knee-0949-anatomical-mirror`, join the code cells. Never vendored here.

It REPLACES `raptor_v4_swa` rather than joining it: nartaa's arm is a retrain of that recipe (96 slices,
94 windows, OAI-supervised PF/Lateral OA and Synovitis, anatomical mirror TTA), so v4 would be a second,
weaker copy of the same family.

WHY 1/2. Two systems, no ruler to fit on: nartaa trained on all studies, so our holdout is in-sample to
it, and gold-58 is 58 rows. Our seven alone read 0.945 on the board, theirs 0.949 — a gap of four ticks,
small next to the 0.021 that put raptor_v4 at 1/3. Equal shares of two near-equal systems, named before
the number.
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

    import re  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    import time  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    from core.findings import STUDY, TARGETS  # noqa: PLC0415
    from core.mounts import Mounts  # noqa: PLC0415 - the kit is on sys.path only after boot
    from kneemri.inference.budget import KERNEL_SECONDS, Budget  # noqa: PLC0415
    from kneemri.inference.ensemble import Serving  # noqa: PLC0415
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415

    chosen = (
        "final_ws_cf",
        "calib_span_ff",
        "stack_increments_ff",
        "through_dense_cf",
        "labelwise_depth24_drop40_long_s2026",
        "labelwise_ext_drop40_long_s2026",
        "labelwise_dense_drop40_long_s2026",
    )
    working = Path("/kaggle/working")
    started = time.time()
    mounts = Mounts()
    source = (mounts.one("nartaa_mirror.py", "nartaa's scoring source") / "nartaa_mirror.py").read_text()
    anchor = re.compile(r"^([ \t]*)sub\.to_csv\(out, index=False\)$", re.MULTILINE)
    found = anchor.findall(source)
    if len(found) != 1:
        message = f"nartaa's write line appears {len(found)} times; the probability tap needs exactly one"
        raise ValueError(message)
    tap = "\n".join(
        found[0] + line
        for line in (
            "sub.to_csv(out, index=False)",
            "_tap = (_w[:, None, None] * np.stack(arm_probs)).sum(0)",
            "_tap[~np.isfinite(_tap)] = 0.5",
            "_tap = pd.DataFrame(_tap.astype(np.float32), columns=LAB)",
            '_tap.insert(0, "StudyInstanceUID", test_ids)',
            '_tap[sub_cols].to_csv("/kaggle/working/nartaa_probs.csv", index=False)',
        )
    )
    script = working / "nartaa_mirror_tapped.py"
    script.write_text(anchor.sub(lambda _: tap, source))
    subprocess.run([sys.executable, str(script)], check=True, cwd=working)  # noqa: S603 - our mounted copy, fixed argv
    shutil.move(working / "submission.csv", working / "nartaa_seat.csv")
    foreign = working / "nartaa_probs.csv"
    probs = pd.read_csv(foreign)[list(TARGETS)].to_numpy()
    seat = pd.read_csv(working / "nartaa_seat.csv")[list(TARGETS)].to_numpy()
    agreement = min(
        np.corrcoef(probs[:, j].argsort().argsort(), seat[:, j].argsort().argsort())[0, 1] for j in range(len(TARGETS))
    )
    logging.getLogger(__name__).info("tapped probabilities vs nartaa's ranks: min per-label rank corr %.6f", agreement)
    if len(probs) > 2 and agreement < 0.999:  # noqa: PLR2004 - three rows is the public test, a pair has no order
        message = f"tapped probabilities do not reproduce nartaa's ranks (min corr {agreement:.6f})"
        raise ValueError(message)
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
    aligned = theirs.set_index(STUDY).reindex(ours[STUDY])[list(TARGETS)]
    if aligned.isna().to_numpy().any():
        message = f"{int(aligned.isna().any(axis=1).sum())} of {len(ours)} studies are absent from nartaa's table"
        raise ValueError(message)
    sides = [
        np.clip(ours[list(TARGETS)].to_numpy(dtype=np.float64), 1e-6, 1 - 1e-6),
        np.clip(aligned.to_numpy(dtype=np.float64), 1e-6, 1 - 1e-6),
    ]
    mean_logit = sum(0.5 * np.log(side / (1 - side)) for side in sides)
    blended = pd.DataFrame(1 / (1 + np.exp(-mean_logit)), columns=list(TARGETS))
    blended.insert(0, STUDY, ours[STUDY].to_numpy())
    blended.to_csv(working / "submission.csv", index=False)


if __name__ == "__main__":
    main()
