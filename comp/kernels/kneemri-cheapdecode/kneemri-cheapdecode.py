"""The CHEAPEST DECODE GROUP this project owns, served as a trio — 224 px at 12 slices, fp16, logit vote.

**THIS IS AN EFFICIENCY-TRACK INSTRUMENT AND IT MOVES ONE THING: THE PIXELS THAT GET DECODED.**
`fasttrio_v1` (0.939, 2026-09-01) already established the cheap SERVING knobs at three seats — fp16 and
LogitAverage, both flat — and this entry holds both of them. What it changes is the decode group the
three seats live in. Every entry this project has shipped decodes 336 px at 20 slices; these three read
224 px at 12 slices, which is the same six slots and the same crop over roughly a quarter of the pixels:
(12/20) x (224/336)^2 = 0.27 of the decode work, and decode is what a kernel here actually spends. A
fleet pass measured the card at 0% while pydicom worked.

**WHY THE PIXEL HALF OF THAT SAVING IS ALREADY LICENSED.** `band224_hold_s2026` finished today at 0.8516
and its 336 px control at 0.8573 on the report holdout — a 0.0057 lead for 336, under the 0.0062 floor.
The gold ruler agrees: `logs/band224_hold_ledger.log` records the pair over the 58 annotated studies as a
paired +0.0016 for 336 with a resampled interval of [-0.0146, +0.0190], inside. Two independent rulers,
two flats, one variable. So 336 -> 224 costs nothing measurable at a matched geometry. The SLICE half is
NOT licensed and is the open question this entry asks: 20 -> 12 has
never been priced at serving time, and the one adjacent number the ledger owns points the wrong way
(Raptor's `coatnet_v4_full` gains +0.0046 going 16 -> 20 slices).

**THE SEATS.** The three strongest members of that group by their own recorded holdout, chosen for
distinct TREATMENTS rather than by a search — `trioseat_v1`'s lesson was that supervision-treatment
diversity inside one decode buys what architecture distance buys, and `ceiling5_v1` read 0.934 today
against an offline maximum that a same-studies selection had inflated by +0.0201, which is the reason
this entry is NAMED and not greedy-picked:

    blendhold_s2026     0.8575  the blend treatment, and the group's strongest member
    ema224_s2026        0.8544  weight averaging
    labelwise224_s2026  0.8517  per-label heads

**PRE-REGISTERED, BEFORE THE NUMBER.** Against `fasttrio_v1`'s 0.939 at the same count, same vote, same
precision and 3.7x the decode:

- **0.933-0.939** — the decode group is nearly free, and the cheap entry is a strictly cheaper 0.936ish.
  This is the expectation: the members are 0.006 back on solo holdout and the board compresses offline
  deltas by about +1.9 (`board_transfer`), so about -0.011 offline maps to roughly one to two ticks.
- **above 0.939** — twelve slices at 224 px serve as well as twenty at 336. That would make every future
  entry cheaper for free and would re-open the serve-geometry axis this project closed on accuracy
  grounds; read it as the board's own scatter first, and confirm before spending anything on it.
- **below 0.933** — the slice count is doing real work at serving that the pixel count is not, which is a
  finding about what the slot embedding retains and a warning against reading `band224_hold`'s flat as a
  licence for decode geometry generally.

**WHAT THIS ENTRY IS NOT.** Not a candidate for the accuracy board and not a reseating: the member axis
is closed there on seven flats, the champion is `raptorweight_v1` at 0.944, and nothing here is expected
to approach it. The deliverable is a PAIR of efficiency rows at a near-held score — this against
`fasttrio_v1` — because `efficiency_read --live` at 22:06Z reads accuracy rank 41 at 0.944 and efficiency
rank 33 at 0.939, with **17 teams ranked above us on efficiency while scoring below us**, and 90 teams
within 0.002 of our score spread across efficiency ranks 13 to 2519. At a fixed accuracy that whole span
is runtime, the podium there pays 7k/6k/5k, and this project has never spent a slot on it deliberately.
The metric's formula is still UNMEASURED (the evaluation page is JS-only), so no rank is predicted here —
only that runtime, not accuracy, is what holds us at 33.
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

    import torch  # noqa: PLC0415 - the kit is on sys.path only after boot

    from kneemri.inference.ensemble import Serving  # noqa: PLC0415 - same
    from kneemri.inference.vote import LogitAverage  # noqa: PLC0415 - same
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - same

    # The cheapest decode group this project owns: 224 px, 12 slices, band 0.2-0.8. Named rather than
    # ranked, so the entry stays quotable: `solo_v4` used `members=1` and its identity is gone.
    chosen = ("blendhold_s2026", "ema224_s2026", "labelwise224_s2026")
    KernelRun(names=chosen, overlap=False, serving=Serving(vote=LogitAverage(), precision=torch.float16)).run()


if __name__ == "__main__":
    main()
