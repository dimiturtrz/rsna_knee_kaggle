"""The champion's anchor member, served ALONE — the only question left that a slot can answer cheaply.

**WHAT THIS IS FOR, AND IT IS NOT A SCORE.** `comp/board.jsonl` holds the count curve as board numbers:
one member 0.914 (`solo_v4`, 2026-08-17), four 0.924-0.928, five 0.931, six-to-seven 0.936-0.939. So
ensembling buys this project **+0.025** over its own best single. The competition forum (topic 735304)
holds five self-reported SINGLE-model scores between 0.915 and 0.942 — rivals' best single model is our
best six-member ensemble. That one comparison says the gap is MEMBER QUALITY, and every remaining hour
of card should follow it. Before spending those hours it is worth one slot to ask whether nine days of
member work moved the single-member number at all, because no offline ruler can answer it: every member
built since has been `--fold -1`, and an all-data member has no holdout left to score on.

**THE MEMBER.** `final_ws_e12` — all 5,143 studies (`--fold -1`), the dense geometry (20 slices,
0.1-0.9), EMA, twelve epochs, whole-stack with max slice pooling. It is the champion's first-listed
seat and the most-trained member this package holds. It is NOT chosen for an offline score, because it
has none; it is chosen because it strictly dominates `solo_v4`'s conditions on data and epochs.

**WHAT THIS ENTRY CANNOT DO.** `solo_v4` was served by `members=1`, which ranks inside the package
rather than naming a member, so the 2026-08-17 identity is not recoverable. The comparison is therefore
"our best single THEN" against "our best single NOW", not a matched member swap. Stated here rather
than discovered when the number arrives.

**SERVING IS THE CHAMPION'S, MINUS THE VOTE.** `overlap=False` and the same kit; `LogitAverage` over one
member is the identity, so no vote rule is passed and none is implied. The only variable against
`kneemri-blendseat` is COUNT, from six to one.

PRE-REGISTRATION, written before the number, against `solo_v4`'s 0.914.

- **0.929 or above** — member quality climbed a full house bar in nine days. The member axis is live,
  ensembling is the smaller half of our 0.939, and every hour of card goes to better members —
  `blendsix` (six blend-supervised seats) first, then better label TABLES.
- **0.920-0.928** — directional. Members improved, and ensembling still carries +0.011 to +0.019 on top.
  Both axes pay; the card keeps going to members because that is the one with room.
- **0.915-0.919** — FLAT. Nine days of member work bought nothing the board can see, and our 0.939 is
  ensembling stacked on a stalled member. The next slot does not go to another member of ours: it goes
  to reproducing the RECIPE behind a rival 0.94 single (topic 735304), because ours has stopped moving.
- **0.913 or below** — indicts `--fold -1` itself at serving. An all-data member has no holdout to
  select on, so nothing stops it drifting past its best epoch, and EMA is the only thing standing in
  for early stopping. That would implicate FIVE of the champion's six seats, and the first test is
  `convergence_check` over the all-data arms rather than another submission.
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

    # Named, not ranked. `solo_v4` used `members=1` and its identity is gone; this one is quotable.
    chosen = ("final_ws_e12",)
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
