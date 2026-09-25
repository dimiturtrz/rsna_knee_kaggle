"""proxyseat — the champion, plus the first three seats this project has ever PRICED before shipping.

**WHY THIS IS AN INSTRUMENT AND NOT A RESEATING.** Eight board entries have varied who sits on the
champion's six and how much each seat counts — `vjepaseat_v1` 0.943, `relabel7_v1` 0.943, `resseat_v1`
0.938, `blendthree_v1` 0.939, `calibsix_v1` 0.941, `monotone_v1` 0.944 — and every one of them read flat
or down against 0.944. All eight chose their seats BLIND: no offline number existed for the tuple,
because all six champion seats are full-fit (`--fold -1`) and hold nothing out, so no ruler on this
project can read the ensemble they form. That is the mechanical reason the axis reads as eight nulls.

**THE RULER THAT DID NOT EXIST.** Five of the champion's six seats have fold-held ancestors. Blended,
those five read **0.8700** offline and predict **0.933** against the champion's actual 0.944, so the
proxy's LEVEL is 0.011 low and is useless. Its DIFFERENCES are the quantity being tested. Nested onto
that proxy by rule — no search, no ranking, no subset selection — the three members added here read:

    proxy alone                                         0.8700
    + chaindense_s2026                                  0.8719   (+0.0019)
    + blendemalabel224_s2026                            0.8716   (+0.0016)
    + blendema224_s2026                                 0.8710   (+0.0010)
    + all three                                         0.8727   (+0.0027, 60% of the sum)

At the top-regime transfer constant read live from the board record — **+0.98 board points per holdout
point, [+0.94, +1.08], 24 resolved pairs at or above 0.8678** — +0.0027 offline is **+0.0026 board**.

**THE QUESTION.** Does a marginal contribution measured on held twins transfer to the full-fit originals?
Nothing on this board answers it. If it does, the seat axis re-opens with a ruler that costs zero slots,
and the eight flats are explained as blind picks rather than as a closed axis. If it does not, the eight
flats gain a single mechanism — offline marginals do not price full-fit seats — and the axis closes on
an argument instead of on an accumulation of nulls. Both branches are worth more than the 0.0026.

**THE CONFOUND, STATED BEFORE THE NUMBER.** The three added members are fold-HELD and the six they join
are full-fit. Full-fit dose is worth about +0.007 board on this record (`honest12` 0.937 -> champion
0.944), so three held seats diluted into a full-fit six can lose on dose alone. A flat reading therefore
does NOT cleanly refute transfer; it is consistent with transfer working and dose eating it. The
follow-up that separates them is re-fitting these three at `--fold -1` and re-running this exact tuple,
which is three arms of card and is only worth spending if this reads flat.

PRE-REGISTRATION, written before the number, against the champion's 0.944 and BOARD_RESOLVABLE = 0.0015.

- **0.9465 or above** — transfer HOLDS and the predicted +0.0026 landed. The held-twin proxy prices seats
  on the champion shape. Re-pin `comp/champion.json`, and every future seat decision is made offline at
  zero slot cost before a slot is spent. This is the single most valuable outcome available tonight.
- **0.944 <= s < 0.9465** — a gain at or under the resolution floor. Directional only. It licenses ONE
  more priced tuple, not a re-pin.
- **0.9425 <= s < 0.944** — FLAT, the ninth. Under the confound above this is ambiguous between "offline
  marginals do not transfer" and "held members dilute a full-fit six". The next slot on this axis has to
  be the full-fit re-fit, or the axis stays ambiguous forever.
- **below 0.9425** — the three DILUTE by more than their offline marginal predicts in any direction. That
  is the dose reading, not the transfer reading, and it prices full-fit dose a second time — the +0.007
  constant currently rests on one confounded comparison and this would be its second point.

The refutation this cannot make: it cannot say WHICH of the three carried or cost, because they enter
together. They enter together on purpose — priced individually they are +0.0019/+0.0016/+0.0010, all
under any board resolution, and three slots to read three numbers that small is not a trade worth making.
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


CHOSEN = (
    # raptorweight_v1's six, in its order. Not one of them moves.
    "final_ws_cf",
    "calib_span_ff",
    "grade_ladder_cf",
    "merged_blend_cf",
    "lr4e-5_ws_cf",
    "through_dense_cf",
    # The three the held-twin proxy priced at +0.0027 offline, together, nested by rule.
    "chaindense_s2026",
    "blendemalabel224_s2026",
    "blendema224_s2026",
)
"""The nine seats, written as ONE literal tuple on purpose.

Splitting it into `CHAMPION + PRICED` reads better and blinds `submit_check`, which resolves a kernel's
membership statically and reports UNKNOWN — "I did not look" — the moment the tuple is computed. A check
that cannot see the set cannot say the board already holds it, and that is the hole a slot fell through
once already. The added three sit in the marginal order the proxy measured, which on these three happens
to match their solo order — a fact about these three and not a rule this project has found to hold."""


def main() -> None:
    boot()

    from kneemri.kaggle.seated_entry import SeatedEntry  # noqa: PLC0415 - the kit is on sys.path only after boot

    chosen = CHOSEN
    SeatedEntry(
        names=chosen,
        # The champion's foreign share, unchanged. This entry moves seats and nothing else.
        weight=1.0 / 3.0,
    ).run()


if __name__ == "__main__":
    main()
