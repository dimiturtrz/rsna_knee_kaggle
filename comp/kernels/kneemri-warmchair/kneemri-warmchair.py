"""warmchair — the champion, two seats re-occupied: one priced, one FORCED by the rule-2.5 licence gate.

**WHAT IS BEING PRICED.** Two members, two chairs, everything else the champion's. The vote, the count, the
serving config, the kit, the foreign seat and its 1/3 weight are `raptorweight_v1`'s untouched. Move 1 is
the priced one: who occupies the seat the champion's own leave-one-out says is its weakest. Move 2 is not a
choice — see the licence section below — and its occupant was picked by the same ruler, pre-specified.

**THE MEASUREMENT.** `seat_swap` on the 896-row fold-0 frame, `LedgerSeventyFive`, against d16's champion
proxy (0.9237): `warm_calib_lowlr_s2026` for `lr4e-5_ws` reads **+0.0014, CI [+0.0006, +0.0022], resolved**,
corr-to-rest 0.899 against the parent's 0.925, verdict GO. Reproduced here from the researcher's D153 to
four decimals. `lr4e-5_ws` is additionally the ONLY champion seat whose removal HELPS on that ruler
(+0.0003 leave-one-out, against -0.0005 to -0.0013 for the other four), so vacating the chair is
member-independent and the occupant argument only decides how much is gained.

**WHY THIS MEMBER AND NOT THE ONE THE MEASUREMENT WAS ORDERED FOR.** The composition ruled in C159 also
seated `raptor_ft_repl2x2_c_swa` as an eighth. That member **has no weights** — only its holdout record
survives on disk — so it cannot be shipped by anything. Its 0.99-corr family twin `raptor_ft_repl2x2_a_swa`
does exist and is packaged, and prices identically in this chair (+0.0004 against c_swa's +0.0003), but it
is the SAME lineage and the same trunk (`coatnet_rmlp_2_rw_384`) as the foreign seat this entry already
carries — which is precisely the second-seat-of-one-family case D147 priced at +0.0004-0.0007 and
`comp/members.declines` already refused. So the eighth seat is not taken, and the count stays at six.

**MOVE 2 IS FORCED, NOT CHOSEN — AND IT CONVICTS THE PIN.** `submit_check` refuses any entry seating
`merged_blend_cf`: it trains on the rijeka table, whose terms are unresolved against competition rule 2.5,
which obliges the winner to license the submission AND its source. `--anyway` does not open that gate. The
CHAMPION ITSELF seats it, so `raptorweight_v1` carries the same exposure and `champion_check` says in
writing that *a clean entry that ties it replaces it*. This is therefore not an optional improvement: every
future entry must vacate that chair, and this is the first one that does. Ten roster lines carry the tag
(nine rijeka, one mrnet) — the whole `merged_*`/`blend_*` family is unshippable.

The occupant was picked by the same `seat_swap` ruler over the untagged roster, seven candidates priced,
each on its own bootstrap: `labelwise_dense_s2026` **+0.0012**, corr-to-rest 0.912 against the parent's
0.919, verdict GO — tied on delta with `warm_calib_dense_s2026` (+0.0012, corr 0.902) and taken on solo
(0.8959 vs 0.8917, the third-best solo the project owns). The other five read +0.0009 or below and fail the
gate. Seven priced and the best taken is a small argmax, so **the +0.0012 is not claimed as resolved the
way move 1's +0.0014 is** — it is the price of a chair that had to be vacated regardless.

**ADDITIVITY IS NOT CLAIMED.** Run composed, the two moves still read +0.0014 and +0.0012 against the
unswapped proxy — that is two independent readings, not a measured joint delta. The pre-registration below
is written against move 1 alone; if the moves compose, the entry lands above its band.

**THE CONFOUND, NAMED BEFORE THE NUMBER.** The chair being vacated holds `lr4e-5_ws_cf`, fitted at
`--fold -1` on all 4,406 studies; `warm_calib_lowlr_s2026` is a fold-0 member fitted on 3,497. So the swap
crosses the all-data line and gives back that seat's share of the all-data dose (~+0.0006/seat, sub-linear,
`interpretations/kneemri/2026-08-24_the_all_data_dose_response.md`). The offline +0.0014 was measured
between fold-0 twins on both sides and cannot see it. `lastseat_v1` against `dark7_v1` read that same
crossing FLAT (0.937 both ways) on one trial, which is the whole of the evidence that it is affordable.

**PROVENANCE OF THE ORDERING.** The +0.0014 is measured against d16's PROXY of the champion — five
ours-members rank-mean at 2/3 plus the foreign anchor at 1/3, with the fold-0 twins standing in for the
shipped `_cf`/`_ff` members. The shipped entry's own six are all `--fold -1` and cannot be seat-scored at
all, so no ruler exists that reads the entry itself. The ordering therefore comes off the proxy, and this
line is here so that a later reader does not attribute it to the entry.

PRE-REGISTRATION, written before the number, against the champion's 0.944 and `BOARD_RESOLVABLE = 0.0015`.
Offline +0.0014 through the REFITTED transfer constant prices at **+0.0014 to +0.0018 board**. D155 s2-s3
(2026-09-08) refits `board_transfer` on the rebuilt L75mid column at **+1.135, CI [+1.02, +1.25]**, n=36,
R2 0.962, resid SD 0.0016, against the mixed column's +1.31 [+1.10, +1.94] — and shows the gap is
POPULATION, not frame: the same mixed column reads +1.073 once the 8 early low-board rows that carry no
member list are dropped. The old constant overstates by ~15%. Bands below are written on +1.135.

- **0.947 or above** — the seat ruler UNDER-priced the move; the transfer constant is too low at the top of
  the board, which is where C160 s2(c) already suspects it is fitted badly. Re-pin, and refit the constant.
- **0.945-0.946** — as predicted. The +0.0014 seat delta survives the board, one to two ticks, DIRECTIONAL
  and not a result. Re-pin, and the seat ruler earns one more forward test.
- **0.944** — FLAT. The move is inside the board's resolution and the twelfth reading of the same plateau.
  It does NOT refute the seat ruler on its own: +0.0018 predicted against a 0.001 quantisation is a coin.
- **0.943 or below** — a LOSS, and the named suspect is the confound above rather than the member: the
  all-data seat was worth more than the +0.0014 the fold-0 ruler could see. That would be the first
  measurement of the crossing cost at a resolvable size, and it prices every future fold-0 seat.

INSTRUMENT OR MECHANISM: **mechanism**, and a small one. The mechanism is the seat ruler itself — an
offline seat delta, resolved on its own bootstrap, pre-specified and not a search maximum (the ~500-
arrangement argmax reads +0.0016 and carries a measured +0.0033 selection overfit; it is not taken).
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
        "labelwise_dense_s2026",
        "warm_calib_lowlr_s2026",
        "through_dense_cf",
    )
    SeatedEntry(
        names=chosen,
        # A quality-weighted share of two SYSTEMS, not `raptorseat_v1`'s 1/7 seat-count convention.
        weight=1.0 / 3.0,
    ).run()


if __name__ == "__main__":
    main()
