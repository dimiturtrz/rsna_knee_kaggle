"""The seventh seat filled by the FURTHEST member this project has ever produced, on a 3D encoder.

**THE augseat GATE ABOVE THIS LINE IS VOID, and `dark6` is what voided it.** The original gate said: if
augseat reads 0.936-0.939, "type of distance does not carry at seven seats," so do not spend a slot here.
augseat read 0.936 and this kernel was correctly left unsubmitted on 2026-08-25 01:14Z. Then `dark6`
(55756018) read **0.937** — six `--fold -1` members and NOTHING ELSE, tying `allfit` and `lastseat`, which
are those same six plus a seventh. The seventh seat is worth ZERO. So augseat's 0.936 was never a
measurement of "type of distance at seven seats"; it was a measurement of an INERT COORDINATE, and the
premise the gate reasoned from does not exist. A gate whose input turned out to be noise around a null
cannot refuse anything.

**AND dark6 IS ALSO WHAT MAKES THIS ENTRY WORTH A SLOT.** It gives this kernel the control it never had.
The tuple below is EXACTLY dark6's six with `vjepa_warm_s2026` appended — verified member-for-member, not
inferred. `dark7` (55756369) is EXACTLY dark6's six with `small_batch6_ff` appended. Three strictly nested
entries, one variable each:

    dark6    six all-data members                      0.937
    dark7    + a seventh of the SAME family            PENDING
    farseat  + a seventh of the ONE OTHER family       this entry

`2026-08-24_architecture_diversity_is_closed...` closes the encoder axis on the mechanism that slice
sampling, the 2D-slice-then-pool architecture, the targets, the objective and the head are held fixed
across every arm, so a trunk swap changes how features are SPELLED and not what is optimised. dark6 is
that closure's board confirmation, and it is far stronger than any seat swap: a whole extra member of the
saturated family buys nothing. `vjepa_warm_s2026` is the only line in the roster that breaks the second
term — it tokenises two slices per token and is handed the stack. Whether the seat is inert for a member
OUTSIDE that family is the one thing none of these entries has measured.

Read against dark7, not against augseat. If dark7 and farseat land together, the reading is COUNT. If they
separate, the reading is FAMILY, and that is the first live lever past the 0.937 plateau this board has
shown in four entries.

**WHY THIS MEMBER, IF THE GATE OPENS.** `vjepa_warm_s2026` is a V-JEPA2-large tubelet transformer — two
slices per token, an encoder family with no other representative in the roster, warm-started from our own
2D weights. Every other member in this file is a 2D image encoder. Its nearest neighbour in the pool is
**0.8210** against a seated band of **0.8801-0.9143**, which puts it 0.06-0.09 further out than anything
that has ever held a seat. Distance is the only quantity at the top of this board with a live mechanism.

**THE ONE VARIABLE IS THE SEVENTH MEMBER, AGAIN.** Same six `--fold -1` members, count 7, `RankAverage`,
overlap False, same kit. `lastseat` (0.937) is the same-type control that says a seat exchange between
members of the same kind moves nothing; `augseat` is the first different-kind exchange; this is the second
and the far one.

**WHAT THE OFFLINE RULER SAYS, AND WHY IT IS NOT THE ARGUMENT.** Solo **0.8337**, second-worst in the file,
and seat value **+0.0002** — 0 of 1 seats clearing the +0.0015 board-resolvable threshold. Both numbers
argue against this member and both are produced by a 909-study same-distribution holdout that is separately
measured to under-price DIVERSITY by **2.4x** (0.0059 offline became 0.014 on the board). The ledger holds
three members mounted on negative offline deltas that carried board entries — `calib_span` (-0.0020),
`base_dense` (-0.0066), `through_dense` (-0.0020, seated in the champion). The offline number is not
evidence for this member; the argument is that this ruler has never been able to see the axis being tested.

**WHAT IS ALREADY SETTLED, so this does not re-litigate it.** The volumetric axis is measured flat ON THE
MEAN and that question is closed: `vjepa_probe` frozen 0.7620 against DINOv2-large 0.7669 is a tie, and
`vjepa_member` fitted cold from natural video read 0.8343 against its own pre-registered 0.845 bar — a
miss, with its mechanism named (video pretraining teaches temporal object permanence and camera motion;
an MRI stack's third axis is anatomical depth, which has neither). This entry does not claim the 3D
encoder is better. It claims it is DIFFERENT, which is a separate quantity and the only one still live.

**WHAT COSTS WHAT.** One slot, ZERO card — the member is trained, the weights are published as
`dimiturnt/vjepa2-hf/vjepa2-vitl`, and `package_members.shippable()` accepts the roster line. P(this
clears the owner's 1.5% bar) is about 0.00; nothing on this board has ever moved that far in one seat.

**NO OFFLINE NUMBER IS REPORTED AND NONE CAN BE.** Six members have seen the entire 909-study holdout.

PRE-REGISTRATION, written before the number, against **dark6 0.937** — the matched control, six of these
exact seven members — and read alongside dark7, the same-family arm.

- **0.940 or above** — the seventh seat is inert for the saturated family and LIVE for a member outside
  it. That is the first thing to move this board since finalfit, and it says the ensemble is capped by
  architectural redundancy rather than by data or by member quality. If dark7 is flat at the same time,
  the attribution is FAMILY and it is clean. The next entries then buy members for what they are, not for
  what they score: RadDINO and the volumetric arms come back off the shelf, and the last slot goes to a
  second out-of-family seat rather than to the vote.
- **0.936-0.939** — flat, and with dark6 in hand this is now a strong closure rather than a shrug. The
  seventh seat is inert for EVERY member in the inventory including the furthest one that exists, so
  member-level diversity is spent as a lever on this board. That retires the 2.4x under-pricing argument,
  `knee-avzz`, and the whole distance program, and it points the remaining slot at the vote and the
  serving path — the only two axes never varied on a board entry.
- **0.933-0.935** — the far member COSTS, on a seat where a same-family member costs nothing. The vote
  wants agreement, not distance, and that is a reading about `RankAverage` rather than about V-JEPA2:
  a rank average over near-identical rankers is robust, and one genuine dissenter in seven is diluted at
  best and disruptive at worst. Argues LogitAverage / a weighted vote for the last slot.
- **0.932 or below** — a 3D member actively breaks the vote, OR the mount is wrong. The mount is the
  first suspect and not the second: this member was remounted 2026-08-25 01:14Z after months unmounted,
  and its trunk `dimiturnt/vjepa2-hf` was declared on seven kernels in one mechanical edit. Re-run
  `witness_check` and `mount_check --live` before any of this is believed.
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
        "vjepa_warm_s2026",
        "base_dense_ff",
        "lr4e-5_ws_ff",
        "through_dense_ff",
    )
    KernelRun(names=chosen, overlap=False).run()


if __name__ == "__main__":
    main()
