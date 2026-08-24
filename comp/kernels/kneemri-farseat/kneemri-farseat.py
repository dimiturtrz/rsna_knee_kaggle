"""The seventh seat filled by the FURTHEST member this project has ever produced, on a 3D encoder.

GATED — DO NOT SUBMIT until `kneemri-augseat` has read. This entry and that one ask the same question
with different members, and augseat's band decides whether this one is worth a slot at all:

- augseat **>= 0.940** — the board pays for a member the offline ruler killed, on distance the ruler
  cannot see. `vjepa_warm_s2026` is then the strongest such candidate that exists here by a wide margin,
  and this entry is the next slot.
- augseat **0.936-0.939** or **below** — type of distance does not carry at seven seats. This entry is
  then a second measurement of a null on a WEAKER member, and the pricing rule refuses it. Do not submit;
  leave this kernel unsubmitted as a written-down plan and spend the slot elsewhere.

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

PRE-REGISTRATION, written before the number, against `allfit` 0.937, `lastseat` 0.937, and whatever
`augseat` reads — which must be filled in here before this is submitted, because the two entries differ
only in the KIND of the seventh member and reading one without the other wastes both.

- **0.940 or above** — distance of a kind the pool has never held buys a board tick where quality does not.
  Combined with augseat this makes the seventh seat a DISTANCE slot rather than a member slot, and the next
  entries build members chosen for remoteness, not for holdout score.
- **0.936-0.939** — neutral. The 2.4x under-pricing survives (a member 0.026 below the seat it replaces
  holding the board flat is itself the blindness argument), but the seat argument does not, and the
  distance axis stops being worth slots.
- **0.933-0.935** — the far member costs about what the all-data dose bought. Distance at this magnitude
  is not free, and the ensemble wants members that AGREE more than it wants members that differ.
- **0.932 or below** — a 3D member actively breaks the vote. That is the reading that would retire the
  whole distance argument, including `knee-avzz`'s, and send the next slot back to member quality.

Four bands, each changing something written down.
"""

import logging
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# >>> KERNEL-BOOTSTRAP
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


# <<< KERNEL-BOOTSTRAP


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    kit = find_kit()
    install_wheels(kit)
    sys.path.insert(0, str(kit))

    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - kit is on sys.path only after install_wheels

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
