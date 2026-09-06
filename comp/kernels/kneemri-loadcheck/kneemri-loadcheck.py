"""Does every member LOAD on the runtime that will score it? No submission, no slot.

THE GAP THIS CLOSES. Four entries failed and each cost a slot to learn one fact. Every one of those
facts was about the RUNTIME rather than the model: a T4 cannot run bf16 autocast (sm_75), and Kaggle's
image ships transformers 5.0.0 while our members were trained under 5.15.0, whose DINOv3 class nests the
encoder one level deeper — so the same weights and the same config build a different key set and the
load fails with 178 unexpected keys.

None of that is visible from a machine with an sm_120 card and current libraries. All of it is visible
in ninety seconds here, for free: a kernel that only LOADS is not a submission and spends nothing.

Run this after any kit rebuild and before any entry. It answers, per member, the only question a
submission cannot afford to ask: does this thing load, on this runtime, from these mounts.
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

import time  # noqa: E402 - a bootstrap: nothing may import before the region installs the kit
from collections.abc import (  # noqa: E402 - a bootstrap: nothing may import before the region installs the kit
    Callable,
)
from typing import (  # noqa: E402 - a bootstrap: nothing may import before the region installs the kit
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    # THE CHECKER CAN IMPORT THE KIT; THE RUNTIME CANNOT. This file bootstraps — it runs before the kit
    # is on sys.path — so these names exist for annotations only and never at import time on Kaggle.
    import torch as torch_types

    from kneemri.inference.ensemble import Ensemble
    from kneemri.inference.member_package import MemberPackage
    from kneemri.inference.members import Member
    from kneemri.kaggle.kernel_run import KernelRun
    from kneemri.models.model import StudyModel


def check(  # noqa: PLR0913 - a probe collecting a runtime's answers, not an API anyone calls twice
    member: "Member",
    run: "KernelRun",
    package: "MemberPackage",
    loaded: "dict[str, StudyModel]",
    ensemble_of: "Callable[..., Ensemble]",
    device: "torch_types.device",
    logger: logging.Logger,
) -> bool:
    """Load, forward and chunk ONE member on this runtime, reporting rather than raising.

    Annotated through `TYPE_CHECKING` because this file is a BOOTSTRAP: it runs before the kit is on
    `sys.path`, so the classes these arguments belong to cannot be imported at runtime — but the type
    checker reads them fine. The three checks are ordered by what each rules out: the weights match the
    class the runtime builds, the arithmetic runs on this card, and a chunk of the shape a submission
    actually sends fits in its memory.
    """
    import torch  # noqa: PLC0415 - after the kit is on sys.path

    from kneemri.kaggle.submission import STUDY_CHUNK  # noqa: PLC0415 - the chunk a submission actually sends
    from kneemri.models.backbone import ENCODERS  # noqa: PLC0415 - after the kit is on sys.path

    encoder = ENCODERS[member.variant]
    try:
        weights = run.find_encoder(encoder)
        # REGISTERED, not merely built. The return value was discarded here, so `models` stayed empty
        # and every forward below died on `models[member.identifier]` — a KeyError of the member's own
        # name, reported as "FORWARD FAILED". On 2026-08-14 that read as six members failing on a T4
        # when all six had in fact loaded their 223 weights cleanly. A probe that manufactures the
        # failure it is looking for is worse than no probe.
        # THROUGH THE REGISTRY, which is what `kernel_run` does and what this probe exists to imitate.
        # Spelled `AutoModel.from_pretrained(weights)` it built the trunk RAW, skipping the adapter the
        # entry names — so a ConvNeXt member would arrive as a feature map and a Swin one as a sequence
        # with no prefix, and `tokens[:, 1:]` would eat the grid's first cell with no error to announce
        # it. A load probe that loads a member differently from the way it ships can only certify the
        # families it happens not to differ on.
        loaded[member.identifier] = package.load(member, encoder.load(weights), encoder.dim, device)
    except Exception as error:  # noqa: BLE001 - reporting every member beats stopping at the first
        logger.error("LOAD FAILED %s (%s): %s", member.identifier, member.variant, str(error)[:200])
        return False

    slots = len(member.config["slots"])
    window = member.window(member.slices, overlap=False)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=member.precision, enabled=device.type == "cuda"):
            pixels = torch.zeros(1, slots, window.width, window.size, window.size, dtype=torch.uint8)
            ensemble_of().models[member.identifier](pixels.to(device), torch.ones(1, slots).to(device), window.size)
    except Exception as error:  # noqa: BLE001 - the point is to name which member and which failure
        logger.error("FORWARD FAILED %s (%s): %s", member.identifier, member.precision, str(error)[:300])
        return False
    logger.info("ok     %s (%s, %s, read %d)", member.identifier, member.variant, member.precision, window.size)

    # A FORWARD OF ONE STUDY PROVES NOTHING ABOUT MEMORY. A submission sends a chunk of 48, and the
    # entries that died had every member loading and forwarding perfectly well.
    #
    # ALL THREE ARITHMETICS, timed and compared, because the choice is not obvious and I have twice
    # guessed it. bf16 shares fp32's exponent range and gives up mantissa; fp16 keeps precision but tops
    # out near 65504, which is the overflow a bf16-trained activation can meet. Turing has fp16 tensor
    # cores, none for bf16, and no TF32 — so if emulated bf16 is really doing fp32 work, fp32 costs
    # nothing over the bf16 we were already paying AND is the safer of the two. fp32 is the reference
    # here: it cannot overflow where the others might, so the diffs say what each precision costs.
    chunk = torch.zeros(STUDY_CHUNK, slots, member.slices, member.image_size, member.image_size, dtype=torch.uint8)
    reference = None
    #
    # THE KNOB IS THE ENSEMBLE'S, not an autocast wrapped around it. This loop used to set an OUTER
    # autocast per dtype, but `predict_member` opens its own with an explicit dtype, and the inner one
    # wins — so all three arms ran `arithmetic(member)` and reported max|diff| of exactly 0.0 twelve
    # times over, which reads as "the precisions agree" when it means "the knob was never connected".
    for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
        ensemble = ensemble_of(dtype)
        # WHAT IT RESOLVED TO, not what was asked: on a T4 a forced bf16 is legitimately downgraded to
        # fp16, and unreported that makes two arms identical for a second time, by a different route.
        served = ensemble.arithmetic(member)
        started = time.time()
        try:
            scored = ensemble.predict_member(
                member, chunk.to(device), torch.ones(STUDY_CHUNK, slots).to(device), overlap=True
            )
        except Exception as error:  # noqa: BLE001 - naming which member and which failure is the point
            logger.error("CHUNK FAILED %s in %s: %s", member.identifier, name, str(error)[:250])
            return False
        elapsed = time.time() - started
        if reference is None:
            reference = scored
            logger.info("       chunk of %d in %s (%s): %5.1fs (reference)", STUDY_CHUNK, name, served, elapsed)
        else:
            worst = float(abs(scored - reference).max())
            logger.info(
                "       chunk of %d in %s (%s): %5.1fs, max|diff| vs fp32 %.2e",
                STUDY_CHUNK,
                name,
                served,
                elapsed,
                worst,
            )
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("loadcheck")

    # WHERE THE STARTUP GOES, and it is worth knowing precisely: the efficiency board weights total
    # runtime, and this kernel spent 343 s before its first line — six minutes of something. The
    # candidates are container boot, ~1 GB of dataset mounts, unzipping a --dir-mode zip kit, the pip
    # install, and importing torch. Only one of them is ours to fix, so measure rather than assume.
    started = time.time()
    logger.info("t=%5.1fs  script running (everything before this is Kaggle's own boot)", 0.0)
    kit = find_kit()
    logger.info("t=%5.1fs  kit located at %s", time.time() - started, kit)
    install_wheels(kit)
    logger.info("t=%5.1fs  wheels installed", time.time() - started)
    sys.path.insert(0, str(kit))

    import torch as _torch  # noqa: PLC0415, F401 - timing the import is the point

    logger.info("t=%5.1fs  torch imported", time.time() - started)

    import torch  # noqa: PLC0415 - after the kit is on sys.path
    import transformers  # noqa: PLC0415 - the version is the point

    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - after the kit is on sys.path

    logger.info("transformers %s, torch %s", transformers.__version__, torch.__version__)
    capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    logger.info(
        "device %s, capability %s, bf16 %s",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        capability,
        torch.cuda.is_bf16_supported() if torch.cuda.is_available() else "n/a",
    )

    run = KernelRun()
    logger.info("kit %s", run.revision())
    package = run.package()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    failures = 0
    for member in package.members:
        # ONE MEMBER RESIDENT AT A TIME. This probe answers "does each member work", not "do they all
        # fit at once" — and holding twenty-two of them on a 16 GB T4 would make every member after the
        # first few fail for a reason that has nothing to do with the member.
        loaded: "dict[str, StudyModel]" = {}

        def ensemble_of(
            precision: "torch_types.dtype | None" = None, models: "dict[str, StudyModel]" = loaded
        ) -> "Ensemble":
            """Over whichever member is currently resident, bound at definition rather than by closure.

            `precision` forces one arithmetic across the member instead of serving it in the one it was
            fitted in — the only lever that makes a three-precision comparison compare anything.
            """
            from kneemri.inference.ensemble import Ensemble, Serving  # noqa: PLC0415 - after the kit is on sys.path

            return Ensemble(package, models, device, Serving(precision=precision))

        if not check(member, run, package, loaded, ensemble_of, device, logger):
            failures += 1
        loaded.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    logger.info("%d members, %d failed", len(package.members), failures)


if __name__ == "__main__":
    main()
