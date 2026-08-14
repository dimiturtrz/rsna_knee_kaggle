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

import logging
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # THE CHECKER CAN IMPORT THE KIT; THE RUNTIME CANNOT. This file bootstraps — it runs before the kit
    # is on sys.path — so these names exist for annotations only and never at import time on Kaggle.
    import torch as torch_types

    from kneemri.ensemble import Ensemble
    from kneemri.kernel_run import KernelRun
    from kneemri.members import Member, MemberPackage
    from kneemri.model import StudyModel


def find_kit() -> Path:
    """The directory to put on `sys.path`, found by looking for the package rather than the wrapper."""
    marker = next(iter(sorted(Path("/kaggle/input").rglob("kneemri/model.py"))), None)
    if marker is not None:
        return marker.parent.parent
    archive = next(iter(sorted(Path("/kaggle/input").rglob("kneemri_src.zip"))), None)
    if archive is None:
        raise FileNotFoundError("no kneemri package under /kaggle/input")
    zipfile.ZipFile(archive).extractall("/tmp/kit")
    return next(iter(sorted(Path("/tmp/kit").rglob("kneemri/model.py")))).parent.parent


def install_wheels(kit: Path) -> None:
    """Everything our modules import that the image lacks — or ships at the wrong version."""
    wheels = sorted(kit.parent.glob("*.whl")) + sorted(kit.glob("*.whl"))
    if wheels:
        subprocess.run(  # noqa: S603 - pip installing wheels we mounted ourselves, fixed argv, no shell
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "-q", *[str(w) for w in wheels]],
            check=True,
        )


def check(  # noqa: PLR0913 - a probe collecting a runtime's answers, not an API anyone calls twice
    member: "Member",
    run: "KernelRun",
    package: "MemberPackage",
    ensemble_of: "Callable[[], Ensemble]",
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
    from transformers import AutoModel  # noqa: PLC0415 - after the kit is on sys.path

    from kneemri.backbone import ENCODERS  # noqa: PLC0415 - after the kit is on sys.path
    from kneemri.submission import STUDY_CHUNK  # noqa: PLC0415 - the chunk a submission actually sends

    encoder = ENCODERS[member.variant]
    try:
        weights = run.find_encoder(encoder)
        package.load(member, AutoModel.from_pretrained(weights), encoder.dim, device)
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
    for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
        started = time.time()
        try:
            with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda" and dtype is not torch.float32):
                scored = ensemble_of().predict_member(
                    member, chunk.to(device), torch.ones(STUDY_CHUNK, slots).to(device), overlap=True
                )
        except Exception as error:  # noqa: BLE001 - naming which member and which failure is the point
            logger.error("CHUNK FAILED %s in %s: %s", member.identifier, name, str(error)[:250])
            return False
        elapsed = time.time() - started
        if reference is None:
            reference = scored
            logger.info("       chunk of %d in %s: %5.1fs (reference)", STUDY_CHUNK, name, elapsed)
        else:
            worst = float(abs(scored - reference).max())
            logger.info("       chunk of %d in %s: %5.1fs, max|diff| vs fp32 %.2e", STUDY_CHUNK, name, elapsed, worst)
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

    from kneemri.kernel_run import KernelRun  # noqa: PLC0415 - after the kit is on sys.path

    logger.info("transformers %s, torch %s", transformers.__version__, torch.__version__)
    capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
    logger.info(
        "device %s, capability %s, bf16 %s",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        capability,
        torch.cuda.is_bf16_supported() if torch.cuda.is_available() else "n/a",
    )

    run = KernelRun()
    models: "dict[str, StudyModel]" = {}
    logger.info("kit %s", run.revision())
    package = run.package()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models: "dict[str, StudyModel]" = {}

    def ensemble_of() -> "Ensemble":
        """Built once, and only after the first member has loaded into `models`."""
        from kneemri.ensemble import Ensemble  # noqa: PLC0415 - after the kit is on sys.path

        return Ensemble(package, models, device)

    failures = 0
    for member in package.members:
        before = dict(models)
        if not check(member, run, package, ensemble_of, device, logger):
            failures += 1
        models.update(before)
    logger.info("%d members, %d failed", len(package.members), failures)


if __name__ == "__main__":
    main()
