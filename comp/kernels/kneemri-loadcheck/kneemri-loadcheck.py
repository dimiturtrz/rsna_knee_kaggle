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
import zipfile
from pathlib import Path


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("loadcheck")
    kit = find_kit()
    install_wheels(kit)
    sys.path.insert(0, str(kit))

    import torch  # noqa: PLC0415 - after the kit is on sys.path
    import transformers  # noqa: PLC0415 - the version is the point
    from transformers import AutoModel  # noqa: PLC0415 - after the kit is on sys.path

    from kneemri.backbone import ENCODERS  # noqa: PLC0415 - after the kit is on sys.path
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
    logger.info("kit %s", run.revision())
    package = run.package()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    failures = 0
    for member in package.members:
        encoder = ENCODERS[member.variant]
        try:
            weights = run.find_encoder(encoder)
            model = package.load(member, AutoModel.from_pretrained(weights), encoder.dim, device)
        except Exception as error:  # noqa: BLE001 - reporting every member beats stopping at the first
            failures += 1
            logger.error("LOAD FAILED %s (%s): %s", member.identifier, member.variant, str(error)[:200])
            continue
        # LOADING IS NOT RUNNING. Two entries died with the model built and the weights in place, so the
        # check has to reach the arithmetic: one forward per member, in the precision that member was
        # fitted under, on this card. A T4 reports bf16 as supported and emulates it, which is exactly
        # the kind of "supported" that fails inside a kernel rather than at the autocast call.
        window = member.window(member.slices, overlap=False)
        pixels = torch.zeros(1, len(member.config["slots"]), window.width, window.size, window.size, dtype=torch.uint8)
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=member.precision, enabled=device.type == "cuda"):
                model(pixels.to(device), torch.ones(1, len(member.config["slots"])).to(device), window.size)
        except Exception as error:  # noqa: BLE001 - the point is to name which member and which failure
            failures += 1
            logger.error(
                "FORWARD FAILED %s (%s, %s): %s", member.identifier, member.variant, member.precision, str(error)[:300]
            )
        else:
            logger.info("ok     %s (%s, %s, read %d)", member.identifier, member.variant, member.precision, window.size)
    logger.info("%d members, %d failed", len(package.members), failures)


if __name__ == "__main__":
    main()
