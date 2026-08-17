"""The four that scored 0.918, with the ONE member that reads through a different ENCODER swapped in.

Every other member in the pool is a DINOv2 backbone; `lora_r8_v3small_224_s2026` is DINOv3, adapted with
rank-8 LoRA at 224 px. So the pool's diversity is optimiser diversity over one set of learned features,
and the errors it averages are correlated by the thing they all share. This entry varies exactly that
coordinate and nothing else: same kit, same vote, same passes, same count, `sliceattn_s2026` replaced by
`lora_r8_v3small_224_s2026`. It is the companion question to `kneemri-crop`, which varies the decode path
while holding the encoder fixed.

Bought on a NEGATIVE offline delta, and a larger one than crop's. Measured 2026-08-17 on the 909-study
fold-0 holdout: the shipped four are 0.8612, the crop swap 0.8604, this four 0.8590 — and the member
itself is the pool's weakest single at 0.8215, 0.031 under res392. The 0.0022 gap is a third of the
0.0062 noise floor, so offline this is still a tie; what it costs is a tie bought with a visibly worse
member, which is the honest shape of the bet. The axis is worth a slot because our holdout has twice
underpriced decode-path diversity by 2.4x (knee-nas5), and encoder diversity is strictly further from the
shared-feature failure mode than crop is.

If it lands near 0.918 then member quality is what the board reads and a weak member cannot be carried by
being different. If it lands above, the pool is feature-correlated and the next arms should be new
backbones rather than new recipes.
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

    encoder_diverse = (
        "res392_ws_s2026",
        "intensity392_ws_s2026",
        "calibrated_targets_s2026",
        "lora_r8_v3small_224_s2026",
    )
    KernelRun(names=encoder_diverse, overlap=False).run()


if __name__ == "__main__":
    main()
