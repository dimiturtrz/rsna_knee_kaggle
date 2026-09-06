"""Assemble the kneemri-kit Kaggle dataset: our source tree + trained weights + offline dependency wheels.

The submission runs in a code competition with no internet, so everything the kernel imports has to be mounted.
This packs the current `kneemri`/`core` source (no vendoring — it is our own tree), the trained weights, and
the wheels for the dependencies missing from Kaggle's base image into a directory, writes the dataset
metadata, and prints the push command.

Which wheels are actually missing is a property of Kaggle's image, not of this script — probe it from a
kernel before assuming (torch/numpy/scipy/pandas ship; pydicom and the timm/DINOv2 path are the open ones).
"""

import argparse
import json
import logging
import shutil
import subprocess
from pathlib import Path

from core.paths import DataRoot
from kneemri.kaggle.kit_check import KitCheck
from kneemri.kaggle.kit_wheels import KitWheels

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[1]
_DATASET_ID = "dimiturnt/kneemri-kit"
_PACKAGES = ("kneemri", "core")
_SOURCE_DIR = "kneemri_src"


def _copy_extras(extras: list[Path], out: Path) -> None:
    """Copy each named file into the kit root, refusing a name that is not a file.

    Refused rather than skipped because the caller names these by hand on a command line, and a
    mistyped path that copies nothing builds a kit that passes every other check and drops a member
    inside the kernel — the failure `--wheels` already cost a slot to learn.
    """
    for extra in extras:
        if not extra.is_file():
            raise SystemExit(f"--extra {extra} is not a file")
        shutil.copy2(extra, out / extra.name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    # OUT OF THE REPO by default. The kit is ~200 MB of weights and wheels beside a copy of our own
    # packages - build output, not source - and "data lives out of the repo" is the house rule for
    # exactly that. It also removes a real hazard: while it sat at comp/kit, adding `comp` to the lint
    # scope put a STALE COPY of kneemri under the type checker, which then reported every refactor as
    # an error at a path nobody edits. `pyrefly check <paths>` discards project_excludes when paths are
    # given on the command line (its own --help says so), so no config could exclude it - the directory
    # simply must not live inside a checked tree.
    ap.add_argument(
        "--out",
        type=Path,
        default=DataRoot.from_config(_REPO / "paths.yaml").processed() / "kit",
        help="where to assemble the kit. Defaults out of the repo, beside the other build artefacts.",
    )
    ap.add_argument("--weights", nargs="*", default=[], help="weight files under processed/")
    # Both optional, because the probe measured that Kaggle's image already carries pydicom, timm,
    # transformers, cv2, SimpleITK, sklearn, scipy and polars — this competition may need no wheels at
    # all, unlike the sibling one that needed six. A kit of pure source is a legitimate kit.
    # BESIDE THE KIT, under the data root. These lived in /tmp - in three scattered copies - and /tmp is
    # wiped on reboot, so the submission path depended on scratch files surviving by luck. verify_kit
    # would have caught the absence (it names the missing wheel), but a gate catching it at submission
    # time is a slot spent discovering that a dependency was never durable.
    ap.add_argument(
        "--wheels",
        type=Path,
        default=DataRoot.from_config(_REPO / "paths.yaml").processed() / "wheels",
        help="dir holding the offline dependency wheels the Kaggle image lacks",
    )
    ap.add_argument("--processed", type=Path, default=None, help="processed data dir holding the weight files")
    # A FILE THAT IS NEITHER OURS NOR A WEIGHT, and the kernel cannot reach any other way. `kernel_sources`
    # mounts a kernel's OUTPUT, not its source, so a lifted foreign program has no route into a no-internet
    # kernel except this one. It also carries the licence texts that route with the materials they cover:
    # the DINOv3 Agreement travels beside the weights it grants, which is the condition, not a courtesy.
    ap.add_argument(
        "--extra",
        nargs="*",
        type=Path,
        default=[],
        help="files copied verbatim into the kit root (foreign source, licences)",
    )
    arguments = ap.parse_args()
    if arguments.weights and arguments.processed is None:
        ap.error("--weights needs --processed to say where the files live")

    out = arguments.out.resolve()
    # The default is out of the repo; an --out that points back INTO it undoes that in one flag, and
    # the cost is not a stray directory. A copy of `kneemri` under a checked path is type-checked as if
    # it were source, so a protocol that grows a field fails the gate at a snapshot taken before the
    # field existed — an error at a path nobody edits, unfixable by editing, and unexcludable because
    # `pyrefly check <paths>` discards project_excludes. `comp/kit_staging/` was exactly that, and it
    # cost a green gate to find. Refused rather than documented, because the comment above already said
    # this and the directory existed anyway.
    if out.is_relative_to(_REPO):
        message = f"--out must be outside the repo ({_REPO}): a source snapshot under a checked path is type-checked"
        raise SystemExit(message)
    if out.exists():
        shutil.rmtree(out)
    (out / _SOURCE_DIR).mkdir(parents=True)
    for package in _PACKAGES:
        shutil.copytree(_REPO / package, out / _SOURCE_DIR / package, ignore=shutil.ignore_patterns("__pycache__"))
    if arguments.wheels is not None:
        for wheel in sorted(arguments.wheels.glob("*.whl")):
            shutil.copy2(wheel, out / wheel.name)
    for weight in arguments.weights:
        shutil.copy2(arguments.processed / weight, out / weight)
    _copy_extras(arguments.extra, out)
    (out / "dataset-metadata.json").write_text(
        json.dumps({"title": "kneemri-kit", "id": _DATASET_ID, "licenses": [{"name": "CC0-1.0"}]}, indent=2)
    )
    # THE CODE ON KAGGLE GOES STALE SILENTLY, and nothing in a kernel run said which version it held.
    # On 2026-08-13 the mounted kit was a day and 54 commits behind, missing Member.precision — so a
    # bf16 member would have been served in fp16 on a T4, which is the arithmetic that property exists
    # to fix. Stamped here and logged by kernel_run, so a run's own output names the code that made it.
    git = shutil.which("git") or "git"

    # `env=KitCheck.unhooked()` because `-C` is overruled by an inherited GIT_DIR/GIT_INDEX_FILE, which git
    # exports into every hook it runs. That matters most HERE: `status --porcelain` against a borrowed
    # index would stamp the kit's `dirty` flag from a different tree, and that stamp is the ground truth
    # `kit_check` compares every later kit against.
    def asked(*command: str) -> str:
        return subprocess.run(  # noqa: S603 - resolved executable, fixed argv, no shell, our own repo
            [git, "-C", str(_REPO), *command], capture_output=True, text=True, check=True, env=KitCheck.unhooked()
        ).stdout.strip()

    revision = asked("rev-parse", "--short", "HEAD")

    # SCOPED TO THE PACKAGES THE KIT COPIES, because a whole-tree `status` is not the question this stamp
    # answers. `kit_check` states the principle from the other side: what matters is not that the kit
    # differs from HEAD but that it differs in the files a kernel LOADS. An unscoped status also reads a
    # SECOND agent's working tree — this repo is shared, and on 2026-09-04 four untracked PDFs and one
    # modified research note under a peer's lane stamped a kit DIRTY whose every copied file was clean,
    # which is a refusal that can only be answered by committing somebody else's work.
    dirty = bool(asked("status", "--porcelain", "--", *_PACKAGES))
    (out / _SOURCE_DIR / "kit.json").write_text(json.dumps({"commit": revision, "dirty": dirty}, indent=2))

    # THE WHEELS DIRECTORY IS AN ARGUMENT, AND THE WRONG ONE BUILDS A KIT THAT PASSES EVERY OTHER CHECK.
    # On 2026-08-17 this ran against a scratch directory holding two of the four wheels, so the kit lost
    # transformers 5.15.0 — under which our DINOv3 checkpoints were fitted — and a v3 member was dropped
    # inside the kernel. Refused at BUILD time rather than only by verify_kit, because a kit is uploaded
    # by hand and the gate that is not on the path from build to push is the gate that is skipped.
    lacking = KitWheels().missing(out)
    if lacking:
        raise SystemExit("the kit is missing wheels it cannot run without:\n  " + "\n  ".join(lacking))

    wheels = [w.name for w in out.glob("*.whl")]
    weights = list(arguments.weights)
    stamp = f"{revision} (DIRTY)" if dirty else revision
    extras = [extra.name for extra in arguments.extra]
    logger.info(
        f"kit at {out} from {stamp}: src={','.join(_PACKAGES)} wheels={wheels} weights={weights} extra={extras}"
    )
    logger.info(f"push a new version:\n  kaggle datasets version -p {out} -m '<message>' --dir-mode zip")


if __name__ == "__main__":
    main()
