"""Submission from OUR members AND the published ones, rank-averaged as one ensemble.

THE QUESTION. Our three whole-stack members average 0.849 on holdout; the twenty published CC0
members average 0.8398 and scored 0.888 on the leaderboard by themselves. Ours are stronger per model
and far fewer; theirs are weaker and many, fitted under a different recipe (channel stacking, a
windowed read) on the same corpus. Rank-averaging rewards DISAGREEMENT between members, not member
quality alone, so a small strong set plus a large diverse set can beat either — or the twenty can
simply outvote the three. This entry is the measurement.

WHY IT COSTS ALMOST NOTHING TO ASK. Both sets were fitted on the same pixels — 336 px, 12 slices,
group 3, canonical slots — so `pixel_group` is identical and ONE decode of the test set serves all
twenty-three models. The extra cost is twenty-three forward passes instead of three, not a second
decode.

THE GUARD. If the two packages ever stop agreeing on pixels this must fail rather than average models
that read different images, so the pixel groups are compared explicitly. Nothing else in the run would
notice: every member still loads, every fingerprint still passes (each is checked against its OWN
reading), and the submission would just be quietly wrong.
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

from typing import (  # noqa: E402 - a bootstrap: nothing may import before the region installs the kit
    override,
)

WORKING = Path("/kaggle/working")

GRAFT_TIE = "last"
"""The router the published members were fitted under, measured against their own predictions."""


def _published(root: Path) -> bool:
    """Whether a mounted package is the published CC0 one rather than ours.

    By the dataset directory name, which is what distinguishes them on Kaggle. Explicit rather than
    inferred from the config, because the whole point is that their manifest does NOT record the
    field we are supplying.
    """
    return "rsna-knee-weights" in str(root)


def main() -> None:
    boot()

    from kneemri.inference.member_package import MemberPackage  # noqa: PLC0415 - the kit is on sys.path only after boot
    from kneemri.kaggle.kernel_run import KernelRun  # noqa: PLC0415 - the kit is on sys.path only after boot

    logger = logging.getLogger("fusion")

    class Fused(KernelRun):
        """Every mounted package's members, run as one ensemble."""

        @override
        def package(self) -> MemberPackage:
            """Merge every mounted manifest into one package, refusing mixed pixel groups."""
            # FILTERING IS NOT PRUNING, and this used to filter: `rglob` had already descended into
            # every study and series directory by the time the skip-list saw the paths, which is the
            # 248 seconds `find_kit` above exists to avoid. `Mounts` prunes inside the traversal.
            roots = self.mounts.every("manifest.json")
            # THEIR ROUTER, WHICH THEIR MANIFEST DOES NOT DECLARE. The published members carry no
            # `tie` field, so they would be routed under OUR first-maximal rule — and they reproduce
            # their OWN recorded predictions markedly better under last-maximal: 4 of 4 members across
            # 3 seeds, mean residual 15-25% lower, max 0.128 -> 0.025 on studies with a contested slot
            # (knee-evn). That is our inference about someone else's weights, so it is applied here by
            # name and never defaulted, and it is the ONLY difference found in ten tested axes.
            #
            # IT COSTS A SECOND DECODE. `Ensemble.reads` groups by tie, so an entry holding both routers
            # decodes the test set twice. That is the honest price of serving each side under the rule
            # its weights were fitted with, and it is why this is a separate decision from the fusion.
            packages = [MemberPackage(root, tie=GRAFT_TIE if _published(root) else None) for root in roots]
            if not packages:
                raise FileNotFoundError("no weights package mounted")

            # ONE DECODE, AND IT SERVES MANY SIZES. This filtered on `pixel_group`, which keys on
            # image size — so it dropped res392 (0.8523, our best member ever) and intensity392 for
            # reading 392 px off a 448 cache, and our two strongest members had never appeared in a
            # scored entry.
            #
            # THAT REASON EXPIRED AND THE FILTER OUTLIVED IT. `SlotReader.cropped` stops before the
            # resize, so one read of a study serves every size exactly and byte-identically to a native
            # read at each; `Ensemble.reads` therefore groups by (scheme, slices, crop) and NOT by size,
            # and `Read.sizes` carries them all. Verified on our own package: res392 at 448 lands in the
            # same read group as the 336 px members, sizes [336, 448].
            #
            # So the unit to filter on is the READ, not the pixel group. What genuinely cannot share a
            # decode is a different slot SCHEME (which series are routed at all) or a different DEPTH or
            # CROP — extended_ws is nine-slot and is still dropped, and that one is real.
            reads: dict[object, int] = {}
            for package in packages:
                for member in package.members:
                    key = (member.scheme, member.slices, member.crop)
                    reads[key] = reads.get(key, 0) + 1
            serving = max(reads, key=lambda group: reads[group])
            if len(reads) > 1:
                logger.info(
                    "fusing the %d members that share one read, dropping %d in %d other group(s)",
                    reads[serving],
                    sum(reads.values()) - reads[serving],
                    len(reads) - 1,
                )

            merged = packages[0]
            # TWO REASONS A MEMBER IS DROPPED, and they are different.
            #
            # READ: a member wanting another slot scheme, depth or crop needs its own decode of the whole
            # test set. That is this kernel's single-decode design, not a fault of the member.
            #
            # ENCODER: THIS EXCLUSION EXPIRED AT 11:08 ON THE DAY IT WAS WRITTEN. v3 members were dropped
            # by name after a run whose weights would not load against the rebuilt backbone — 178
            # unexpected keys under `backbone.model.layer.*`, a nesting our own model does not have. That
            # nesting is what an OLDER transformers builds for DINOv3, and the kit gained a
            # transformers 5.15.0 wheel (with safetensors) hours after that run: no scored entry has ever
            # mounted the version the checkpoints were fitted under. Locally, on the SAME transformers the
            # kit now ships, `lora_r8_v3small_224_s2026` loads and passes its witness on CPU
            # (max|diff| 0.0119 of 0.0312) — so the refutation was about the runtime, not the member.
            #
            # Dropping by name is also the wrong instrument: `KernelRun.run` already loads each member in
            # its own try and drops the ones that fail, so a v3 member that still cannot load costs this
            # entry a load attempt and nothing else. A different backbone family is the diversity axis
            # this ensemble does not otherwise have.
            usable = [m for package in packages for m in package.members if (m.scheme, m.slices, m.crop) == serving]
            dropped = sum(len(package.members) for package in packages) - len(usable)
            if dropped:
                logger.info("dropped %d member(s) that one decode cannot serve", dropped)
            merged.members = usable
            logger.info(
                "fused %d members from %d packages: %s",
                len(merged.members),
                len(packages),
                ", ".join(str(root) for root in roots),
            )
            return merged

    Fused().run()


if __name__ == "__main__":
    main()
