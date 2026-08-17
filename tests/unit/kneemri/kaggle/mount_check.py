"""Every kernel's declared mounts against the members it names, swept without running anything."""

import json
from pathlib import Path

from kneemri.inference.members import MemberPackage
from kneemri.kaggle.mount_check import MountCheck

DINOV2 = "metaresearch/dinov2/PyTorch/small/1"
DINOV3 = "dimiturnt/dinov3-hf"


def _package(tmp_path: Path) -> MemberPackage:
    """Two members on two different encoders, which is the only property these checks read."""
    root = tmp_path / "members"
    root.mkdir()
    entries = []
    for identifier, variant in (("v2member", "small"), ("v3member", "v3small")):
        (root / f"{identifier}.pt").write_bytes(b"")
        entries.append(
            {
                "id": identifier,
                "file": f"{identifier}.pt",
                "fold": 0,
                "config": {"variant": variant},
                "fingerprint": [0.0],
                "holdout": 0.84,
                "annot": 0.87,
            }
        )
    (root / "manifest.json").write_text(json.dumps({"format": 2, "members": entries}), encoding="utf-8")
    return MemberPackage(root)


def _kernel(tmp_path: Path, name: str, body: str, sources: list[str]) -> Path:
    kernel = tmp_path / name
    kernel.mkdir()
    (kernel / f"{name}.py").write_text(body, encoding="utf-8")
    (kernel / "kernel-metadata.json").write_text(
        json.dumps({"id": f"dimiturnt/{name}", "model_sources": [DINOV2], "dataset_sources": sources}),
        encoding="utf-8",
    )
    return kernel


def test_entries(tmp_path: Path):
    """Kernel directories are found by their metadata, so a stray directory is not one."""
    _kernel(tmp_path, "kneemri-a", "KernelRun().run()\n", [DINOV3])
    (tmp_path / "notakernel").mkdir()
    assert [kernel.name for kernel in MountCheck(_package(tmp_path)).entries(tmp_path)] == ["kneemri-a"]


def test_verdict(tmp_path: Path):
    """A named subset is judged on ITS encoders; an unresolvable selection on the whole package.

    The refusal is the case that already happened: `kneemri-solo` runs `strongest(1)` over a package
    that gained a DINOv3 member, and its metadata still declared DINOv2 only — every fingerprint agrees
    and the kernel cannot build the backbone.
    """
    check = MountCheck(_package(tmp_path))

    subset = _kernel(tmp_path, "kneemri-subset", "KernelRun(names=('v2member',)).run()\n", [])
    assert check.verdict(subset).startswith("ok"), "a DINOv2-only subset needs no DINOv3 mount"

    ranked = _kernel(tmp_path, "kneemri-ranked", "KernelRun(members=1).run()\n", [])
    assert "does not declare dimiturnt/dinov3-hf/vits16" in check.verdict(ranked)

    declared = _kernel(tmp_path, "kneemri-declared", "KernelRun(members=1).run()\n", [DINOV3])
    assert check.verdict(declared).startswith("ok")

    absent = _kernel(tmp_path, "kneemri-absent", "KernelRun(names=('gone',)).run()\n", [DINOV3])
    assert check.verdict(absent).startswith("REFUSED"), "a name the package does not hold is a finding too"

    foreign = _kernel(tmp_path, "kneemri-foreign", "MemberPackage.find(mounts)\n", [])
    assert check.verdict(foreign).startswith("skipped"), "a kernel running someone else's package is not checkable"
