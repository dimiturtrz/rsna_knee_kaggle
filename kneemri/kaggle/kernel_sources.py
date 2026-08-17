"""What a kernel declares it will have attached, read from its `kernel-metadata.json`.

A kernel has no internet. Everything it runs on — weights, kit, competition data — arrives as a mount
declared in that file, and a source the metadata does not name simply is not there at runtime. The
failure has no early symptom: the kernel starts, installs the kit, loads the manifest, and dies looking
for a directory when the submission slot is already spent.

The declaration is hand-written per kernel while the requirement is derived from the members
(`MemberPackage.mounts`), so the two can drift the moment a package gains a member on an encoder the
kernel was never told about — which is exactly what adding a DINOv3 member to a DINOv2-only entry does.
This module is the side that reads the declaration; `kernel_script` reads the members a kernel asks for,
and `mount_check` compares the two over every entry at once.
"""

import json
from pathlib import Path

SOURCE_FIELDS = ("model_sources", "dataset_sources")
"""The two lists a weights mount can live in, and it depends on what Kaggle calls the thing: DINOv2 is a
Kaggle *model* (`metaresearch/dinov2/PyTorch/small/1`), our DINOv3 snapshot is a private *dataset*
(`dimiturnt/dinov3-hf`). Both are read, because an encoder's mount states a path, not a kind."""


class KernelSources:
    """One kernel's declared attachments, and whether they cover what a package needs."""

    def __init__(self, metadata: Path) -> None:
        """The metadata file itself, not the kernel directory — a directory has two JSON files in it."""
        self.metadata = metadata
        declared = json.loads(metadata.read_text(encoding="utf-8"))
        self.declared = [str(source) for field in SOURCE_FIELDS for source in declared.get(field, [])]

    def covers(self, mount: str) -> bool:
        """Whether some declared source attaches this mount — by PREFIX, not equality.

        A model source names the mount exactly, while a dataset source names its ROOT and the encoder
        names a directory inside it: `dimiturnt/dinov3-hf` declared, `dimiturnt/dinov3-hf/vits16`
        needed. A path segment boundary is required so that a declared `dimiturnt/dinov3-hf` cannot be
        read as covering a differently-named `dimiturnt/dinov3-hf-v2/vits16`.
        """
        return any(mount == source or mount.startswith(f"{source}/") for source in self.declared)

    def undeclared(self, mounts: list[str]) -> list[str]:
        """The needed mounts this kernel has not declared — empty when the entry can actually run.

        Only this direction is an error. A kernel that declares MORE than the package needs costs some
        seconds of mount time and nothing else, and one entry's metadata is routinely reused for the
        next package.
        """
        return [mount for mount in mounts if not self.covers(mount)]
