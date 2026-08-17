"""Which wheels a kit MUST carry, and why each one decides whether a member loads at all.

Kaggle runs offline, so a kit's wheels are not a convenience — they are part of the runtime a member is
rebuilt under. Two kinds live here and they fail differently: a wheel our own source IMPORTS (the kernel
dies at import, loudly, in the first minute) and a wheel that pins a VERSION Kaggle already has (nothing
fails; a checkpoint is rebuilt against a different module tree and its member is dropped mid-run).

The second kind cost an entry on 2026-08-17. `build_kit` was pointed at a scratch wheels directory
holding only the two jaxtyping wheels, so transformers 5.15.0 went missing; Kaggle's own older
transformers builds DINOv3 as `backbone.layer.*` where ours builds `backbone.model.layer.*`, and
`kneemri-encoder` — an entry whose whole design was its one DINOv3 member — dropped that member and
wrote the three-member control under the encoder entry's name. It scored, so nothing looked wrong.

One home for the requirement because both the builder and the pre-push verifier must ask the same
question: a checker the build path does not run is a checker that is skipped by whoever uploads by hand.
"""

from pathlib import Path

REQUIRED = {"jaxtyping": 0, "wadler_lindig": 0, "transformers": 5, "safetensors": 0}
"""Wheel name → the major version the kit must carry at least; 0 when any version does.

Probed from a kernel, not guessed: pydicom, timm, transformers, polars, cv2, SimpleITK, einops and
sklearn are all present in Kaggle's image, and `jaxtyping` is not — which `kneemri.models.model` imports
at module top. `wadler_lindig` is jaxtyping's own dependency. `transformers` is here for its VERSION and
`safetensors` because that transformers needs it to read the mounted encoders.
"""


class KitWheels:
    """The requirement, asked of one built kit."""

    def __init__(self, required: dict[str, int] | None = None) -> None:
        self.required = REQUIRED if required is None else required

    def major(self, wheel: Path) -> int:
        """The major version in a wheel's filename, or 0 when it does not start with a number.

        Filename rather than metadata: a kit is a directory of files and this must answer without
        unpacking one. Wheel names are `name-version-...`, so the version is the second field.
        """
        version = wheel.name.split("-")[1] if "-" in wheel.name else ""
        first = version.split(".")[0]
        return int(first) if first.isdigit() else 0

    def missing(self, kit: Path) -> list[str]:
        """Every required wheel the kit lacks or carries too old a copy of, one sentence each."""
        problems: list[str] = []
        for name, floor in self.required.items():
            # The dash/underscore spelling is not ours to choose: a distribution named with an
            # underscore is published under either, so the glob has to accept both.
            found = sorted(kit.glob(f"{name.replace('_', '?')}-*.whl"))
            if not found:
                problems.append(f"missing wheel: {name} (the kernel fails at import, or rebuilds a member wrong)")
            elif floor and all(self.major(wheel) < floor for wheel in found):
                carried = ", ".join(wheel.name for wheel in found)
                problems.append(f"wheel too old: {name} needs major {floor}, kit carries {carried}")
        return problems
