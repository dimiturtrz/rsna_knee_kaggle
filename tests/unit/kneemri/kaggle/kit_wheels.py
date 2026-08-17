"""What a kit must carry, and the two ways it can fail to."""

from pathlib import Path

from kneemri.kaggle.kit_wheels import REQUIRED, KitWheels


def _kit(into: Path, *wheels: str) -> Path:
    """A kit directory holding exactly these wheel filenames."""
    into.mkdir(parents=True, exist_ok=True)
    for wheel in wheels:
        (into / wheel).write_bytes(b"not a real wheel, and nothing here unpacks one")
    return into


def test_major():
    """The version is the wheel filename's second field, and an unparseable one is not a version."""
    assert KitWheels().major(Path("transformers-5.15.0-py3-none-any.whl")) == 5
    assert KitWheels().major(Path("safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.whl")) == 0
    assert KitWheels().major(Path("something.whl")) == 0, "a name with no version cannot claim to satisfy a floor"


def test_missing(tmp_path: Path):
    """Absent is one failure and TOO OLD is the other — the second is why this file exists.

    A kit carrying transformers 4 imports fine and rebuilds DINOv3 under the older nesting, so its v3
    members are dropped inside the kernel rather than at import; the wheel is present the whole time.
    """
    complete = _kit(
        tmp_path / "complete",
        "jaxtyping-0.3.11-py3-none-any.whl",
        "wadler_lindig-0.1.7-py3-none-any.whl",
        "transformers-5.15.0-py3-none-any.whl",
        "safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.whl",
    )

    assert KitWheels().missing(complete) == [], "every requirement met, nothing to say"

    short = _kit(tmp_path / "short", "jaxtyping-0.3.11-py3-none-any.whl", "wadler_lindig-0.1.7-py3-none-any.whl")
    complaints = KitWheels().missing(short)

    assert len(complaints) == 2, "both wheels that are not there are named, not the first one"
    assert any("transformers" in line for line in complaints)

    old = _kit(
        tmp_path / "old",
        "jaxtyping-0.3.11-py3-none-any.whl",
        "wadler_lindig-0.1.7-py3-none-any.whl",
        "transformers-4.44.0-py3-none-any.whl",
        "safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.whl",
    )
    aged = KitWheels().missing(old)

    assert len(aged) == 1
    assert "too old" in aged[0] and "4.44.0" in aged[0], "the version it carries, so the fix is one download"


def test_required(tmp_path: Path):
    """An underscore-named distribution counts under the dash spelling its wheel is published with."""
    assert REQUIRED["wadler_lindig"] == 0, "no floor: any version of jaxtyping's own dependency will do"
    dashed = _kit(
        tmp_path,
        "jaxtyping-0.3.11-py3-none-any.whl",
        "wadler-lindig-0.1.7-py3-none-any.whl",
        "transformers-5.15.0-py3-none-any.whl",
        "safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.whl",
    )

    assert KitWheels().missing(dashed) == [], "wadler-lindig is wadler_lindig, and a kit is not rebuilt over a hyphen"
