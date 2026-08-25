"""The bootstrap each kernel carries must be the canonical one, because a kernel is what a slot runs.

The gate is a test rather than a nox session: `noxfile.py` is template-owned and has no local slot to
extend, while the suite already runs on every commit and in CI. The failure it guards is not
hypothetical — the 248-second mount walk had to be fixed in seven files, and a kernel that missed it
would have cost a submission slot to discover.
"""

from pathlib import Path

from comp.build_kernels import CLOSE, KERNELS, OPEN, Bootstrap


def _kernels() -> list[Path]:
    return sorted(KERNELS.glob("*/*.py"))


def test_every_kernel_carries_the_canonical_bootstrap():
    """No kernel's copy may differ from `comp/kernel_bootstrap.py` — that is what the copies cost.

    The builder is the fix: `python -m comp.build_kernels` rewrites every marked region, so a drifted
    kernel is a file nobody regenerated rather than a merge to resolve by hand.
    """
    bootstrap = Bootstrap()
    drifted = [
        kernel.parent.name
        for kernel in _kernels()
        if bootstrap.marked(kernel) and not bootstrap.apply(kernel, write=False)
    ]
    assert not drifted, f"{drifted} drifted from comp/kernel_bootstrap.py — run python -m comp.build_kernels"


def test_a_stray_import_in_the_region_refuses_the_splice():
    """Code that a tool moved into the region must stop the build, not be overwritten by it.

    The region is regenerated, so anything sitting in it that the canonical text lacks is deleted on
    the next run. `ruff check --select I --fix` puts code there without anyone asking: the OPEN marker
    reads as a comment on the imports below it, so a kernel's own imports can be sorted down across it.
    Written with the canonical text itself so it keeps testing the real region rather than a fixture.
    """
    bootstrap = Bootstrap()
    moved = f"{OPEN}\nimport pandas as pd\n{bootstrap.text}{CLOSE}\n"
    assert bootstrap.strays(moved) == ["import pandas as pd"]
    assert bootstrap.strays(f"{OPEN}{bootstrap.text}{CLOSE}\n") == []


def test_every_kernel_that_finds_the_kit_takes_it_from_the_region():
    """A kernel spelling its own `find_kit` is the duplication coming back, so it must sit in the region.

    Checked by TEXT rather than by import: these files run on Kaggle before our package exists, so
    nothing here can import them, and a kernel that quietly grew a second definition beside the region
    would satisfy the drift test above while running its own walk.
    """
    for kernel in _kernels():
        text = kernel.read_text(encoding="utf-8")
        if "def find_kit" not in text:
            continue
        assert text.count("def find_kit") == 1, f"{kernel.parent.name} defines find_kit twice"
        assert text.find(OPEN) < text.find("def find_kit") < text.find(CLOSE), (
            f"{kernel.parent.name} defines find_kit outside the canonical region"
        )
