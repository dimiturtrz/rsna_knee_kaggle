from pathlib import Path

import pytest

from kneemri.inference.foreign_source import ForeignModule, ForeignSource

KERNEL = """\
CROP_MM = 10_000.0
SLOTS = ["the training program's slots"]

N_SLOT_TYPES, MASK_IDX = (6, 0)

CROP_MM = 130.0
SLOTS = [("Sagittal", 1), ("Coronal", 0)]


def render(value):
    return value * CROP_MM


class Net:
    def slots(self):
        return SLOTS


preds = render(2.0)

CROP_MM = 1.0
SLOTS = ["a second family's slots"]
"""

SERVING = (6, 20)
"""The line range of the one program in KERNEL whose checkpoints are being served."""


def _source(tmp_path: Path, scope: tuple[int, int] | None = None) -> ForeignSource:
    kernel = tmp_path / "their_kernel.py"
    kernel.write_text(KERNEL, encoding="utf-8")
    return ForeignSource(kernel, scope=scope)


def test_tree(tmp_path: Path) -> None:
    """The kernel parses, and its top level is the statement list the lift cuts from."""
    assert len(_source(tmp_path).tree().body) == 10


def test_first_definitions(tmp_path: Path) -> None:
    """Unscoped, a symbol resolves to its first binding — which is a DIFFERENT program's constant."""
    definitions = _source(tmp_path).first_definitions()
    assert set(definitions) >= {"CROP_MM", "SLOTS", "N_SLOT_TYPES", "MASK_IDX", "render", "Net", "preds"}
    assert definitions["CROP_MM"].lineno == 1, "the training program's crop, because nothing said otherwise"
    assert definitions["MASK_IDX"].lineno == 4, "an unpacking assignment binds both of its names"


def test_in_scope(tmp_path: Path) -> None:
    """The scope picks the serving program's binding out of three, and neither neighbour is it.

    In `bend_the_knee_to_dinov3_ensembled.py` this is `CROP_MM` at 130 mm (a box around the knee) between
    a training binding and a second family's 10,000 mm — the whole image. Taking either neighbour feeds
    their model an input their model has never seen.
    """
    scoped = _source(tmp_path, SERVING).first_definitions()
    assert scoped["CROP_MM"].lineno == 6, "the serving binding, not the one above it nor the one below"
    assert "N_SLOT_TYPES" not in scoped, "a symbol outside the ported program is not this program's symbol"


def test_lift(tmp_path: Path) -> None:
    """The named symbols come out verbatim, in file order, and the other programs are left behind."""
    lifted = _source(tmp_path, SERVING).lift(["CROP_MM", "SLOTS", "render", "Net"])
    assert lifted.source.index("CROP_MM = 130.0") < lifted.source.index("def render")
    assert "10_000.0" not in lifted.source
    assert "second family" not in lifted.source
    assert "preds = render" not in lifted.source, "a top-level statement that RUNS is not a lifted symbol"


def test_start(tmp_path: Path) -> None:
    """A decorated definition is cut from its `@`, not from its `def` — `ast` puts `lineno` on the `def`.

    Their `_micro` is `@torch.no_grad()`. Lifted without it, their serving loop builds a graph and dies
    on `Can't call numpy() on Tensor that requires grad`; a decorator that changed values instead of
    raising would not have announced itself at all.
    """
    kernel = tmp_path / "decorated.py"
    kernel.write_text("import functools\n\n\n@functools.cache\ndef once(value):\n    return value\n", encoding="utf-8")
    lifted = ForeignSource(kernel, header="import functools\n").lift(["once"])
    assert "@functools.cache" in lifted.source
    assert lifted.build().once.__wrapped__ is not None, "the decorator is applied, not merely present"


def test_lift_missing(tmp_path: Path) -> None:
    """A name the ported program does not define means the kernel moved under the port; that is a fault."""
    with pytest.raises(KeyError, match="defines none of"):
        _source(tmp_path, SERVING).lift(["Net", "ViTSlotToken"])


def test_build(tmp_path: Path) -> None:
    """The lifted source executes, and their code computes what THEIR program's constants say it should."""
    module = _source(tmp_path, SERVING).lift(["CROP_MM", "SLOTS", "render", "Net"]).build()
    assert module.render(2.0) == 260.0, "130 mm, not the 10 m crop above it or the 1 mm below"
    assert module.Net().slots() == [("Sagittal", 1), ("Coronal", 0)]


def test_inject() -> None:
    """The globals their author filled from a Kaggle mount are bound by the caller, and resolve at CALL time."""
    module = ForeignModule(name="lifted", source="def serve():\n    return LABELS\n").build()
    ForeignModule.inject(module, LABELS=["ACL", "MCL"])
    assert module.serve() == ["ACL", "MCL"]
