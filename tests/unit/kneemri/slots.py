from pathlib import Path

from kneemri.acquisition import Acquisition
from kneemri.slots import LAST_MAXIMAL, SLOTS, Routing, Slot, SlotScheme

from .conftest import write_series


def _acquisition(tmp_path: Path, name: str, **parts: object) -> Acquisition:
    directory = write_series(tmp_path / name, **parts)  # pyrefly: ignore[bad-argument-type]
    return Acquisition.probe("study", name, directory)


def test_accepts(tmp_path: Path):
    """A slot's predicate is all three of plane, fat suppression and weighting — not any two."""
    fluid_fs = _acquisition(tmp_path, "a")
    slot = Slot("SAG_FLUID_FS", "Sagittal", True, True)
    assert slot.accepts(fluid_fs, "Sagittal")
    assert not slot.accepts(fluid_fs, "Coronal")
    assert not Slot("SAG_T1", "Sagittal", False, False).accepts(fluid_fs, "Sagittal")


def test_assign(tmp_path: Path):
    """Each series lands in the slot its predicate matches, and ties go to the denser stack."""
    thin = _acquisition(tmp_path, "thin", slices=6)
    thick = _acquisition(tmp_path, "thick", slices=20)
    t1 = _acquisition(tmp_path, "t1", description="COR T1", repetition=500.0, echo=10.0, scan_options="")
    planes = {"thin": "Sagittal", "thick": "Sagittal", "t1": "Coronal"}
    chosen = SlotScheme().assign([thin, thick, t1], planes)
    assert chosen["SAG_FLUID_FS"].series == "thick", "the denser stack wins the tie"
    assert chosen["COR_T1"].series == "t1"
    assert "AX_FLUID_FS" not in chosen, "a slot with no matching series stays empty"


def test_assign_admits_no_substitute_for_an_empty_slot(tmp_path: Path):
    """Relaxing a predicate would put one series in two slots and make the mask assert an acquisition."""
    fluid = _acquisition(tmp_path, "fluid", description="SAG PD", scan_options="")
    chosen = SlotScheme().assign([fluid], {"fluid": "Sagittal"})
    assert set(chosen) == {"SAG_FLUID_NOFS"}
    assert "SAG_T1" not in chosen


def test_unrouted(tmp_path: Path):
    """A series matching no slot predicate is reported as lost, and a routed one is not.

    The scheme has no AX_T1, AX_FLUID_NOFS or COR_FLUID_NOFS, so an axial T1 is invisible to the
    model - `assign` cannot say so, because a slot with no candidate just stays empty.
    """
    axial_t1 = _acquisition(tmp_path, "ax_t1", description="AX T1", repetition=500.0, echo=10.0, scan_options="")
    sagittal = _acquisition(tmp_path, "sag")
    planes = {"ax_t1": "Axial", "sag": "Sagittal"}

    lost = SlotScheme().unrouted([axial_t1, sagittal], planes)

    assert [acquisition.series for acquisition in lost] == ["ax_t1"]


def test_contested(tmp_path: Path):
    """A slot is contested when two candidates share the top slice count, and clear when one leads.

    The tie is the whole point: `assign` resolves it by list order and the published members resolved
    it by frame order, so a contested slot is where two pipelines can put different images.
    """
    same_a = _acquisition(tmp_path, "same_a", slices=12)
    same_b = _acquisition(tmp_path, "same_b", slices=12)
    planes = {"same_a": "Sagittal", "same_b": "Sagittal"}
    assert SlotScheme().contested([same_a, same_b], planes)["SAG_FLUID_FS"] == 1

    clear = _acquisition(tmp_path, "clear", slices=20)
    planes["clear"] = "Sagittal"
    assert SlotScheme().contested([same_a, clear], planes)["SAG_FLUID_FS"] == 0
    assert SlotScheme().contested([same_a, same_b], planes)["AX_FLUID_FS"] == 0, "an empty slot is not contested"


def test_presence(tmp_path: Path):
    """The mask is one flag per slot, in the fixed slot order the head's embedding assumes."""
    fluid = _acquisition(tmp_path, "fluid")
    mask = SlotScheme().presence(SlotScheme().assign([fluid], {"fluid": "Sagittal"}))
    assert len(mask) == len(SLOTS)
    assert mask[0] == 1.0
    assert sum(mask) == 1.0


def test_won(tmp_path: Path):
    """Which of several equally dense candidates fills a slot — the open question, as a strategy.

    Our residual against a published member drops from max|diff| 0.128 to 0.025 on studies with a
    contested slot when the LAST maximal is taken instead of the first, so their router is not the
    first-maximal rule ours is. `first` stays the default because every cache and member on disk was
    built with it.
    """
    dense_a = _acquisition(tmp_path, "a", slices=12)
    dense_b = _acquisition(tmp_path, "b", slices=12)
    thin = _acquisition(tmp_path, "c", slices=4)
    candidates = [dense_a, thin, dense_b]

    assert SlotScheme().won(candidates) is dense_a, "the default keeps the first of the densest"
    assert SlotScheme(tie=LAST_MAXIMAL).won(candidates) is dense_b
    assert SlotScheme(tie=LAST_MAXIMAL).won([thin, dense_a]) is dense_a, "density still outranks the rule"


def test_suffix():
    """What a routing adds to a cache's name — nothing for the historic one, so 99 GB stays findable."""
    assert Routing().suffix() == ""
    assert Routing(tie=LAST_MAXIMAL).suffix() == "_last"
    assert Routing(scheme="extended").suffix() == "_extended"
    assert Routing(scheme="extended", tie=LAST_MAXIMAL).suffix() == "_extended_last"


def test_built():
    """The routing builds the scheme it names, carrying its tie rule rather than dropping it."""
    built = Routing(tie=LAST_MAXIMAL).built()

    assert built.tie == LAST_MAXIMAL
    assert [slot.name for slot in built.slots] == [slot.name for slot in SLOTS]
    assert len(Routing(scheme="extended").built().slots) > len(SLOTS), "the named scheme, not a default"
