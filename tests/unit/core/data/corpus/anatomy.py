"""The anatomical prior: does the table say what knee MRI says?

These assert radiology, not arithmetic. If a change makes the ACL prefer an axial sequence over a
sagittal one, the table is wrong however cleanly it computes.
"""

import math

import pytest

from core.data.corpus.anatomy import ANATOMY, PRIMARY, SECONDARY, Extent, FindingAnatomy, ScaleProfile, SlotAffinity
from core.data.corpus.slots import SLOTS, Slot
from core.findings import TARGETS

SAG_FS = Slot("SAG_FLUID_FS", "Sagittal", fluid_sensitive=True, fat_suppressed=True)
COR_T1 = Slot("COR_T1", "Coronal", fluid_sensitive=False, fat_suppressed=False)
AX_FS = Slot("AX_FLUID_FS", "Axial", fluid_sensitive=True, fat_suppressed=True)


def test_anatomy_keys_are_the_competition_targets():
    """The table is keyed by exactly the twelve findings; a missing one would read a flat prior silently."""
    assert set(ANATOMY) == set(TARGETS)


def test_affinity():
    """A primary matched plane outranks a secondary one, and an unrelated plane scores nothing."""
    acl = FindingAnatomy(primary=("Sagittal",), secondary=("Coronal",))
    assert acl.affinity(SAG_FS) == PRIMARY
    assert acl.affinity(COR_T1) == SECONDARY * 0.5
    assert acl.affinity(AX_FS) == 0.0


def test_affinity_penalises_a_bone_finding_without_fat_suppression():
    """Marrow oedema needs fat suppression, so a contusion on a non-suppressed sequence is discounted."""
    contusion = ANATOMY["Contusion"]
    suppressed = contusion.affinity(SAG_FS)
    plain = contusion.affinity(Slot("SAG_FLUID_NOFS", "Sagittal", fluid_sensitive=True, fat_suppressed=False))
    assert plain < suppressed


def test_affinity_does_not_penalise_cartilage_for_being_on_t1():
    """Osteoarthritis is morphology, so T1 is the RIGHT sequence for it rather than a compromise."""
    assert ANATOMY["Medial OA"].affinity(COR_T1) > ANATOMY["Medial OA"].affinity(SAG_FS)


def test_matrix():
    """One row per finding, one column per slot, each finding's best slot being where it is read."""
    matrix = SlotAffinity(ANATOMY).matrix(TARGETS, SLOTS)
    assert matrix.shape == (len(TARGETS), len(SLOTS))
    best = {target: SLOTS[int(row.argmax())].plane for target, row in zip(TARGETS, matrix, strict=True)}
    assert best["ACL"] == "Sagittal"
    assert best["MCL"] == "Coronal"
    assert best["PF OA"] == "Axial"
    assert best["Medial OA"] == "Coronal"


def test_matrix_gives_an_unknown_finding_a_flat_row():
    """A thirteenth target must not be silently masked out of every slot."""
    row = SlotAffinity(ANATOMY).matrix(("Chondrocalcinosis",), SLOTS)
    assert float(row.min()) == float(row.max()) == 0.0


def test_requested():
    """The three answers a caller can get: the table, nothing, or a refusal — never a silent mismatch."""
    affinity = SlotAffinity(ANATOMY)

    asked = affinity.requested(TARGETS, len(SLOTS), wanted=True)
    assert asked is not None
    assert asked.shape == (len(TARGETS), len(SLOTS))

    assert affinity.requested(TARGETS, len(SLOTS), wanted=False) is None

    with pytest.raises(ValueError, match="slots"):
        affinity.requested(TARGETS, len(SLOTS) - 1, wanted=True)


def test_of():
    """The deltas split by structure scale, and MEAN — not a finding — is not averaged into either."""
    profile = ScaleProfile.of({"ACL": 0.02, "Medial Meniscus": 0.04, "Effusion": -0.02, "MEAN": 0.9})

    assert profile.focal == pytest.approx(0.03)
    assert profile.regional == pytest.approx(-0.02)


def test_of_reports_a_side_no_arm_scored_as_nan():
    """An empty side is nan rather than zero: unread is not 'moved nothing'."""
    profile = ScaleProfile.of({"ACL": 0.02})

    assert math.isnan(profile.regional)


def test_verdict():
    """Each of the four patterns is named, and only the two-sided one supports the multi-scale build."""
    assert "TRADED" in ScaleProfile(focal=0.02, regional=-0.01).verdict()
    assert "UNIFORM" in ScaleProfile(focal=0.02, regional=0.01).verdict()
    assert "INVERTED" in ScaleProfile(focal=-0.02, regional=0.01).verdict()
    assert "REFUTED" in ScaleProfile(focal=-0.02, regional=-0.01).verdict()


def test_line():
    """The bare line carries both means and NO verdict — it describes any arm, whatever it moved."""
    line = ScaleProfile(focal=0.02, regional=-0.01).line()

    assert "focal +0.0200" in line
    assert "regional -0.0100" in line
    assert "TRADED" not in line, "the verdict is a claim about the view, and this line makes no claim"


def test_judged():
    """The judged form adds the pattern, for the arms that earned a statement about the view."""
    judged = ScaleProfile(focal=0.02, regional=-0.01).judged()

    assert "focal +0.0200" in judged
    assert "TRADED" in judged


def test_extent_is_assigned_from_the_structure_and_not_from_a_measured_result():
    """The millimetre-scale structures are focal and the fluid collections are regional.

    Cartilage is asserted focal DELIBERATELY: our annotation study found osteoarthritis
    label-bottlenecked, and a grouping edited to match that would agree with any future arm.
    """
    focal = {name for name, finding in ANATOMY.items() if finding.extent is Extent.FOCAL}

    assert focal == {"ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA"}
