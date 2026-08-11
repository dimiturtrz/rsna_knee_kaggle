"""The anatomical prior: does the table say what knee MRI says?

These assert radiology, not arithmetic. If a change makes the ACL prefer an axial sequence over a
sagittal one, the table is wrong however cleanly it computes.
"""

from kneemri.anatomy import ANATOMY, PRIMARY, SECONDARY, FindingAnatomy, SlotAffinity
from kneemri.model import TARGETS
from kneemri.slots import SLOTS, Slot

SAG_FS = Slot("SAG_FLUID_FS", "Sagittal", fluid_sensitive=True, fat_suppressed=True)
COR_T1 = Slot("COR_T1", "Coronal", fluid_sensitive=False, fat_suppressed=False)
AX_FS = Slot("AX_FLUID_FS", "Axial", fluid_sensitive=True, fat_suppressed=True)


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
