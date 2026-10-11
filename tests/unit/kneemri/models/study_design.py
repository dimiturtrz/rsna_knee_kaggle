import pytest

from kneemri.models.encoder import IMAGENET
from kneemri.models.slice_pooling import ATTEND
from kneemri.models.study_design import StudyDesign
from kneemri.models.token_pooling import CLS_MEAN_FOCAL
from kneemri.models.variants import RAD_DINO


def test_from_config():
    """Every switch a run records comes back, with the defaults spelled in one place.

    The equivalence classes a config falls into: empty (every published member, which predates all of
    these keys and must come back as the design it was fitted under), and one that states each switch
    away from its default. `pool` is in here because the ranking tool read every switch but that one.
    """
    assert StudyDesign.from_config({}) == StudyDesign(), "an old manifest means what it was fitted with"

    stated = StudyDesign.from_config(
        {
            "pool": CLS_MEAN_FOCAL,
            "slice_pooling": ATTEND,
            "slot_pooling": "mean",
            "anatomical_prior": True,
            "encoder_layer": -4,
        }
    )
    assert stated == StudyDesign(
        pool=CLS_MEAN_FOCAL, slice_pooling=ATTEND, slot_pooling="mean", anatomical_prior=True, layer=-4
    )


def test_lora_rank_of():
    """Zero for every manifest predating adaptation, and zero however the key spells "none".

    Those members carry no adapter keys and must be built without them; a stored `null` or an empty
    string means the same thing as an absent key, and reading either as a rank would wrap an encoder
    whose weights have no adapters to load into.
    """
    assert StudyDesign.lora_rank_of({}) == 0
    assert StudyDesign.lora_rank_of({"lora_rank": None}) == 0
    assert StudyDesign.lora_rank_of({"lora_rank": 8}) == 8


def test_normalization_of():
    """The registered variant's own constants; ImageNet's for an absent or unregistered variant."""
    rad = StudyDesign.normalization_of({"variant": RAD_DINO})
    assert list(rad.mean) == pytest.approx([0.5307] * 3), "the mount's own constants, not ImageNet's"
    assert list(rad.std) == pytest.approx([0.2583] * 3)
    assert StudyDesign.normalization_of({}) == IMAGENET, "an unnamed variant is the default encoder's"
    assert StudyDesign.normalization_of({"variant": "no-such-encoder"}) == IMAGENET
