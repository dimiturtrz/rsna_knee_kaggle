"""Every encoder's DECLARED normalization must be the one its own checkpoint states.

THE DEFECT THIS EXISTS FOR COST TWO DAYS AND ONE FALSE CONCLUSION. `StudyModel` normalised every trunk
against the ImageNet constants, which is what six of the registry's natural-image encoders really
declare — so the assumption was right often enough to stay invisible, and wrong exactly once, on the
one arm built to test whether a RADIOLOGY initialisation pays. RAD-DINO was fed inputs 0.20 std off its
pretraining distribution, trained a plausible curve, and produced a real number supporting a false
answer to the exact question it was queued to settle.

`Normalization.assert_declared` was written with that fix and then called by NOTHING, which left the
registry docstring's claim — "verified against each one's own config" — as prose. Prose is what failed
the first time. This is the call.

INTEGRATION, not unit, because it fetches each checkpoint's config from HuggingFace: a unit test that
reaches the network is exactly what `devtools.hygiene.small` forbids, and asking the real snapshot is
the whole point. Nothing here loads weights, so it costs one small JSON per encoder.
"""

import json
from pathlib import Path

import pytest
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from kneemri.models.backbone import ENCODERS, PREPROCESSOR_FILES, Encoder


def _declaring_document(encoder: Encoder) -> dict[str, object]:
    """The snapshot's own preprocessing config, whichever of its names this trunk ships it under."""
    for name in PREPROCESSOR_FILES:
        try:
            path = hf_hub_download(encoder.repository, name)
        except EntryNotFoundError:
            continue
        return json.loads(Path(path).read_text(encoding="utf-8"))
    message = f"{encoder.variant} ({encoder.repository}) ships none of {PREPROCESSOR_FILES}"
    raise AssertionError(message)


@pytest.mark.parametrize("variant", sorted(ENCODERS))
def test_declared_normalization_matches_the_checkpoint(variant: str) -> None:
    """A registry entry the checkpoint contradicts is refused, naming both sides."""
    encoder = ENCODERS[variant]
    encoder.normalization.assert_declared(_declaring_document(encoder), variant)


def test_every_encoder_states_its_own_constants() -> None:
    """No entry may be silently undeclared — an absent document reads as agreement otherwise.

    Separate from the parametrised check because it guards a DIFFERENT hole: that one asks whether the
    two numbers agree, and cannot fire at all for a trunk whose declaration was never found.
    """
    for variant, encoder in ENCODERS.items():
        document = _declaring_document(encoder)
        assert "image_mean" in document, f"{variant} states no image_mean"
        assert "image_std" in document, f"{variant} states no image_std"
