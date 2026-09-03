"""What a vision model is asked about the ACL, and the vocabulary its answer must use.

Separate from both the transport and the scoring so the question can be read and diffed without a
server running and without a cache on disk. Everything here is the WORDING and the scale it maps
onto — nothing in this module knows how an image reaches a model or how an answer becomes a number.
"""

import re

from core.findings import CRITERIA

ACL = "ACL"
"""The one finding this probe reads. Chosen because it is the finding the first run tested, so the two
readings are about the same thing, and because it is the finding a sagittal fluid-sensitive series is
most nearly sufficient for."""

GRADES = ("intact", "possible", "torn")
"""The vocabulary the answer must use. Three rungs rather than two, so a model that is uncertain has
somewhere to say so instead of being forced into a coin flip that then reads as blindness."""

SCORE_OF = {"intact": 0.0, "possible": 0.5, "torn": 1.0}
"""Each grade as the number AUC is computed over. Ordinal and evenly spaced — the probe has no
evidence for any other spacing, and inventing one would be fitting the ruler to the result."""

CALLED_POSITIVE = 0.75
"""The threshold the confusion matrix is cut at: `torn` is a positive call, `possible` is not.

Stated as a constant rather than buried in a comparison because it is a CHOICE, and the graded scores
are written out so the same run can be re-cut at another threshold without re-querying the model."""

USER_TURN = "Grade the ACL."
"""The user turn, identical for every model. Constant rather than a literal because the served and the
locally loaded readers build their turns separately and the two must not be allowed to drift."""

JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
"""What an answer is dug out of. The prompt asks for a bare object; models wrap it in prose anyway."""


class AclPrompt:
    """What the model is asked, and the only place the task is stated in words.

    It states the competition's own criterion rather than a paraphrase, because the answer is scored
    against a label defined by that criterion.

    IT NAMES NO STUDY AND NO PRIOR. The run is blinded: nothing in these words tells the model which
    case it is looking at, what the base rate is, or what the other lane found. A prompt that leaked
    any of the three would produce a number about the prompt.
    """

    def system(self) -> str:
        """The reader's role, the criterion it applies, and the JSON it must answer in."""
        grades = ", ".join(GRADES)
        return (
            "You are reading a knee MRI. The image is a grid of consecutive sagittal fluid-sensitive "
            "fat-suppressed slices from one study, in order, left to right then top to bottom.\n\n"
            "Judge ONE structure: the anterior cruciate ligament.\n\n"
            f"The criterion is: {CRITERIA[ACL]}\n\n"
            f'Answer with a single JSON object: {{"acl": one of [{grades}], "because": one short '
            "sentence naming what you saw in the image}. No other text.\n\n"
            "Judge only what is visible. Do not assume a knee sent for MRI is injured."
        )

    def turns(self, image: dict[str, object]) -> list[dict[str, object]]:
        """The two chat turns, with the montage attached to the user turn.

        The image PART is passed in rather than built here because a served model wants a data URL
        under `image_url` and a locally loaded one wants a bare `{"type": "image"}` placeholder beside
        the pixels. Everything else about the two calls — the role split, the wording, the order — has
        to be byte-identical or the models are not being asked the same question, which is the whole
        point of running more than one.
        """
        return [
            {"role": "system", "content": [{"type": "text", "text": self.system()}]},
            {"role": "user", "content": [image, {"type": "text", "text": USER_TURN}]},
        ]
