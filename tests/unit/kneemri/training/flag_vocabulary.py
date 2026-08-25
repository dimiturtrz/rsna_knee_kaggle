"""What argv is allowed to say, asserted against the value it lands in — the two halves of one rule."""

import dataclasses

from kneemri.training.flag_vocabulary import FlagVocabulary
from kneemri.training.options import TrainingOptions


def test_parser():
    """Every field is reachable from argv, and every flag lands on a field.

    A field with no flag is a knob only a test can turn; a flag with no field would raise on `parse`,
    which is loud — the silent half is the first, and it is the half that queues an arm whose recipe
    cannot be asked for. This is HERE rather than beside the dataclass because the vocabulary is what
    the assertion is about; the dataclass is the ruler it is held against.
    """
    dests = set(vars(FlagVocabulary.parser().parse_args(["--labels", "labels.csv"])))

    assert dests == {field.name for field in dataclasses.fields(TrainingOptions)}
