"""One training knob as a dataclass field that also says how the command line spells it.

Its own module, apart from `config_tree`, so a node can be declared beside the module that consumes it
(`augmentation`, `sampling`, `teacher_targets`, `adaptation`) without that module importing the root
that imports it back.
"""

from collections.abc import Sequence
from dataclasses import field
from typing import TypedDict, Unpack

FLAG = "argparse"
"""The metadata key a leaf's spelling lives under; a field without it has no flag of its own."""


class Spelling(TypedDict, total=False):
    """What argparse needs beyond a leaf's name, type and default — the exceptions, never the rule."""

    help: str | None
    flag: str
    action: str
    choices: Sequence[str]
    required: bool


class ConfigLeaf:
    """A field's default, its help line and whatever argparse needs beyond the type — one declaration.

    The flag used to be declared a second time in `FlagVocabulary`, which is how a default came to have
    two homes and a test existed only to assert they agreed.
    """

    @staticmethod
    def of[T](*, default: T, says: str | None = None, **spelling: Unpack[Spelling]) -> T:
        """A field defaulting to `default` whose flag prints `says`; the rest of `Spelling` passes through."""
        return field(default=default, metadata={FLAG: {**spelling, "help": says}})
