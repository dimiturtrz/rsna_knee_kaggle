"""A leaf carries its default and its spelling on one field."""

from dataclasses import MISSING, dataclass, fields

from kneemri.training.config_leaf import FLAG, ConfigLeaf


def test_of():
    """The default is the field's, and the help line plus any exception ride in the metadata."""

    @dataclass(frozen=True)
    class Node:
        knob: int = ConfigLeaf.of(default=3, says="how many", flag="--knobs")

    (leaf,) = fields(Node)

    assert Node().knob == 3
    assert leaf.default_factory is MISSING
    assert leaf.metadata[FLAG] == {"help": "how many", "flag": "--knobs"}
