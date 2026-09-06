"""The decode bill: what one shared reading costs, and what moving a single member costs instead.

These four tests are the mirror of `kneemri/inference/read_groups.py`. They build on the `reading`
fixture rather than on a whole `Ensemble`, because a regroup is a statement about ONE member's config
and asserting it through a served ensemble would be testing the caller.
"""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from kneemri.inference.members import RESIZED_FROM, Member, MemberConfig
from kneemri.inference.read_groups import ReadGroups, SharedRead


@pytest.fixture
def fitted(reading: Callable[[Path], Member], tmp_path: Path) -> Member:
    """A member as its trainer recorded it: twelve slices on the 0.2-0.8 band."""
    return reading(tmp_path)


def test_shared_read_regrouped(fitted: Member):
    """A re-group moves the DECODE — depth and band — so the member reads from another cache entirely.

    That is the whole point: the entry's decode bill is its number of read groups, and two members only
    share a decode when both of these agree. Sizes collapse inside a group and do not.
    """
    priced = SharedRead(20, (0.1, 0.9)).regrouped(fitted)

    assert priced.slices == 20, "the read opens twenty files per study, not twelve"
    assert priced.decode_key != fitted.decode_key, "which is a different decode"
    assert RESIZED_FROM in priced.config, "so the trainer's own array is refused for it"


def test_read_groups_regrouped(fitted: Member):
    """A partial regroup moves ONLY the members it names, and leaves every other one as fitted.

    The whole reason this type exists beside `SharedRead` is that the champion's three read groups can
    become two by moving one member, and `SharedRead` can only make them one. A mapping that swept up
    unnamed members would be the same all-or-nothing knob under another name.
    """
    deeper: MemberConfig = {**fitted.config, "slices": fitted.slices * 2}
    other = replace(fitted, identifier="deep", config=deeper)
    groups = ReadGroups({"deep": SharedRead(fitted.slices, (0.2, 0.8))})

    assert groups.regrouped(fitted) is fitted, "a member nobody named is served exactly as it was fitted"
    assert groups.regrouped(other).slices == fitted.slices, "the named one moves onto the shared reading"


def test_readings(fitted: Member):
    """The count is of DISTINCT readings, not of moved members — that is what the decode is billed by."""
    one = SharedRead(fitted.slices, (0.2, 0.8))

    assert ReadGroups({"a": one, "b": one}).readings == 1, "two members onto one reading is one decode"
    assert ReadGroups({"a": one, "b": SharedRead(20, (0.1, 0.9))}).readings == 2, "two destinations, two"


def test_beside(tmp_path: Path):
    """A re-grouped record carries BOTH axes in its name, because either alone would collide.

    The same member priced at 12 slices on two different bands is two different reads and two different
    numbers; one name for both would silently overwrite the first with the second.
    """
    checkpoint = tmp_path / "member.pt"

    destination = SharedRead(12, (0.1, 0.9)).beside(checkpoint)

    assert destination == tmp_path / "member_n12b0.1-0.9.pt"
    assert destination != SharedRead(12, (0.2, 0.8)).beside(checkpoint), "the band is part of the read"
