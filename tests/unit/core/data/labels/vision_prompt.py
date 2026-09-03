"""The wording the probe puts to a model, checked without a server or a cache.

A prompt is the one part of this instrument that can be wrong silently: a missing grade word or a
dropped criterion produces answers that parse cleanly and mean something else.
"""

from core.data.labels.vision_prompt import ACL, CALLED_POSITIVE, GRADES, SCORE_OF, AclPrompt


def test_system() -> None:
    """The prompt names the finding, its published criterion and the three legal answers."""
    spoken = AclPrompt().system()
    assert "anterior cruciate ligament" in spoken
    for grade in GRADES:
        assert grade in spoken


def test_turns() -> None:
    """Two turns, with whatever image part the reader supplied attached to the user turn."""
    turns = AclPrompt().turns({"type": "image"})
    assert [turn["role"] for turn in turns] == ["system", "user"]
    parts: list[dict[str, object]] = list(turns[1]["content"])  # type: ignore[call-overload]
    assert [part["type"] for part in parts] == ["image", "text"]


def test_acl_is_the_finding() -> None:
    """The probe reads the column the competition calls ACL, spelled the same way."""
    assert ACL == "ACL"


def test_the_cut_puts_possible_on_the_negative_side() -> None:
    """`torn` is a positive call and `possible` is not — the choice the matrix is read through."""
    assert SCORE_OF["torn"] >= CALLED_POSITIVE
    assert SCORE_OF["possible"] < CALLED_POSITIVE
