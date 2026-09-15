"""Every config key a consumer reads back must be a key a fresh checkpoint actually stamps.

THE NINE DEFECTS THIS PROJECT HAS PAID FOR ARE ALL ONE SHAPE: a value recorded in a checkpoint's
config while a consumer reads a default. `MemberRecord`'s own walk guards the writing end — one option
moves, exactly the stamped field of that name follows. This guards the READING end, which no unit test
of either side can see: a consumer asking for a key nobody stamps gets its default, silently, forever.

Scraped rather than listed, because a hand-listed set of served keys would be a fourth transcription of
the same vocabulary — and the transcriptions are the defect.
"""

import ast
from pathlib import Path

import torch

from kneemri.models.backbone import Encoder
from kneemri.runs.member_record import MemberRecord
from kneemri.training.config_tree import TrainConfig

from ..unit.kneemri.models.model import SLOTS, _FakeBackbone  # the stand-in encoder this suite already trusts

DIM = 8
PACKAGE = Path(__file__).resolve().parents[2] / "kneemri"
NOT_A_CHECKPOINT_KEY = {"config"}
"""`saved.get("config", {})` reaches for the block itself, which is stamped one level up."""


def _reads_a_config(node: ast.expr) -> bool:
    """Is this receiver a checkpoint config block, however it was reached?"""
    if isinstance(node, ast.Name):
        return "config" in node.id
    if isinstance(node, ast.Attribute):
        return "config" in node.attr or _reads_a_config(node.value)
    if isinstance(node, ast.Subscript):
        return _reads_a_config(node.value)
    if isinstance(node, ast.Call):
        reaches = node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "config"
        return bool(reaches) or (isinstance(node.func, ast.Attribute) and _reads_a_config(node.func.value))
    return False


def _keys_read_by(module: Path) -> set[str]:
    keys = set()
    for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            continue
        if _reads_a_config(node.func.value):
            keys.add(node.args[0].value)
    return keys - NOT_A_CHECKPOINT_KEY


def _stamped() -> set[str]:
    from kneemri.models.model import StudyModel  # noqa: PLC0415 - keeps the fake backbone import local

    torch.manual_seed(2026)
    record = MemberRecord(
        TrainConfig().stamp,
        Encoder(variant="small", repository="facebook/dinov2-small", dim=DIM),
        StudyModel(_FakeBackbone(), DIM, SLOTS),
        torch.device("cpu"),
        [f"S{index}" for index in range(SLOTS)],
    )
    payload = record.payload({"w": torch.zeros(1)}, holdout=0.8, epoch=0)
    config = payload["config"]
    assert isinstance(config, dict)
    return set(config)


def test_every_served_config_key_is_stamped_by_a_fresh_checkpoint():
    """A key read with a default and never written is a member served under a rule it was not fitted by.

    That is not hypothetical here: `lora_rank`, `slot_pooling` and `band` each reached a consumer
    before they reached the stamp, and the first two of those cost a member apiece. The fingerprint
    cannot catch it — it is checked against the member's OWN reading, so a wrong-but-consistent rebuild
    passes.
    """
    stamped = _stamped()
    read = {module: _keys_read_by(module) for module in PACKAGE.rglob("*.py")}
    read = {module: keys for module, keys in read.items() if keys}

    assert read, "the scrape found no consumer at all, which means it stopped seeing them, not that they stopped"

    unstamped = {
        module.relative_to(PACKAGE).as_posix(): sorted(keys - stamped)
        for module, keys in read.items()
        if keys - stamped
    }
    assert not unstamped, "these consumers read a key no fresh checkpoint writes, so every new member takes its default"
