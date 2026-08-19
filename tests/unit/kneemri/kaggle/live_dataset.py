"""What the live Kaggle dataset holds, parsed from the CLI listing without touching the network."""

import json
from pathlib import Path

from kneemri.kaggle.live_dataset import LiveDataset

LISTING = "name,size,creationDate\nv2member.pt,89,2026-08-19 05:30:55\nmanifest.json,12,2026-08-19 05:30:55\n"


def test_members():
    """The listing is CSV with a header, and only the `.pt` rows name members."""
    live = LiveDataset("dimiturnt/kneemri-members", lambda _: LISTING)
    assert live.members() == {"v2member"}, "manifest.json is not a member and the header is not a row"


def test_identified(tmp_path: Path):
    """The dataset id comes from the metadata that is pushed with the package, not from a constant."""
    (tmp_path / "dataset-metadata.json").write_text(json.dumps({"id": "dimiturnt/kneemri-members"}), encoding="utf-8")
    assert LiveDataset.identified(tmp_path) == "dimiturnt/kneemri-members"


def test_entrypoint():
    """The CLI is addressed by a path or by name, never by something that cannot be executed."""
    assert LiveDataset.entrypoint().endswith(("kaggle", "kaggle.exe"))
