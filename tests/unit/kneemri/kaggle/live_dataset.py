"""What the live Kaggle dataset holds, parsed from the CLI listing without touching the network."""

import json
from pathlib import Path

import pytest

from kneemri.kaggle.live_dataset import LiveDataset, TruncatedListingError

LISTING = "name,size,creationDate\nv2member.pt,89,2026-08-19 05:30:55\nmanifest.json,12,2026-08-19 05:30:55\n"
PAGED = LISTING + "Next Page Token: eyJvZmZzZXQiOjIwfQ==\n"


def test_members():
    """The listing is CSV with a header, and only the `.pt` rows name members."""
    live = LiveDataset("dimiturnt/kneemri-members", lambda _: LISTING)
    assert live.members() == {"v2member"}, "manifest.json is not a member and the header is not a row"


def test_members_refuses_a_paged_listing():
    """A page-one answer is REFUSED, because a name missing from page one is not a name Kaggle lacks.

    This is the 2026-08-25 failure verbatim: the CLI pages at 20, the members package held 38, and the
    live check refused nearly every kernel in the tree including one whose members had scored 0.937.
    """
    live = LiveDataset("dimiturnt/kneemri-members", lambda _: PAGED)
    with pytest.raises(TruncatedListingError, match="not a name the dataset lacks"):
        live.members()


def test_identified(tmp_path: Path):
    """The dataset id comes from the metadata that is pushed with the package, not from a constant."""
    (tmp_path / "dataset-metadata.json").write_text(json.dumps({"id": "dimiturnt/kneemri-members"}), encoding="utf-8")
    assert LiveDataset.identified(tmp_path) == "dimiturnt/kneemri-members"


def test_entrypoint():
    """The CLI is addressed by a path or by name, never by something that cannot be executed."""
    assert LiveDataset.entrypoint().endswith(("kaggle", "kaggle.exe"))


NESTED = (
    "name,size,creationDate\n"
    "vits16/config.json,785,2026-08-23 12:42:39\n"
    "vits16/model.safetensors,89,2026-08-23 12:43:21\n"
)
FLAT = "name,size,creationDate\nconfig.json,785,2026-08-24 12:42:39\nmodel.safetensors,89,2026-08-24 12:43:21\n"


def test_paths():
    """Every listed path, directories kept — the question `members` throws away by design."""
    live = LiveDataset("dimiturnt/dinov3-hf", lambda _: NESTED)
    assert live.paths() == {"vits16/config.json", "vits16/model.safetensors"}


def test_paths_refuses_a_paged_listing():
    """A prefix missing from page one is not a prefix Kaggle lacks, so a partial listing is refused."""
    live = LiveDataset("dimiturnt/dinov3-hf", lambda _: PAGED)
    with pytest.raises(TruncatedListingError, match="not a prefix the dataset lacks"):
        live.paths()


def test_holds():
    """A nested mount is held only when the leaf is a real prefix, and a flat dataset lacks every leaf.

    The second assertion is the 2026-08-24 defect exactly: `dimiturnt/vjepa2-hf` was pushed with
    `--dir-mode zip`, which unpacks a subdirectory into the root, so the declared `vjepa2-vitl` segment
    never existed and every kernel naming that encoder silently dropped its member.
    """
    assert LiveDataset("dimiturnt/dinov3-hf", lambda _: NESTED).holds("vits16")
    assert not LiveDataset("dimiturnt/vjepa2-hf", lambda _: FLAT).holds("vjepa2-vitl")
    assert LiveDataset("dimiturnt/vjepa2-hf", lambda _: FLAT).holds(""), "a flat mount is its own leaf"
