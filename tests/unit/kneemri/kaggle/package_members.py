"""The manifest rows, which are the contract between a checkpoint on disk and every kernel.

Nothing here needs a data root or a checkpoint file: a row is a pure function of a path and a payload,
and that is the point — while it lived inside `main` it could only be exercised by packaging 8 GB.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from kneemri.kaggle.package_members import FORMAT, MemberManifest, MemberRoster, PackagedMember, RosterShelf

CONFIG = {"crop_mm": 140.0, "whole_stack": True}
FINGERPRINT = [0.1, 0.2]


def _member(name: str, holdout: float = 0.84) -> PackagedMember:
    return PackagedMember(Path(name), {"config": CONFIG, "fingerprint": FINGERPRINT, "holdout": holdout})


def test_identifier():
    """The id comes off the filename by string surgery, and a member is addressed by it."""
    assert _member("member_fold3.pt").identifier == "fold3"
    assert _member("member_seed7_fold0.pt").identifier == "seed7_fold0"


def test_variant():
    """The encoder a member was fitted on, defaulting the same way the manifest side defaults it."""
    assert _member("member_fold0.pt").variant == "small", "a checkpoint predating the field reads as small"
    assert PackagedMember(Path("member_cnx.pt"), {"config": {"variant": "cnxbase"}}).variant == "cnxbase"


def test_shippable():
    """A packaged set on mounted encoders passes without a word."""
    assert MemberManifest([_member("member_fold0.pt")]).shippable() is None


def test_shippable_refuses_a_member_whose_encoder_no_dataset_mirrors():
    """The check that used to fire at kernel-metadata time, i.e. after the card was already spent.

    `cnxbase` has a trained checkpoint on disk and no Kaggle mount, so a roster line naming it used to
    package and push clean; the refusal arrived from `MemberPackage.mounts` much later, as a traceback.
    """
    member = PackagedMember(Path("member_cnx.pt"), {"config": {"variant": "cnxbase"}})
    with pytest.raises(ValueError, match="no Kaggle dataset mirrors"):
        MemberManifest([member]).shippable()


def test_document_refuses_an_unshippable_member_before_any_weights_are_copied():
    """Packaging asks the question, because packaging is where a member enters the shippable set."""
    member = PackagedMember(Path("member_cnx.pt"), {"config": {"variant": "cnxbase"}})
    with pytest.raises(ValueError, match="no Kaggle dataset mirrors"):
        MemberManifest([member]).document()


def test_entry():
    """The row, at the keys `Member` reads back — fold parsed out of the id."""
    entry = _member("member_fold3.pt").entry()

    assert entry["id"] == "fold3"
    assert entry["fold"] == 3
    assert entry["file"] == "member_fold3.pt"
    assert entry["config"] == CONFIG
    assert entry["holdout"] == pytest.approx(0.84)
    assert np.isnan(float(str(entry["annot"]))), "the annotation check is the competition's, not ours"


def test_a_member_whose_name_carries_no_fold_packages_as_fold_zero():
    """`member_res392.pt` and `member_seed7_fold0.pt` are both real names in the processed root."""
    assert _member("member_res392.pt").entry()["fold"] == 0
    assert _member("member_seed7_fold0.pt").entry()["fold"] == 0


def test_a_full_data_fit_packages_with_no_holdout_rather_than_a_stand_in():
    """A fit with nothing held back records NaN by construction, and that must survive the manifest.

    Rewriting it as a number here would undo `fitting.NO_HOLDOUT`: a final fit advertising a score it
    never earned outranks every real member (knee-2d7).
    """
    member = _member("member_final.pt", holdout=float("nan"))

    assert np.isnan(float(str(member.entry()["holdout"])))
    assert np.isnan(json.loads(MemberManifest([member]).document())["members"][0]["holdout"])


def test_document():
    """What a mounted package is read from, at the format version `MemberPackage` expects."""
    document = json.loads(MemberManifest([_member("member_fold0.pt")]).document())

    assert document["format"] == FORMAT
    assert [member["id"] for member in document["members"]] == ["fold0"]


def test_two_checkpoints_sharing_one_id_are_refused_rather_than_overwriting_each_other():
    """Several globs can match the same NAME under different directories — the copy keeps one file.

    The manifest would then list two members and address one set of weights twice, which loads and
    fingerprints cleanly and predicts from something one of those rows never saw.
    """
    with pytest.raises(ValueError, match="one id"):
        MemberManifest([_member(f"{parent}/member_fold0.pt") for parent in ("a", "b")]).document()


ROSTER = """
# what the live package holds
res392_ws_s2026.pt   # our best single member
sliceattn_s2026.pt

lora_r8_*.pt
"""


def _rostered(tmp_path: Path) -> MemberRoster:
    (tmp_path / "roster").write_text(ROSTER, encoding="utf-8")
    return MemberRoster.read(tmp_path / "roster")


def test_read(tmp_path: Path):
    """A roster is a file, so that `git log` can date the set of members a package holds."""
    assert _rostered(tmp_path).text == ROSTER


def test_patterns(tmp_path: Path):
    """Names only: the prose saying WHY a member is in the package is not a filename."""
    assert _rostered(tmp_path).patterns() == ["res392_ws_s2026.pt", "sliceattn_s2026.pt", "lora_r8_*.pt"]


def test_licence_refusals(tmp_path: Path):
    """A tagged member is refused a SEAT while still packaging — and the tag is invisible to `patterns`."""
    text = "clean_s2026.pt   # nothing external\ntainted_s2026.pt # 0.85  # licence: rijeka\n"
    (tmp_path / "roster").write_text(text, encoding="utf-8")
    roster = MemberRoster.read(tmp_path / "roster")
    assert roster.patterns() == ["clean_s2026.pt", "tainted_s2026.pt"]
    assert roster.licence_refusals({"clean_s2026"}) == []
    refused = roster.licence_refusals({"clean_s2026", "tainted_s2026"})
    assert len(refused) == 1
    assert refused[0].startswith("tainted_s2026 trains on the rijeka table")


def test_checkpoints(tmp_path: Path):
    """Resolved under the processed root, in roster order — a glob line may name more than one."""
    source = tmp_path / "processed"
    source.mkdir()
    for name in ("res392_ws_s2026.pt", "sliceattn_s2026.pt", "lora_r8_v3small_224_s2026.pt", "unrostered.pt"):
        (source / name).touch()
    assert [path.name for path in _rostered(tmp_path).checkpoints(source)] == [
        "res392_ws_s2026.pt",
        "sliceattn_s2026.pt",
        "lora_r8_v3small_224_s2026.pt",
    ]


def test_checkpoints_refuses_a_name_that_matches_nothing(tmp_path: Path):
    """A typo drops a member SILENTLY otherwise, and an ensemble one member short still scores."""
    source = tmp_path / "processed"
    source.mkdir()
    (source / "res392_ws_s2026.pt").touch()
    with pytest.raises(FileNotFoundError, match="sliceattn_s2026.pt"):
        _rostered(tmp_path).checkpoints(source)


SHELF = "# shelf: kneemri-members-2\n# out: members_2\n#\n# prose\noof_teacher_s2026.pt  # 0.8681\n"


def test_read_declares_both_halves_of_the_shelf():
    """The roster names its dataset id and its directory; nothing else may decide them."""
    shelf = RosterShelf.read(SHELF)
    assert (shelf.title, shelf.directory) == ("kneemri-members-2", "members_2")


def test_read_refuses_a_roster_that_does_not_declare_one():
    """A missing header is the incident: the destination would otherwise fall back to the live shelf."""
    with pytest.raises(ValueError, match="# shelf:"):
        RosterShelf.read("oof_teacher_s2026.pt\n")


def test_root():
    """Both shelves sit under the models root, so the directory is one segment of it."""
    assert RosterShelf.read(SHELF).root(Path("models")) == Path("models") / "members_2"


def test_agrees():
    """A flag pointing at the OTHER shelf is refused; one that agrees passes silently."""
    shelf = RosterShelf.read(SHELF)
    assert shelf.agrees(Path("models") / "members_2", "kneemri-members-2", Path("models")) is None
    assert shelf.agrees(None, None, Path("models")) is None
    with pytest.raises(ValueError, match="contradicts the roster"):
        shelf.agrees(Path("models") / "all_ours", None, Path("models"))
    with pytest.raises(ValueError, match="contradicts the roster"):
        shelf.agrees(None, "kneemri-members", Path("models"))


def test_shelf(tmp_path: Path):
    """A roster hands out its own shelf, so a caller cannot read one file and package to another."""
    path = tmp_path / "members2.roster"
    path.write_text(SHELF, encoding="utf-8")
    assert MemberRoster.read(path).shelf().title == "kneemri-members-2"
