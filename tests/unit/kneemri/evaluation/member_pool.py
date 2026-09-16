"""The pool a pooled reading starts from: what the declines file says, and what it takes out."""

from pathlib import Path

from kneemri.evaluation.member_pool import MemberDeclines, MemberPool

VOIDED = "merged_blend_cf_n12b0.1-0.9"
HONEST = "labelwise_dense_s2026"
DECLINED = "smoke_run"
STATEMENT = (
    f"{VOIDED}.pt  # LEDGER75 0.9425 - void, in-sample on 703 of 909.\n{DECLINED}.pt  # a smoke run, sixty studies\n"
)


def _records(directory: Path, *names: str) -> Path:
    for name in names:
        (directory / f"{name}.holdout.npz").write_text("predictions", encoding="utf-8")
    return directory


def _declines(directory: Path, text: str = STATEMENT) -> Path:
    written = directory / "members.declines"
    written.write_text(text, encoding="utf-8")
    return written


def test_read(tmp_path: Path):
    """A declines file that does not exist yet reads as no declines, not as a crash."""
    written = tmp_path / "members.declines"
    written.write_text("a.pt # smoke\n", encoding="utf-8")

    assert MemberDeclines.read(written).reasons() == {"a": "smoke"}, "the file on disk"
    assert MemberDeclines.read(tmp_path / "absent").reasons() == {}, "a repo before anything was declined"


def test_reasons():
    """The declines file keeps the prose, which is the whole point of it existing."""
    declines = MemberDeclines("a.pt  # a smoke run\nb.pt\n\n# a whole-line comment\n")

    assert declines.reasons() == {"a": "a smoke run", "b": ""}, "a whole-line comment names no member"


def test_reasons_folds_a_wrapped_argument_into_the_entry_above_it():
    """A decline is an argument, and an argument wraps. Each wrapped line is prose, never a member."""
    declines = MemberDeclines("a.pt  # declined because the delta is under the floor\nand the sign is wrong.\nb.pt\n")

    assert declines.reasons() == {
        "a": "declined because the delta is under the floor and the sign is wrong.",
        "b": "",
    }, "the continuation joins its entry, and the bare name after it is still a finding"


def test_void():
    """A void decline says the NUMBER is not a reading; every other decline is about shipping only.

    Both kinds are in this file and only one of them may leave a pool — a filter that dropped every
    declined member would take 277 of 438 real readings with it.
    """
    declines = MemberDeclines(STATEMENT)

    assert declines.void() == {VOIDED}, "the smoke run's recorded number is still quotable"


def test_at(tmp_path: Path):
    """A pool is a directory plus the statement about it, and it reads that statement once."""
    _records(tmp_path, HONEST)
    pool = MemberPool.at(tmp_path, _declines(tmp_path))

    assert pool.directory == tmp_path
    assert pool.declines.void() == {VOIDED}, "the pool carries the statement, not just the path"


def test_names(tmp_path: Path):
    """The default pool drops the void records; an explicit ask is never second-guessed.

    The void member here carries a DOT in its name, because that is the case that defeated the filter:
    deriving a name at the first dot truncated it, matched no decline, and walked it back into the pool.
    """
    _records(tmp_path, HONEST, VOIDED, DECLINED)
    pool = MemberPool.at(tmp_path, _declines(tmp_path))

    assert pool.names() == sorted([HONEST, DECLINED]), "the void reading leaves, the shipping decline stays"
    assert pool.names([VOIDED]) == [VOIDED], "--only <member> prices exactly the member it names"
    assert MemberPool.at(tmp_path, tmp_path / "absent").names() == sorted([VOIDED, HONEST, DECLINED]), (
        "with nothing declared void, every record on disk is in the pool"
    )
