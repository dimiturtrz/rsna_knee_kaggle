"""The competition forum read through injected listers — every path but the round trip itself."""

from collections.abc import Sequence
from dataclasses import dataclass

from kneemri.kaggle.discussion_read import Discussion, DiscussionTopic


@dataclass(frozen=True)
class _Topic:
    """What the Kaggle client hands back: attributes, no schema, and fields that go missing."""

    id: int
    title: str
    votes: int | None = 0
    comment_count: int | None = 0
    last_comment_post_date: str = ""


@dataclass(frozen=True)
class _Post:
    raw_content: str | None = None
    content: str = ""


PAGES: dict[int, Sequence[object]] = {
    1: [
        _Topic(1, "OOF vs LB", votes=9, comment_count=3, last_comment_post_date="2026-08-25 11:00:00"),
        _Topic(2, "single model 0.942", votes=12, last_comment_post_date="2026-08-20 09:00:00"),
    ],
    2: [_Topic(3, "site split?", votes=12, comment_count=1, last_comment_post_date="2026-08-24 08:00:00")],
}
"""Stamps SPACE-separated, which is the shape the API really returns. They were written here with an ISO
`T` instead, and that one character was the whole reason `since` shipped broken: the fixture agreed with
the filter's assumption rather than with the server's output, so both were wrong together and green."""


def _lister(page: int) -> Sequence[object]:
    return PAGES.get(page, [])


def test_plain():
    """A forum post is an HTML fragment, and every consumer downstream wants the sentence, not the markup."""
    assert DiscussionTopic.plain("<p>OOF&nbsp;is <b>0.92</b></p>\n<p>LB   0.938</p>") == "OOF is 0.92 LB 0.938"
    assert DiscussionTopic.plain("") == ""
    assert DiscussionTopic.plain("<img src='https://storage.googleapis.com/x'/>") == "", "no sentence in it"


def test_seen():
    """One client object as a row, with the null votes their API really returns treated as zero."""
    row = DiscussionTopic.seen(_Topic(735304, "Single model <b>0.942</b>", votes=None, comment_count=None))

    assert (row.identifier, row.title) == (735304, "Single model 0.942"), "the title is flattened too"
    assert (row.votes, row.replies) == (0, 0), "None is not a crash and not a missing row"

    counted = DiscussionTopic.seen(_Topic(1, "x", comment_count=4))

    assert counted.replies == 4, "comment_count, NOT num_replies — the wrong name reads back as a clean 0"


def test_taken():
    """The sweep stops on the first empty page — the endpoint gives no total, so emptiness is the only end."""
    discussion = Discussion.taken(lister=_lister, pages=5)

    assert [topic.identifier for topic in discussion.topics] == [1, 2, 3]
    assert Discussion.taken(lister=_lister, pages=1).topics == discussion.topics[:2], "the ceiling is honoured"
    assert Discussion.taken(lister=lambda _: []).topics == (), "no topics is an empty read, not an exception"


def test_ranked():
    """The field's own ordering of what mattered, ties broken so two runs never disagree."""
    ordered = Discussion.taken(lister=_lister).ranked()

    assert [topic.identifier for topic in ordered] == [3, 2, 1], "12 votes over 9, then replies break the tie"


def test_matching():
    """A title filter, because sixty threads is more than a reader spends attention on at once."""
    found = Discussion.taken(lister=_lister).matching("MODEL")

    assert [topic.identifier for topic in found] == [2], "case is the writer's choice, not the search's"
    assert Discussion.taken(lister=_lister).matching("efficiency") == []


def test_stamped():
    """Both sides of the comparison in one shape, because the two sides are written by different hands."""
    assert Discussion.stamped("2026-08-25T00:00:00") == "2026-08-25 00:00:00", "the ISO T a caller types"
    assert Discussion.stamped("2026-08-25 11:00:00") == "2026-08-25 11:00:00", "the space the API writes"
    assert Discussion.stamped("2026-08-25T11:00:00+00:00") == "2026-08-25 11:00:00", "a zone sorts above no zone"
    assert Discussion.stamped("") == ""


def test_since():
    """The delta a second reading wants, ordered by when it moved rather than by how it was voted."""
    moved = Discussion.taken(lister=_lister).since("2026-08-23T00:00:00")

    assert [topic.identifier for topic in moved] == [1, 3], "the 08-20 thread is settled, whatever its votes"
    assert Discussion.taken(lister=_lister).since("2026-08-26T00:00:00") == [], "nothing new is an empty list"
    assert Discussion.taken(lister=_lister).since("2026-08-25T00:00:00") == moved[:1], (
        "an ISO T against the API's space is the case that shipped broken: ' ' sorts BELOW 'T', so every "
        "thread posted in that same day was dropped and the filter reported an all-clear"
    )


def test_posts():
    """One thread's messages, and the content fallback for the ones that carry no raw_content."""
    read = [_Post(raw_content="<p>my OOF   0.92</p>"), _Post(content="<i>LB</i>")]
    posts = Discussion.posts(735304, reader=lambda _: read)

    assert posts == ["my OOF 0.92", "LB"], "raw first, rendered second, both flattened"
