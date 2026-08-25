"""What are the other competitors saying? `python -m kneemri.kaggle.discussion_read --live`.

THE DISCUSSION IS THE ONLY EVIDENCE SOURCE HERE THAT IS ABOUT RIVALS RATHER THAN ABOUT US. `board_read`
reads what the leaders SCORE; this reads what they say they did to score it. On 2026-08-25 topic 735304
held five self-reported single-model scores between 0.915 and 0.942 — against our best single member at
0.914 and our best SIX-member ensemble at 0.938. One sentence in a forum thread localised the board gap
in member quality, which no number we own could have done, because every number we own is measured inside
our own pool.

IT COSTS NO SUBMISSION SLOT, and it was believed unreachable for most of a session. The web route is a
dead end — `kaggle.com/competitions/<slug>/discussion` serves a 5KB JavaScript shell with no topic data in
it, so WebFetch and curl both come back with a page title and nothing else. The API has carried
`competition_list_topics` and `competition_list_topic_messages` the whole time. That is the cost of
probing a tool by its website rather than by its client.

Messages arrive as HTML fragments and are flattened here, because the point is to read them, and because
a `<p>` in a quoted claim is noise in every consumer this feeds.
"""

import html
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from core import cli

logger = logging.getLogger(__name__)

COMPETITION = "rsna-knee-abnormality-detection"
"""The slug. A WRONG one answers 403 rather than 404, which reads as a permissions problem."""

PAGES = 3
"""How many pages of topics to sweep. The endpoint pages at twenty and gives no total, so the only honest
stopping rule is an empty page — this is the ceiling on how long we look for one."""

TAG = re.compile(r"<[^>]+>")
WHITESPACE = re.compile(r"\s+")

TOP = 20
"""How many topics the summary prints. The rest are still read and still counted."""


@dataclass(frozen=True)
class DiscussionTopic:
    """One thread: what it is called, how the field voted on it, and how much argument it drew."""

    identifier: int
    title: str
    votes: int
    replies: int
    active: str

    @staticmethod
    def plain(fragment: str) -> str:
        """An HTML message as one line of readable text, entities resolved and runs of space collapsed.

        Tags are dropped rather than rendered. An `<img>` in a forum post is a screenshot of somebody's
        OOF-versus-LB plot, and its URL is 200 characters of Google Storage token that says nothing a
        reader can act on — the sentence around it is the evidence.
        """
        return WHITESPACE.sub(" ", html.unescape(TAG.sub(" ", fragment))).strip()

    @classmethod
    def seen(cls, topic: object) -> "DiscussionTopic":
        """One API topic object as a row, every field defaulted — the client's shape is not ours to fix.

        The reply count is `comment_count`. It is NOT `num_replies`, which was the first guess here and
        which reads back as a clean 0 on every row rather than as an error — sixty threads all reporting
        no discussion is the shape a wrong attribute name takes when `getattr` carries a default.
        """
        return cls(
            identifier=int(getattr(topic, "id", 0)),
            title=cls.plain(str(getattr(topic, "title", ""))),
            votes=int(getattr(topic, "votes", 0) or 0),
            replies=int(getattr(topic, "comment_count", 0) or 0),
            active=str(getattr(topic, "last_comment_post_date", "") or ""),
        )


@dataclass(frozen=True)
class Discussion:
    """Every thread on the competition's forum, ranked the way a reader should spend attention on it."""

    topics: tuple[DiscussionTopic, ...]

    @classmethod
    def taken(cls, lister: Callable[[int], Sequence[object]] | None = None, pages: int = PAGES) -> "Discussion":
        """Sweep the topic pages until one comes back empty, or until `pages` have been read.

        Injected for the reason `board_read.LeaderboardRead` gives: a path reachable only through real
        credentials is a path with no test.
        """
        read = lister if lister is not None else cls._kaggle_topics
        seen: list[DiscussionTopic] = []
        for page in range(1, pages + 1):
            listed = list(read(page))
            if not listed:
                break
            seen += [DiscussionTopic.seen(topic) for topic in listed]
        return cls(tuple(seen))

    def ranked(self) -> list[DiscussionTopic]:
        """The threads, most-voted first — the field's own ordering of what was worth writing about."""
        return sorted(self.topics, key=lambda topic: (-topic.votes, -topic.replies, topic.title))

    def matching(self, term: str) -> list[DiscussionTopic]:
        """The threads whose title holds `term`, case-insensitively. Ranked, so the answer is readable."""
        return [topic for topic in self.ranked() if term.lower() in topic.title.lower()]

    def since(self, stamp: str) -> list[DiscussionTopic]:
        """The threads that took a post after `stamp` — the delta, which is all a second reading wants.

        Compared as text, which is what an ISO stamp is FOR — but ONLY after both sides are put in the
        same shape. The API writes `2026-08-25 11:00:00` with a space and a caller types the ISO `T`,
        and a space sorts BELOW `T`, so `--since 2026-08-25T00:00:00` silently excluded every thread
        that had been posted in that day and reported no movement. A filter whose failure mode is an
        empty result is a filter that reports an all-clear when it breaks.

        Sorted most-recent first rather than by votes: a thread argued in today outranks a settled
        announcement with more of them.
        """
        wanted = self.stamped(stamp)
        moved = [topic for topic in self.topics if self.stamped(topic.active) > wanted]
        return sorted(moved, key=lambda topic: topic.active, reverse=True)

    @staticmethod
    def stamped(value: str) -> str:
        """One stamp in the one shape both sides are compared in: space-separated, no timezone suffix."""
        return value.replace("T", " ").split("+")[0].strip()

    @staticmethod
    def _kaggle_topics(page: int) -> Sequence[object]:
        """The real round trip, kept in one place so every other path is credential-free."""
        from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

        api = KaggleApi()
        api.authenticate()
        return Discussion._items(api.competition_list_topics(COMPETITION, page=page), "topics")

    @staticmethod
    def _items(listed: object, field: str) -> list[object]:
        """The payload's rows, whether the client wraps them in a response object or hands them back bare.

        A shape it cannot iterate reads as an empty page rather than a crash, and an empty page ends the
        sweep — so an API change costs a short read, never a wrong one.
        """
        found = getattr(listed, field, listed)
        return list(found) if isinstance(found, Sequence) else []

    @staticmethod
    def posts(identifier: int, reader: Callable[[int], Sequence[object]] | None = None) -> list[str]:
        """One thread's messages, flattened to text. Injected the same way, for the same reason."""
        read = reader if reader is not None else Discussion._kaggle_posts
        return [
            DiscussionTopic.plain(str(getattr(post, "raw_content", None) or getattr(post, "content", "")))
            for post in read(identifier)
        ]

    @staticmethod
    def _kaggle_posts(identifier: int) -> Sequence[object]:
        """The real round trip for one thread's messages."""
        from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

        api = KaggleApi()
        api.authenticate()
        listed = api.competition_list_topic_messages(COMPETITION, topic_id=identifier)
        return Discussion._items(listed, "messages")


def main() -> None:
    """Read the forum: `--live` for the ranked topic list, `--topic ID` for one thread's posts."""
    command = cli.Command(Discussion.__doc__, logger)
    command.parser.add_argument("--live", action="store_true", help="ask Kaggle (credentials, NO slot)")
    command.parser.add_argument("--topic", type=int, default=None, help="dump one thread's messages")
    command.parser.add_argument("--find", type=str, default=None, help="only titles holding this term")
    command.parser.add_argument("--since", type=str, default=None, help="only threads posted in after this stamp")
    arguments = command.parsed()

    if arguments.topic is not None:
        for post in Discussion.posts(arguments.topic):
            logger.info("-- %s", post)
        return
    if not arguments.live:
        raise SystemExit("nothing to read without --live or --topic; the forum has no local record")

    discussion = Discussion.taken()
    if arguments.since:
        found = discussion.since(arguments.since)
    elif arguments.find:
        found = discussion.matching(arguments.find)
    else:
        found = discussion.ranked()
    logger.info("%d topics read, %d shown", len(discussion.topics), min(len(found), TOP))
    for topic in found[:TOP]:
        logger.info(
            "%7d %4dv %3dr %s  %s", topic.identifier, topic.votes, topic.replies, topic.active[:16], topic.title
        )


if __name__ == "__main__":
    main()
