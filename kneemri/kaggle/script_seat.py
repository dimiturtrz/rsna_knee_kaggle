"""A text patch to a foreign scoring source, refused unless its anchor appears exactly as often as declared.

WHY EXACTLY ONE COPY OF EACH ANCHOR. A foreign source is published, not pinned. A patch that lands on no
site leaves the run unpatched and silent; one that lands on two rewrites a line the anchor never meant.
Either way the run costs a slot to discover, so the count is asserted before anything executes.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Anchor:
    """One text patch to a foreign source: `spelling` becomes `replacement`, found exactly `sites` times."""

    spelling: str
    replacement: str
    sites: int = 1

    def applied(self, text: str, origin: str) -> str:
        """`text` with every `spelling` replaced, refused unless it appears exactly `sites` times in it."""
        found = text.count(self.spelling)
        if found != self.sites:
            message = f"{origin} spells {self.spelling!r} {found} times, not {self.sites}"
            raise ValueError(message)
        return text.replace(self.spelling, self.replacement)
