"""The one door to Kaggle: the competition's slug and an authenticated client.

Imported LAZILY, inside `authenticated`: `kaggle` is the optional `submit` extra, and importing its
package already attempts a credential read. So every reader here keeps its offline path free of both,
and touches the client only on a live call.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaggle.api.kaggle_api_extended import KaggleApi

SLUG = "rsna-knee-abnormality-detection"
"""The competition as Kaggle's API names it — not `core.paths.COMPETITION`, the directory it unzips into."""


class Kaggle:
    """The client's one construction site, so signing in is written once."""

    @staticmethod
    def authenticated() -> "KaggleApi":
        """A client signed in from `~/.kaggle`. Constructing one spends no submission slot."""
        from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

        api = KaggleApi()
        api.authenticate()
        return api
