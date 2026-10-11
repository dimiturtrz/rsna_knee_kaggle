"""A mounted foreign scoring script run as one seat: patched by anchor, executed, its table required.

Eight S6 kernels spell the same two steps inline. They run nartaa's mounted mirror and move its
`submission.csv` aside, then read heliosli's S6 source, rewrite two lines that must each appear exactly
once, write the copy to `/tmp`, run it under an environment carrying its deadline, and refuse the run if
its table is absent. Both are one seat with different settings, so they are one class here.

WHY THE TABLE IS THE SUCCESS MARKER, NOT THE EXIT CODE. The S6 outer cell exits non-zero when a fold
falls back, after it has already written a usable table — which is what the `if rc != 0:` patch makes it
tolerate. A seat that trusts the exit code either dies on a usable table or, with `check=False`, blends a
table some EARLIER step left at the same path. So the output is deleted before the run and required after
it: the exit code decides only whether a non-zero status is itself fatal.

WHY EXACTLY ONE COPY OF EACH ANCHOR. A foreign source is published, not pinned. A patch that lands on no
site leaves the run unpatched and silent; one that lands on two rewrites a line the anchor never meant.
Either way the run costs a slot to discover, so the count is asserted before anything executes.
"""

import logging
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SUBMISSION = "submission.csv"
"""What a Kaggle scoring script writes into its working directory unless told otherwise."""


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


@dataclass(frozen=True, kw_only=True)
class ScriptSeat:
    """One mounted script, the patches it needs, and where the table it leaves should end up."""

    script: Path
    anchors: tuple[Anchor, ...] = ()
    environment: Mapping[str, str] = field(default_factory=dict)
    table: str = SUBMISSION
    kept_as: str | None = None
    staging: Path = field(default_factory=lambda: Path(tempfile.gettempdir()))
    exit_tolerant: bool = False

    def staged(self) -> Path:
        """The file to execute: the mount itself when nothing is patched, else a compiled, patched copy.

        Unpatched runs from the mount so the script's own directory is still the first import path, as
        it is when the kernel ran it directly. A patched copy is `compile`d before it is written, so a
        patch that breaks the syntax fails here rather than as a child's exit status nobody reads.
        """
        if not self.anchors:
            return self.script
        text = self.script.read_text(encoding="utf-8")
        for anchor in self.anchors:
            text = anchor.applied(text, str(self.script))
        destination = self.staging / f"{self.script.stem}_patched{self.script.suffix}"
        compile(text, str(destination), "exec")
        destination.write_text(text, encoding="utf-8")
        logger.info("%s patched at %d anchors into %s", self.script.name, len(self.anchors), destination)
        return destination

    def run(self, working: Path) -> Path:
        """Execute the staged script in `working` and return the path its table was kept at.

        Its output is not captured: a long child whose log arrives only when it exits is a child nobody
        can watch, so its lines land in the kernel log beside ours.
        """
        produced = working / self.table
        produced.unlink(missing_ok=True)
        started = time.time()
        finished = subprocess.run(  # noqa: S603 - a mounted script, fixed argv, no shell
            [sys.executable, "-u", str(self.staged())],
            check=not self.exit_tolerant,
            cwd=working,
            env={**os.environ, **self.environment},
        )
        logger.info("%s exited %d after %.0fs", self.script.name, finished.returncode, time.time() - started)
        if not produced.is_file():
            message = f"{self.script.name} left no {self.table} (exit {finished.returncode}), so this seat is empty"
            raise RuntimeError(message)
        return produced.replace(working / (self.kept_as or self.table))
