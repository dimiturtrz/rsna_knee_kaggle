"""Is the kit we are about to ship the code we are running? `python -m kneemri.kaggle.kit_check`.

THE FOURTH CHECK, and like the other three it exists for a failure that already happened — twice, and
the second time after the fix was already committed.

2026-08-13: a submission ran on a kit built 2026-08-12 09:46 UTC while HEAD was 54 commits ahead. The
mounted code therefore lacked `9f31b35` (13:11 UTC that day), which bounds the forward pass by IMAGES
rather than studies — so whole-stack members sent four times the activation into a 16 GB T4, no
submission was written, and the fallback's constant table scored 0.500. The members were never measured
and the slot was gone. The same stale kit also predated `Member.precision`, which serves a bf16 member
in the arithmetic it was fitted under.

Members are versioned, kernels are versioned, and the code between them was not. `build_kit` now stamps
the commit into `kit.json` and `kernel_run` logs it, so the drift is VISIBLE. This makes it BLOCKING:
what matters is not that the kit differs from HEAD, but that it differs in the files a kernel LOADS —
the entrypoint's import closure (`kernel_closure`), which is a fraction of the packages a kit ships.

`--mounted` ASKS THE ONLY QUESTION A SLOT CARES ABOUT, and `MountedKit` fetches the answer. The kernel's
log names the mounted commit, but only after the slot is spent — the same too-late that cost the first two.
"""

import json
import logging
import os
import shutil
from pathlib import Path

from core import cli
from core.denominator import Denominator
from kneemri.kaggle.kernel_closure import KernelClosure
from kneemri.kaggle.mounted_kit import DATASET, MountedKit
from kneemri.spawn import Spawn

logger = logging.getLogger(__name__)


class KitCheck:
    """The built kit's commit against the working tree's, in the files a kernel actually loads."""

    @staticmethod
    def unhooked() -> dict[str, str]:
        """This process's environment with git's hook variables removed.

        `GIT_DIR`, `GIT_INDEX_FILE` and `GIT_WORK_TREE` are how a hook is told which repository invoked
        it, and a git child inherits them: `git -C somewhere-else` then still answers about the hook's
        repo. Any git call meant to be about a NAMED directory needs an environment without them.

        STATIC because it is needed BEFORE a KitCheck can exist — `__init__` shells git itself, and
        `build_kit` stamps a kit without constructing one at all.
        """
        return {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}

    def __init__(self, inside: Path) -> None:
        """Anchored at the repository ROOT, whatever path inside it is handed over.

        The pathspec is what makes this a root and not a directory: `git -C <dir> log -- kneemri core`
        resolves those names against <dir>, so anchoring one level down asked about `kneemri/kneemri`
        and `kneemri/core`, matched nothing, and reported EVERY kit as current. The mounted check
        passed that way for 53 shipped-package commits, including the restructure that moved this
        module down a level and caused it. Asking git where the root is cannot drift when a file moves.
        """
        self.git = shutil.which("git") or "git"
        self.repo = inside
        self.repo = Path(self.asked("rev-parse", "--show-toplevel"))
        self.loaded = KernelClosure(self.repo).files()

    def asked(self, *command: str) -> str:
        """One read-only git question, answered from this repo — and from THIS repo under a git hook.

        `-C` IS NOT ENOUGH. Git exports `GIT_DIR`, `GIT_INDEX_FILE` and `GIT_WORK_TREE` into every hook
        it runs, and an inherited `GIT_DIR` OUTRANKS `-C`: the child then answers about whatever
        repository invoked the hook, no matter which directory it was pointed at. This check runs from
        pre-commit and pre-push, which is exactly where those variables are set.

        In this repo the two happen to be the same tree, so the wrong answer is the right one and
        nothing said otherwise — until the unit tests, which build their own repository in a temp
        directory, failed inside a pre-push run and passed everywhere else. A check that silently
        reports on a different repository than the one it was handed is the failure `--mounted` exists
        to prevent, so it is fixed here rather than in the tests that found it.
        """
        # resolved executable, fixed argv, no shell, our own repo
        return (
            Spawn()
            .ran(
                [self.git, "-C", str(self.repo), *command],
                capture_output=True,
                text=True,
                check=True,
                env=self.unhooked(),
            )
            .stdout.strip()
        )

    def stamped(self, kit: Path) -> str | None:
        """The commit a built kit records, or None when it predates the stamp.

        None rather than a guess: a kit whose version cannot be named is the state that spent the slot,
        so it is reported as unknown and refused rather than assumed current.
        """
        recorded = self.recorded(kit)
        return str(recorded["commit"]) if recorded else None

    def recorded(self, kit: Path) -> dict[str, object] | None:
        """The whole stamp — the commit AND whether the tree was dirty when it was built."""
        stamp = kit / "kneemri_src" / "kit.json"
        return dict(json.loads(stamp.read_text(encoding="utf-8"))) if stamp.exists() else None

    def drift(self, kit: Path) -> list[str]:
        """Commits touching the files a kernel loads that the built kit does not have, newest first."""
        return self.behind(self.recorded(kit))

    def behind(self, recorded: dict[str, object] | None) -> list[str]:
        """The same judgement for any stamp, wherever it was read from — a build directory or Kaggle.

        The pathspec is the ENTRYPOINT'S IMPORT CLOSURE, not the shipped packages: a kit ships every
        module under `kneemri` and `core`, but a kernel imports a fraction of them, and a commit to a
        measurement module cannot change what the kernel computes any more than a commit to a test can.
        """
        if recorded is None:
            return ["the kit carries no commit stamp — rebuild it before shipping"]
        behind = self.asked("log", "--oneline", f"{recorded['commit']}..HEAD", "--", *self.loaded).splitlines()
        # A DIRTY KIT NAMES A COMMIT IT IS NOT. On 2026-08-14 one was uploaded and run from a tree with
        # uncommitted changes, so its stamp pointed at code that was never what shipped — which is the
        # same unnameable-version state the stamp exists to end, wearing a commit hash.
        if recorded.get("dirty"):
            behind.append(f"built from a DIRTY tree at {recorded['commit']} — the stamp names code that did not ship")
        return behind


def main() -> None:
    """Refuse a kit that is behind the working tree in the packages a kernel runs."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--kit", type=Path, help="the built kit; defaults beside the other build artefacts")
    command.parser.add_argument(
        "--mounted", action="store_true", help="ask Kaggle what it will mount, not the local build"
    )
    command.parser.add_argument("--dataset", default=DATASET)
    arguments = command.parsed()

    repo = Path(__file__).resolve().parent
    check = KitCheck(repo)
    # The closure IS the pathspec, and an empty one is how this check passed 53 stale kits: git asked
    # about files that matched nothing named no commit, so every kit read as current. A kernel that
    # imports no module cannot exist, so zero here is the resolver failing, never a small closure.
    Denominator(len(check.loaded), "the kernel's import closure is empty — the resolver failed").refuse_empty()
    if arguments.mounted:
        where, recorded = arguments.dataset, MountedKit(arguments.dataset).recorded()
    else:
        kit = arguments.kit or command.rooted(arguments).processed() / "kit"
        where, recorded = str(kit), check.recorded(kit)
    behind = check.behind(recorded)
    for line in behind:
        logger.info("MISSING FROM THE KIT: %s", line)
    stamp = str(recorded["commit"]) if recorded else "UNSTAMPED"
    logger.info(
        "%s at %s, %d commit(s) behind in the %d files the kernel loads", where, stamp, len(behind), len(check.loaded)
    )
    if behind:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
