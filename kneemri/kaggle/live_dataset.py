"""What a Kaggle dataset actually holds, asked of Kaggle rather than of the directory it was built in."""

import json
import sys
from collections.abc import Callable
from pathlib import Path

from kneemri.spawn import Spawn

DATASET_METADATA = "dataset-metadata.json"
LIVE_TIMEOUT_SECONDS = 120
LIVE_PAGE_SIZE = 1000
NEXT_PAGE_MARKER = "Next Page Token"


class TruncatedListingError(RuntimeError):
    """The live listing did not fit in one page, so a missing name proves nothing.

    Raised rather than returned because the failure is indistinguishable from a real absence at the
    call site, and the whole value of this class is that a missing name means something.
    """


class LiveDataset:
    """What the members dataset holds ON KAGGLE, which is the only place a kernel can read it from.

    WHY THIS EXISTS, and it cost a kernel run on 2026-08-19. `MountCheck` reads the LOCAL package, so
    it answers "do the kernels name members this machine has packaged" — a question about a directory
    under the processed root. The kernel then ran on Kaggle and died with `WeightsError: ... holds
    [nine members] and was asked for ['band_dense_s2026']`, because the package had been rebuilt
    locally and never pushed. The check was green for the whole of that failure, and it was green
    CORRECTLY: nothing in it had ever addressed the live dataset.

    A stale local package is the harmless direction — a member absent locally cannot be named by a
    kernel that passed. The dangerous direction is the one that happened, where the local package leads
    the live one, and only Kaggle can answer it.

    The listing is INJECTED rather than fetched in `__init__` so the class is testable without a
    network: `hygiene.small` refuses a unit test that touches anything it did not create, and a check
    whose only test needs credentials is a check nobody runs under the gates.
    """

    def __init__(self, identifier: str, listing: Callable[[str], str] | None = None) -> None:
        self.identifier = identifier
        self.listing = listing or self._cli

    @staticmethod
    def entrypoint() -> str:
        """The kaggle CLI, BESIDE THE RUNNING INTERPRETER before anything on PATH.

        `kaggle` is declared in the `submit` extra, so it installs into `.venv/Scripts` and a bare
        `kaggle` resolves only when that directory happens to be on PATH — which it is not under
        `uv run` on Windows, where the first attempt died with `WinError 2` for a binary sitting next
        to the interpreter that raised it.
        """
        beside = Path(sys.executable).parent / "kaggle.exe"
        return str(beside) if beside.exists() else "kaggle"

    subcommand: tuple[str, ...] = ("datasets", "files")
    """The CLI verbs that list this KIND of Kaggle thing — see `LiveModel` for the other one."""

    def _cli(self, identifier: str) -> str:
        """The CLI's own CSV listing, which is the read path that works without a bearer token."""
        # fixed argv, a dataset id from our own metadata, no shell
        done = Spawn().ran(
            [
                self.entrypoint(),
                *self.subcommand,
                identifier,
                "--csv",
                "--page-size",
                str(LIVE_PAGE_SIZE),
            ],
            capture_output=True,
            text=True,
            timeout=LIVE_TIMEOUT_SECONDS,
            check=True,
        )
        return done.stdout

    def members(self) -> set[str]:
        """Every member identifier the live dataset holds, from the `.pt` filenames it lists.

        REFUSES A PARTIAL ANSWER, which is the failure this method shipped with. The CLI pages at 20
        files by default; the members package held 38. So on 2026-08-25 the live check compared kernel
        tuples against page one and refused nearly every kernel in the tree, `allfit` among them, whose
        members had produced a COMPLETE 0.937 board entry that same day.

        Both directions of that are wrong and the quiet one is worse: a valid kernel refused costs a
        submission window to diagnose, but a listing that happens to page in the members a bad kernel
        names passes it silently. A gate trusted in both directions is worse than no gate.
        """
        rows = self.listing(self.identifier).splitlines()
        if any(row.startswith(NEXT_PAGE_MARKER) for row in rows):
            raise TruncatedListingError(
                f"{self.identifier} listed more than {LIVE_PAGE_SIZE} files, so a name missing from "
                f"this page is not a name the dataset lacks - raise LIVE_PAGE_SIZE or page the listing"
            )
        names = (row.split(",", 1)[0].strip() for row in rows[1:] if row.strip())
        return {name.removesuffix(".pt") for name in names if name.endswith(".pt")}

    def paths(self) -> set[str]:
        """Every file path the live dataset lists, as the kernel will see them under its mount root.

        `members` asks which `.pt` files are there and throws the directories away, which is the right
        question for the members package and blind to the one that cost `farseat_v1`. An ENCODER dataset
        is addressed by a PATH: `Encoder.locates` matches its `leaf` as a path SEGMENT, so
        `dimiturnt/dinov3-hf/vits16` needs a `vits16/` prefix to exist inside the dataset and not merely
        the dataset to exist. On 2026-08-24 `dimiturnt/vjepa2-hf` was pushed with `--dir-mode zip`, which
        unpacks a subdirectory into the ROOT, so the declared `vjepa2-vitl` segment was never published.
        Every check stayed green — the kernel declared the dataset, the dataset existed, the member was
        packaged — and the kernel then dropped the member and scored as a smaller ensemble.

        Truncation is refused for the same reason `members` refuses it, and it matters more here: a
        listing that pages out the one prefix being asked about answers NO to a question whose true
        answer is yes, and the caller cannot tell that from a real absence.
        """
        rows = self.listing(self.identifier).splitlines()
        if any(row.startswith(NEXT_PAGE_MARKER) for row in rows):
            raise TruncatedListingError(
                f"{self.identifier} listed more than {LIVE_PAGE_SIZE} files, so a prefix missing from "
                f"this page is not a prefix the dataset lacks - raise LIVE_PAGE_SIZE or page the listing"
            )
        return {row.split(",", 1)[0].strip() for row in rows[1:] if row.strip()}

    def holds(self, subpath: str) -> bool:
        """Whether the live dataset publishes anything under a subpath — the mount's own leaf question.

        An empty subpath is the flat layout (`dimiturnt/rad-dino`, `dimiturnt/vjepa2-hf`), where the
        dataset name IS the leaf and any file at all answers it.
        """
        listed = self.paths()
        if not subpath:
            return bool(listed)
        prefix = subpath.strip("/") + "/"
        return any(path.replace("\\", "/").startswith(prefix) for path in listed)

    @staticmethod
    def identified(root: Path) -> str:
        """The dataset id a package ships under, read from the metadata that is pushed with it."""
        return str(json.loads((root / DATASET_METADATA).read_text(encoding="utf-8"))["id"])


class LiveModel(LiveDataset):
    """The same question asked of a Kaggle MODEL VERSION, which answers on a different endpoint.

    `metaresearch/dinov2/PyTorch/small/1` is not a dataset, and the datasets endpoint does not say so
    politely: `kaggle datasets files metaresearch/dinov2` returns `403 Forbidden`, which the CLI raises
    as a non-zero exit and the caller sees as a traceback out of a gate that was asking about something
    else. That is how the encoder-mount check crashed on its first live run over the whole package,
    twenty minutes into a submission window, on the three mounts every champion entry has always used.

    A model version publishes a FLAT list of files (`config.json`, `pytorch_model.bin`), so the whole
    slug is the identifier and there is no subpath inside it to ask about — the mount IS the leaf. The
    inherited `holds("")` is therefore the right question, and it is a real one: a model version that
    was renamed or withdrawn lists nothing.
    """

    subcommand: tuple[str, ...] = ("models", "instances", "versions", "files")
