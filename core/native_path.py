"""A data root written for one OS, read on another.

WHY THIS IS A MODULE AND NOT A HELPER. A Windows drive path used on POSIX is not an error to pathlib —
it is a RELATIVE path, so `Path("D:/data") / "raw"` silently creates a repo-local `D:/` tree instead of
failing. Six empty directories were found at this repo's root on 2026-08-22, named after CLI values and
arm names, from exactly this class of mistake; cardiac-seg, which shares the data root, records the same
failure leaking dataset metadata into git. So the rule is translate-when-possible, raise-when-not, and
never silently relative.
"""

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

DRIVE_ANCHOR = 2
"""Length of a Windows drive anchor as pathlib reports it: `D:` is a letter and a colon."""

MOUNT_PARTS = 3
"""Parts a WSL mount path has before its content: the root, `mnt`, and the drive letter."""


@dataclass(frozen=True)
class NativePath:
    """Cross-platform data-root resolution of ONE configured path string, with no filesystem access.

    OS detection is injected (`os_name` / `wsl`) rather than read, so the whole matrix — a Windows path
    on Windows, on WSL and on bare Linux, and a mount path on each — is unit-testable from any host.
    `PureWindowsPath` / `PurePosixPath` do the parsing; there is no slash-regex here.

    The mapping is the DEFAULT WSL one (`D:/x` <-> `/mnt/d/x`). A machine with a custom `automount.root`
    needs its own path in the environment override, which is the escape hatch the raising branch names.
    """

    raw: str
    """The path exactly as it was configured, before any translation."""

    @staticmethod
    def detect_wsl() -> bool:
        """True iff running under WSL — a POSIX kernel exposing the Windows drives under `/mnt`."""
        if os.name == "nt":
            return False
        if os.environ.get("WSL_DISTRO_NAME"):
            return True
        try:
            return "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower()
        except OSError:
            return False

    def win_drive(self) -> str | None:
        """The lowercase drive letter if this is a Windows drive path, else None.

        `PureWindowsPath` is used rather than a string test so the drive is still detected when this
        runs on POSIX, which is the case the whole module exists for.
        """
        drive = PureWindowsPath(self.raw).drive
        return drive[0].lower() if len(drive) == DRIVE_ANCHOR and drive[1] == ":" else None

    def mount_drive(self) -> str | None:
        """The lowercase drive letter if this is a WSL mount path (`/mnt/d/x` -> `d`), else None."""
        parts = PurePosixPath(self.raw).parts
        if len(parts) >= MOUNT_PARTS and parts[1] == "mnt" and len(parts[2]) == 1 and parts[2].isalpha():
            return parts[2].lower()
        return None

    def to_mount(self) -> str:
        """A Windows drive path as its WSL mount (`D:/data/x` -> `/mnt/d/data/x`)."""
        drive = self.win_drive()
        if drive is None:
            message = f"to_mount expects a Windows drive path, got {self.raw!r}"
            raise ValueError(message)
        return str(PurePosixPath("/mnt", drive, *PureWindowsPath(self.raw).parts[1:]))

    def to_windows(self) -> str:
        """A WSL mount path as its Windows drive path (`/mnt/d/data/x` -> `D:/data/x`)."""
        drive = self.mount_drive()
        if drive is None:
            message = f"to_windows expects a WSL mount path, got {self.raw!r}"
            raise ValueError(message)
        parts = PurePosixPath(self.raw).parts
        return PureWindowsPath(f"{drive.upper()}:/", *parts[MOUNT_PARTS:]).as_posix()

    def resolve(self, escape: str, *, os_name: str | None = None, wsl: bool | None = None) -> str:
        """The configured root as a path that means something on THIS machine.

        Same family (a Windows path on Windows, a POSIX path on POSIX) passes through untouched. A
        Windows path on WSL becomes its mount and a mount path on Windows becomes its drive path. A
        Windows path on non-WSL POSIX cannot be translated and RAISES — that is the branch this module
        exists for, because the alternative pathlib offers is a silent relative directory.

        `escape` is the environment variable a caller can set instead, named in the refusal rather than
        hardcoded: this module knows how to translate a path and not which config it came from.
        """
        os_name = os.name if os_name is None else os_name
        wsl = NativePath.detect_wsl() if wsl is None else wsl
        drive = self.win_drive()
        if os_name == "nt":
            return self.to_windows() if drive is None and self.mount_drive() else self.raw
        if drive is None:
            return self.raw
        if wsl:
            return self.to_mount()
        message = (
            f"data root {self.raw!r} is a Windows drive path and this is non-WSL POSIX, so it cannot be "
            f"translated to a real path here. Set {escape} to a path on this machine. Refusing to "
            f"treat it as relative, which would silently create a repo-local {drive.upper()}:/ directory."
        )
        raise RuntimeError(message)
