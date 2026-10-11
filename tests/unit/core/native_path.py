import pytest

from core.native_path import NativePath

WINDOWS = "nt"
POSIX = "posix"
ESCAPE = "KNEEMRI_DATA"


def test_detect_wsl(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setattr("os.name", POSIX)
    assert NativePath.detect_wsl()


def test_detect_wsl_is_false_on_windows(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("os.name", WINDOWS)
    assert not NativePath.detect_wsl()


def test_win_drive():
    assert NativePath("D:/data/volumetric").win_drive() == "d"


def test_win_drive_is_none_for_a_posix_path():
    assert NativePath("/mnt/d/data").win_drive() is None


def test_mount_drive():
    assert NativePath("/mnt/d/data/volumetric").mount_drive() == "d"


def test_mount_drive_is_none_for_a_plain_posix_path():
    assert NativePath("/home/user/data").mount_drive() is None


def test_to_mount():
    assert NativePath("D:/data/volumetric").to_mount() == "/mnt/d/data/volumetric"


def test_to_mount_refuses_a_path_that_is_not_a_drive_path():
    with pytest.raises(ValueError, match="expects a Windows drive path"):
        NativePath("/home/user/data").to_mount()


def test_to_windows():
    assert NativePath("/mnt/d/data/volumetric").to_windows() == "D:/data/volumetric"


def test_to_windows_refuses_a_path_that_is_not_a_mount():
    with pytest.raises(ValueError, match="expects a WSL mount path"):
        NativePath("D:/data").to_windows()


def test_resolve():
    """The common case: the machine that wrote paths.yaml is the machine reading it."""
    assert NativePath("D:/data/volumetric").resolve(ESCAPE, os_name=WINDOWS, wsl=False) == "D:/data/volumetric"


def test_resolve_leaves_a_windows_path_alone_on_windows():
    assert NativePath("D:/data").resolve(ESCAPE, os_name=WINDOWS, wsl=False) == "D:/data"


def test_resolve_leaves_a_posix_path_alone_on_posix():
    assert NativePath("/home/user/data").resolve(ESCAPE, os_name=POSIX, wsl=False) == "/home/user/data"


def test_resolve_translates_a_windows_path_on_wsl():
    assert NativePath("D:/data/volumetric").resolve(ESCAPE, os_name=POSIX, wsl=True) == "/mnt/d/data/volumetric"


def test_resolve_translates_a_mount_path_on_windows():
    assert NativePath("/mnt/d/data/volumetric").resolve(ESCAPE, os_name=WINDOWS, wsl=False) == "D:/data/volumetric"


def test_resolve_raises_on_a_windows_path_under_non_wsl_posix():
    """The branch the module exists for: pathlib would make this a RELATIVE directory instead."""
    with pytest.raises(RuntimeError, match="Refusing to treat it as relative"):
        NativePath("D:/data").resolve(ESCAPE, os_name=POSIX, wsl=False)


def test_resolve_names_the_escape_hatch_it_was_given():
    """The module knows how to translate a path, not which config it came from — the caller names it."""
    with pytest.raises(RuntimeError, match="Set OTHER_VAR to a path"):
        NativePath("D:/data").resolve("OTHER_VAR", os_name=POSIX, wsl=False)
