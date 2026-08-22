"""Which files a kernel loads, over a repository laid out on the spot."""

from pathlib import Path

from kneemri.kaggle.kernel_closure import KernelClosure

ENTRYPOINT = "kneemri/kaggle/kernel_run.py"


def _repo(root: Path) -> KernelClosure:
    """A miniature of the real layout: an entrypoint, what it reaches, and what nothing reaches."""
    written = {
        ENTRYPOINT: "import torch\nfrom core.mounts import Mounts\nfrom kneemri.inference import members\n",
        "core/mounts.py": "from core.data.corpus import study\n",
        "core/__init__.py": "",
        "kneemri/inference/__init__.py": "",
        "kneemri/inference/members.py": "def load():\n    from kneemri.models.model import Model\n",
        "core/data/corpus/__init__.py": "",
        "core/data/corpus/study.py": "",
        "kneemri/models/model.py": "",
        "core/data/corpus/band_coverage.py": "from core.data.corpus import study\n",
    }
    for name, source in written.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return KernelClosure(root, ENTRYPOINT)


def test_module_file(tmp_path: Path) -> None:
    """A dotted name resolves to a module, a package to its `__init__`, and a dependency to nothing."""
    closure = _repo(tmp_path)

    assert closure.module_file("core.data.corpus.study") == "core/data/corpus/study.py"
    assert closure.module_file("core.data.corpus") == "core/data/corpus/__init__.py"
    assert closure.module_file("torch") is None
    assert closure.module_file("core.data.corpus.absent") is None


def test_imported(tmp_path: Path) -> None:
    """Both readings of a `from X import y` are offered, and a deferred import is not missed."""
    closure = _repo(tmp_path)

    assert closure.imported(ENTRYPOINT) == [
        "torch",
        "core.mounts",
        "core.mounts.Mounts",
        "kneemri.inference",
        "kneemri.inference.members",
    ]
    assert closure.imported("kneemri/inference/members.py") == ["kneemri.models.model", "kneemri.models.model.Model"]


def test_files(tmp_path: Path) -> None:
    """The closure carries what the entrypoint reaches transitively, and nothing that only reaches it."""
    closure = _repo(tmp_path)

    assert closure.files() == [
        "core/data/corpus/__init__.py",
        "core/data/corpus/study.py",
        "core/mounts.py",
        "kneemri/inference/__init__.py",
        "kneemri/inference/members.py",
        "kneemri/kaggle/kernel_run.py",
        "kneemri/models/model.py",
    ]
