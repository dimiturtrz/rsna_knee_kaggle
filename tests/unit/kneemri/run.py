from pathlib import Path

from kneemri.experiments.default_drift import DEFAULT
from kneemri.run import DRY_HEAD, LIST_HEAD, SEPARATOR, UNRUN, RecipeTree


def _tree(root: Path) -> RecipeTree:
    (root / "base.toml").write_text("[optim]\nepochs = 12\n", encoding="utf-8")
    (root / "trunks").mkdir()
    (root / "trunks" / "probe.toml").write_text("[data]\nimage_size = 224\n", encoding="utf-8")
    family = root / "ctl"
    family.mkdir()
    (family / "_family.toml").write_text(
        'extends = ["../base.toml", "../trunks/probe.toml"]\n[meta]\nstatus = "done"\n', encoding="utf-8"
    )
    (family / "ctl_fast.toml").write_text(
        'extends = ["_family.toml"]\n[encoder]\nbackbone_lr = 1e-4\n'
        '[meta]\nhypothesis = "faster"\nbead = "knee-a"\nout = "ctl_fast_s2026"\n',
        encoding="utf-8",
    )
    other = root / "abl"
    other.mkdir()
    (other / "abl_slow.toml").write_text('[meta]\nstatus = "queued"\n', encoding="utf-8")
    return RecipeTree(root)


def test_leaves(tmp_path: Path):
    """Runnable recipes only: no family file, no trunk, no top-level layer."""
    tree = _tree(tmp_path)

    assert tree.leaves() == [tmp_path / "abl" / "abl_slow.toml", tmp_path / "ctl" / "ctl_fast.toml"]


def test_runs(tmp_path: Path):
    """The runs table keyed by name, and empty where no table exists."""
    table = tmp_path / "runs.tsv"
    table.write_text("name\tbest_auc\tepochs\nctl_fast_s2026\t0.8360\t12\n", encoding="utf-8")

    row = {"name": "ctl_fast_s2026", "best_auc": "0.8360", "epochs": "12"}
    assert RecipeTree.runs(table) == {"ctl_fast_s2026": row}
    assert RecipeTree.runs(tmp_path / "absent.tsv") == {}


def test_rows(tmp_path: Path):
    """Meta joined to the run its `out` names; filters by status and by family; an unrun recipe reads `-`."""
    tree = _tree(tmp_path)
    runs = {"ctl_fast_s2026": {"best_auc": "0.8360", "epochs": "12"}}
    head = SEPARATOR.join(LIST_HEAD)

    assert tree.rows(runs, None, None) == [
        head,
        SEPARATOR.join(("abl_slow", "queued", "", UNRUN, UNRUN, "")),
        SEPARATOR.join(("ctl_fast", "done", "knee-a", "0.8360", "12", "faster")),
    ]
    assert tree.rows(runs, "done", None) == tree.rows(runs, None, "ctl")
    assert tree.rows(runs, "refuted", None) == [head]


def test_dry(tmp_path: Path):
    """Every option with its value and the layer that set it; untouched options read as the default."""
    tree = _tree(tmp_path)

    lines = tree.dry(tmp_path / "ctl" / "ctl_fast.toml")
    table = {line.split(SEPARATOR)[0]: line.split(SEPARATOR)[1:] for line in lines[1:]}

    assert lines[0] == SEPARATOR.join(DRY_HEAD)
    assert table["epochs"] == ["12", str(tmp_path / "base.toml")]
    assert table["image_size"] == ["224", str(tmp_path / "trunks" / "probe.toml")]
    assert table["backbone_lr"] == ["0.0001", str(tmp_path / "ctl" / "ctl_fast.toml")]
    assert table["mixup"][1] == DEFAULT
