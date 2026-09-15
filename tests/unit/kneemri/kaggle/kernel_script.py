"""Which members a kernel script asks for, parsed rather than run."""

from pathlib import Path

from kneemri.kaggle.kernel_script import KernelScript


def _script(tmp_path: Path, body: str) -> KernelScript:
    written = tmp_path / "kernel.py"
    written.write_text(body, encoding="utf-8")
    return KernelScript(written)


def test_call(tmp_path: Path):
    """The `KernelRun(...)` a script builds, and None when it builds none rather than a guess."""
    assert _script(tmp_path, "KernelRun(members=1).run()\n").call() is not None
    assert _script(tmp_path, "MemberPackage.find(mounts)\n").call() is None, "graft picks members another way"


def test_member_names(tmp_path: Path):
    """A literal selection resolves — through one variable — and a ranked one reads as the whole package.

    `members=N` is `strongest(N)` over recorded holdouts, which a parse cannot evaluate, so None is the
    answer that keeps the check conservative: every member is a candidate and every mount must be there.
    """
    assert _script(tmp_path, "KernelRun(names=('a', 'b'), overlap=False).run()\n").member_names() == ["a", "b"]

    indirect = "chosen = ('res392_ws_s2026', 'sliceattn_s2026')\nKernelRun(names=chosen).run()\n"
    assert _script(tmp_path, indirect).member_names() == ["res392_ws_s2026", "sliceattn_s2026"], (
        "the chosen entry names its members through a variable"
    )

    assert _script(tmp_path, "KernelRun(members=2).run()\n").member_names() is None
    assert _script(tmp_path, "KernelRun().run()\n").member_names() is None
    assert _script(tmp_path, "KernelRun(names=picked()).run()\n").member_names() is None, "not a literal"


def test_foreign_seats(tmp_path: Path):
    """Every way a kernel names its raptor checkpoint resolves to the board's name for it; no call is no seat."""
    default = "RaptorMember.loaded(RaptorEntry.weights(mounts=mounts), device)\n"
    assert _script(tmp_path, default).foreign_seats() == {"raptor_v4_swa"}, "no filename is DEFAULT_WEIGHTS"

    literal = 'RaptorEntry.weights("raptor_ft_effv2l480.pt", mounts)\n'
    assert _script(tmp_path, literal).foreign_seats() == {"raptor_ft_effv2l480"}

    variable = 'WEIGHTS = "raptor_ft_repl2x2_a_swa.pt"\nRaptorEntry.weights(WEIGHTS, mounts)\n'
    assert _script(tmp_path, variable).foreign_seats() == {"raptor_ft_repl2x2_a_swa"}

    looped = (
        'FOREIGN = ("raptor_ft_coatnet_v4_full_swa.pt", "raptor_ft_effv2l480.pt")\n'
        "kept = [name for name in directories]\n"
        "members = {name: RaptorEntry.weights(name, mounts=mounts) for name in FOREIGN}\n"
    )
    assert _script(tmp_path, looped).foreign_seats() == {"raptor_v4_swa", "raptor_ft_effv2l480"}, (
        "the loop AROUND the call binds the name, not another loop reusing it"
    )

    assert _script(tmp_path, "KernelRun(names=('a',)).run()\n").foreign_seats() == set()
    assert _script(tmp_path, "RaptorEntry.weights(picked(), mounts)\n").foreign_seats() == set(), "not a literal"
