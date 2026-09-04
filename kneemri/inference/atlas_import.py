"""Importing a stranger's trunk class out of a pinned checkout, with the deps it never calls stubbed.

Separated from `pillar_member` because it is a different subject: that file is a member, this one is an
import. The split is also what makes the bitten half testable — every failure this module exists to
prevent surfaced far away from its own frame, inside `inspect` or inside `torchvision`'s operator
registration, and a unit test can only pin them where the stub is built.

    git clone https://github.com/YalaLab/pillar-pretrain external/pillar-pretrain
    git -C external/pillar-pretrain checkout ab06083

`external/` is gitignored, so the checkout is reproduced from that pair and never vendored.
"""

import importlib
import sys
import types
from pathlib import Path

from torch import nn

ATLAS_MODULE = "miniclip.multimodal_atlas"
"""Where the trunk class lives. The archive's README names `CLIPMultimodalAtlas`; the class in the pinned
tree is `MultiModalAtlas`, and following the README instead costs an import error at member-build time."""

ATLAS_CLASS = "MultiModalAtlas"

SOURCE = Path("external/pillar-pretrain/src")

ABSENT = ("lifelines", "lifelines.utils", "lifelines.utils.btree", "torchmetrics", "torchmetrics.functional", "ftfy")
"""Packages `miniclip/__init__` imports on the way to the trunk, none of which the trunk itself calls:
survival metrics, torchmetrics wrappers and a text-encoder unicode fixer. They are stubbed rather than
installed because installing them would put a stranger's metric code in the environment that scores our
members, to satisfy an import that no forward ever reaches."""

STUB_FILE = "<stub>"
"""A stubbed module needs a real `__file__`. `inspect.getmodule` walks every entry of `sys.modules` and
reads that attribute, so a stub which answers it through `__getattr__` hands `inspect` a class where it
expects a path — and the failure surfaces as an `AttributeError` inside `torchvision`'s operator
registration, in a frame that has nothing to do with this file."""


class AtlasImport:
    """The publisher's trunk class, and the `sys.modules` state its import chain has to walk through."""

    class _Absent:
        """Whatever a stubbed package is asked for: a class, because `miniclip` both SUBCLASSES and CALLS them.

        Permissive rather than raising, and that is deliberate but narrow: the names resolved this way are
        survival-analysis and metric helpers reached only from `miniclip`'s import chain, and the scope is the
        fixed `ABSENT` tuple. A blanket auto-stub is what this replaced — it swallowed a name `torchvision`
        needed and produced an `AttributeError` from inside `inspect`, three frames deep in unrelated code.
        """

        def __init__(self, *arguments: object, **keywords: object) -> None:
            del arguments, keywords

        def __getattr__(self, name: str) -> type["AtlasImport._Absent"]:
            del name
            return AtlasImport._Absent

        def __call__(self, *arguments: object, **keywords: object) -> "AtlasImport._Absent":
            del arguments, keywords
            return self

    @staticmethod
    def _unreachable(name: str) -> type["AtlasImport._Absent"]:
        """Any attribute of a stubbed package, which is always the same permissive class."""
        del name
        return AtlasImport._Absent

    @staticmethod
    def install_stubs() -> None:
        """Put a permissive placeholder in `sys.modules` for each package in `ABSENT`, if not already real.

        Separated from `atlas_class` because this half is the part that has bitten, and it is the part that
        can be checked without a third-party checkout: a stub missing `__path__` breaks import machinery, and
        one missing `__file__` breaks `inspect` from inside unrelated libraries. Both failures surfaced far
        from here, and both are assertions a unit test can make in isolation.
        """
        for name in ABSENT:
            if name not in sys.modules:
                stub = types.ModuleType(name)
                stub.__path__ = []
                stub.__file__ = STUB_FILE
                stub.__getattr__ = AtlasImport._unreachable
                sys.modules[name] = stub

    @staticmethod
    def atlas_class() -> type[nn.Module]:  # devtools-ignore: test-mirror
        """The publisher's trunk class, imported from the pinned checkout with its unused deps stubbed.

        Imported through `importlib` and behind a function because the module does not exist until the
        checkout above has been made, and a top-level import would make every other member in this package
        unimportable on a machine that has not cloned it. That is also why it carries no mirror test: the
        only thing it does beyond `install_stubs` is import a module this repo does not vendor, so a unit
        test of it would assert that somebody ran `git clone`. The reviewable half is tested; this half is
        exercised by `pillar_gate`, which cannot run at all if it is wrong.
        """
        AtlasImport.install_stubs()
        if str(SOURCE) not in sys.path:
            sys.path.insert(0, str(SOURCE))
        return getattr(importlib.import_module(ATLAS_MODULE), ATLAS_CLASS)
