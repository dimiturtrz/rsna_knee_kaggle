"""The stub half of a stranger's import chain, checked on a machine that has never run `git clone`.

`atlas_class` itself is not tested here and says so in its own docstring: it imports a module this repo
does not vendor, so a test of it would assert that somebody made the checkout. What IS testable is the
`sys.modules` state it installs first, and both of this module's failures have been exactly there.
"""

import sys

from kneemri.inference.atlas_import import ABSENT, STUB_FILE, AtlasImport


def test_install_stubs():
    """Every absent package lands in `sys.modules` with the two attributes import machinery reads."""
    AtlasImport.install_stubs()

    for name in ABSENT:
        assert name in sys.modules, f"{name} must be importable or miniclip's import chain dies"
        assert sys.modules[name].__file__ == STUB_FILE, "inspect.getmodule reads __file__ on every entry"
        assert hasattr(sys.modules[name], "__path__"), "a package stub without __path__ breaks submodules"


def test_stub_attribute():
    """Any attribute of a stubbed package answers with a class that can be both CALLED and SUBCLASSED."""
    AtlasImport.install_stubs()

    resolved = sys.modules[ABSENT[0]].whatever_miniclip_asks_for

    assert isinstance(resolved, type), "miniclip subclasses these names, so an instance would not survive"
    assert resolved()() is not None, "and it calls them, so the instance has to be callable too"
