import json
from pathlib import Path

import pandas as pd

from core.data.corpus.mrnet_cached import CachedMRNet
from core.data.corpus.mrnet_corpus import ACL, CONFIDENCE, MRNET_LEVEL, PREFIX
from core.data.corpus.mrnet_labels import TableLabels
from core.data.corpus.slots import MRNET_SCHEME, Routing
from core.data.corpus.study_cache import StudyCache
from core.findings import STUDY

VALID_CASE = "1130"


def _cached(tmp_path: Path, identifiers: list[str]) -> CachedMRNet:
    """A cache whose PIXELS an earlier run built, represented by the index that run recorded.

    The tables are unpadded and carry a header, which is how a portal serves them and is the shape the
    case-number join has to survive.
    """
    directory = tmp_path / "tables"
    directory.mkdir()
    (directory / "mrnet-train-acl.csv").write_text("case,acl\n0,0\n1,1\n", encoding="utf-8")
    (directory / "mrnet-valid-acl.csv").write_text(f"case,acl\n{int(VALID_CASE)},1\n", encoding="utf-8")
    cache = StudyCache(tmp_path, slices=4, image_size=8, routing=Routing(scheme=MRNET_SCHEME))
    cache.index_path.write_text(json.dumps(identifiers), encoding="utf-8")
    return CachedMRNet(cache, TableLabels(directory))


def test_verdicts(tmp_path: Path):
    """Both downloadable splits come back as ONE table, keyed by the padded case the identifiers carry."""
    assert _cached(tmp_path, []).verdicts() == {"0000": 0, "0001": 1, VALID_CASE: 1}


def test_volumes(tmp_path: Path):
    """The index is the join: the cache's own order is kept, and a cached exam with no label is dropped.

    Reading the label tables as the denominator instead would write rows for exams whose pixels are not
    in the array, and every downstream join against that array is positional.
    """
    identifiers = [f"{PREFIX}0000", f"{PREFIX}9999", f"{PREFIX}{VALID_CASE}", f"{PREFIX}0001"]

    rows = _cached(tmp_path, identifiers).volumes()

    assert rows == [
        (f"{PREFIX}0000", MRNET_LEVEL[0]),
        (f"{PREFIX}{VALID_CASE}", MRNET_LEVEL[1]),
        (f"{PREFIX}0001", MRNET_LEVEL[1]),
    ], "the cache's order, with the unlabelled case dropped"


def test_labels(tmp_path: Path):
    """The cached producer writes the table `MRNetCorpus.labels` would have, with no archive present."""
    destination = tmp_path / "labels" / "mrnet_acl.csv"

    _cached(tmp_path, [f"{PREFIX}0000", f"{PREFIX}0001"]).labels(destination)

    table = pd.read_csv(destination)
    assert list(table.columns) == [STUDY, ACL, CONFIDENCE]
    assert table[STUDY].tolist() == [f"{PREFIX}0000", f"{PREFIX}0001"]
    assert table[ACL].tolist() == [MRNET_LEVEL[0], MRNET_LEVEL[1]]
