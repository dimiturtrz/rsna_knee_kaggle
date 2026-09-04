"""Every kernel's declared mounts against the members it names, swept without running anything."""

import json
from pathlib import Path

from kneemri.inference.members import MemberPackage
from kneemri.kaggle.mount_check import MEMBERS_DATASET, EncoderMounts, LiveReading, MountCheck, MountSweep

DINOV2 = "metaresearch/dinov2/PyTorch/small/1"
DINOV3 = "dimiturnt/dinov3-hf"


def _package(tmp_path: Path) -> MemberPackage:
    """Two members on two different encoders, which is the only property these checks read."""
    root = tmp_path / "members"
    root.mkdir()
    entries = []
    for identifier, variant in (("v2member", "small"), ("v3member", "v3small")):
        (root / f"{identifier}.pt").write_bytes(b"")
        entries.append(
            {
                "id": identifier,
                "file": f"{identifier}.pt",
                "fold": 0,
                "config": {"variant": variant},
                "fingerprint": [0.0],
                "holdout": 0.84,
                "annot": 0.87,
            }
        )
    (root / "manifest.json").write_text(json.dumps({"format": 2, "members": entries}), encoding="utf-8")
    return MemberPackage(root)


def _kernel(tmp_path: Path, name: str, body: str, sources: list[str]) -> Path:
    kernel = tmp_path / name
    kernel.mkdir()
    (kernel / f"{name}.py").write_text(body, encoding="utf-8")
    (kernel / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": f"dimiturnt/{name}",
                "model_sources": [DINOV2],
                "dataset_sources": [MEMBERS_DATASET, *sources],
            }
        ),
        encoding="utf-8",
    )
    return kernel


def test_entries(tmp_path: Path):
    """A directory qualifies on metadata OR on code — the second half is what dark7's silence needed.

    Keying only on the metadata dropped a written-but-unpushed kernel out of the sweep entirely, which
    reads exactly like a kernel with nothing to say. An empty directory is still not a kernel.
    """
    _kernel(tmp_path, "kneemri-a", "KernelRun().run()\n", [DINOV3])
    (tmp_path / "notakernel").mkdir()
    (tmp_path / "kneemri-unpushed").mkdir()
    (tmp_path / "kneemri-unpushed" / "kneemri-unpushed.py").write_text("KernelRun().run()\n", encoding="utf-8")
    names = [kernel.name for kernel in MountCheck(_package(tmp_path)).entries(tmp_path)]
    assert names == ["kneemri-a", "kneemri-unpushed"], (
        "code with no metadata is a kernel the sweep must SAY something about"
    )


def test_verdict_refuses_a_kernel_with_no_metadata(tmp_path: Path):
    """And what it says is a REFUSAL: `push.sh kernel` cannot push it, and the slot clock is running."""
    (tmp_path / "kneemri-unpushed").mkdir()
    (tmp_path / "kneemri-unpushed" / "kneemri-unpushed.py").write_text("KernelRun().run()\n", encoding="utf-8")
    verdict = MountCheck(_package(tmp_path)).verdict(tmp_path / "kneemri-unpushed")
    assert verdict.startswith("REFUSED"), "not a skip - a skip is what silence already sounded like"
    assert "kernel-metadata.json" in verdict


def test_verdict(tmp_path: Path):
    """A named subset is judged on ITS encoders; an unresolvable selection on the whole package.

    The refusal is the case that already happened: `kneemri-solo` runs `strongest(1)` over a package
    that gained a DINOv3 member, and its metadata still declared DINOv2 only — every fingerprint agrees
    and the kernel cannot build the backbone.
    """
    check = MountCheck(_package(tmp_path))

    subset = _kernel(tmp_path, "kneemri-subset", "KernelRun(names=('v2member',)).run()\n", [])
    assert check.verdict(subset).startswith("ok"), "a DINOv2-only subset needs no DINOv3 mount"

    rehearsal = _kernel(tmp_path, "kneemri-rehearsal", "KernelRun(names=('v2member',)).run()\n", [])
    (rehearsal / "kernel-metadata.json").write_text(
        json.dumps({"id": "dimiturnt/kneemri-rehearsal", "dataset_sources": ["dimiturnt/kneemri-ortho-pkg"]}),
        encoding="utf-8",
    )
    assert check.verdict(rehearsal).startswith("skipped"), (
        "a kernel mounting ANOTHER weights package is not this sweep's to refuse - orthorehearse read "
        "as a refusal for a day because its one-member dataset is absent from the shipped package"
    )

    ranked = _kernel(tmp_path, "kneemri-ranked", "KernelRun(members=1).run()\n", [])
    assert "does not declare dimiturnt/dinov3-hf/vits16" in check.verdict(ranked)

    declared = _kernel(tmp_path, "kneemri-declared", "KernelRun(members=1).run()\n", [DINOV3])
    assert check.verdict(declared).startswith("ok")

    absent = _kernel(tmp_path, "kneemri-absent", "KernelRun(names=('gone',)).run()\n", [DINOV3])
    assert check.verdict(absent).startswith("REFUSED"), "a name the package does not hold is a finding too"

    foreign = _kernel(tmp_path, "kneemri-foreign", "MemberPackage.find(mounts)\n", [])
    assert check.verdict(foreign).startswith("skipped"), "a kernel running someone else's package is not checkable"


def test_unpushed(tmp_path: Path):
    """The failure of 2026-08-19: a member packaged locally, named by a kernel, never pushed."""
    package = _package(tmp_path)
    assert MountCheck(package).unpushed(["v3member"]) == [], "without --live there is nothing to compare against"

    check = MountCheck(package, live={"v2member"})
    assert check.unpushed(["v3member"]) == ["v3member"]
    assert check.unpushed(["v2member"]) == []
    unnamed = "an unnamed selection ranks the whole package, so all of it must be live"
    assert check.unpushed(None) == ["v3member"], unnamed


def test_absent(tmp_path: Path):
    """The failure of 2026-08-25: a member the manifest names, live nowhere, and named by NO kernel.

    `unpushed` is per-kernel and cannot see it — the one it misses is exactly the one that matters,
    because `MemberPackage` refuses a manifest whose files it cannot all find and takes every kernel
    down with it.
    """
    package = _package(tmp_path)
    assert MountCheck(package).absent() == [], "without --live there is nothing to compare against"

    check = MountCheck(package, live={"v2member"})
    assert check.absent() == ["v3member"], "a manifest row with no live file, regardless of who names it"
    assert MountCheck(package, live={"v2member", "v3member"}).absent() == []


def test_settled(tmp_path: Path):
    """A flip-flopping listing is a REFUSAL: one green read of a materialising version is not proof."""
    package = _package(tmp_path)
    check = MountCheck(package)
    reads: list[set[str]] = []
    waits: list[str] = []

    def listing() -> set[str]:
        return reads.pop(0)

    def pause() -> None:
        waits.append("slept")

    both = {"v2member", "v3member"}
    reads[:] = [{"v2member"}, both, both]
    assert check.settled(listing, pause=pause) == [], "two consecutive complete reads settle it"
    assert (len(waits), reads) == (2, []), "and it stops reading, and sleeping, the moment they do"

    reads[:] = [both, {"v2member"}, both, both]
    flip = "one green read then an absence is not settled - it takes two consecutive ones after it"
    assert check.settled(listing, pause=pause) == [], flip
    assert reads == [], flip

    reads[:] = [{"v2member"}, both, {"v3member"}]
    union = "a listing that never settles reports every member seen absent, not the last read alone"
    assert check.settled(listing, attempts=3, pause=pause) == ["v2member", "v3member"], union


def test_verdict_refuses_an_unpushed_member(tmp_path: Path):
    """And it reads as a REFUSAL on the kernel, not as a mount problem — the mounts are fine."""
    check = MountCheck(_package(tmp_path), live={"v2member"})
    kernel = _kernel(tmp_path, "kneemri-unpushed", "KernelRun(names=('v3member',)).run()\n", [DINOV3])
    verdict = check.verdict(kernel)
    assert verdict.startswith("REFUSED")
    assert "LIVE members dataset" in verdict


def _listable(tmp_path: Path, *pages: str) -> tuple[MemberPackage, list[str]]:
    """A package that can be asked about live, and the CSV pages the fake CLI will answer with."""
    package = _package(tmp_path)
    (package.root / "dataset-metadata.json").write_text(json.dumps({"id": "dimiturnt/members"}), encoding="utf-8")
    return package, list(pages)


def test_taken(tmp_path: Path):
    """ONE REP PER CLASS of what asking the live dataset can ANSWER: not asked, read, and UNKNOWN.

    The third class is the one that had no test: the truncated listing used to call `sys.exit(2)` from
    the middle of `main`, so the only way to reach it was a real Kaggle round trip. As a state it is a
    value, and a partial listing stays neither a refusal nor a pass — `live` is None, not a short set
    that would read as members the dataset lacks.
    """
    package, pages = _listable(tmp_path)
    complete = "name,size\nv2member.pt,1\nv3member.pt,1\n"

    unasked = LiveReading.taken(package, live=False, settle=False)
    assert (unasked.live, unasked.unknown) == (None, ""), "no flag means no round trip and no verdict"

    pages[:] = [complete]
    read = LiveReading.taken(package, live=True, settle=False, listing=lambda _: pages.pop(0))
    assert (read.live, read.unsettled, read.unknown) == ({"v2member", "v3member"}, [], "")

    pages[:] = ["Next Page Token = abc\nname,size\nv2member.pt,1\n"]
    truncated = LiveReading.taken(package, live=True, settle=False, listing=lambda _: pages.pop(0))
    assert truncated.live is None, "a partial listing must not be read as a dataset missing members"
    assert "listed more than" in truncated.unknown


def test_taken_polls_until_settled(tmp_path: Path):
    """`--settle` implies `--live`, and a truncation part-way THROUGH the poll is the same UNKNOWN."""
    package, pages = _listable(tmp_path)
    complete = "name,size\nv2member.pt,1\nv3member.pt,1\n"
    pages[:] = ["name,size\nv2member.pt,1\n", complete, complete]
    settled = LiveReading.taken(package, live=False, settle=True, listing=lambda _: pages.pop(0), pause=lambda: None)
    assert (settled.unsettled, settled.unknown) == ([], ""), "settle alone asks, and two clear reads settle it"

    pages[:] = [complete, "Next Page Token = abc\nname,size\n", complete]
    mid = LiveReading.taken(package, live=False, settle=True, listing=lambda _: pages.pop(0), pause=lambda: None)
    assert mid.live is None and "listed more than" in mid.unknown, "the poll's reads are inside the same try"


def _swept(tmp_path: Path) -> MountSweep:
    """One kernel of each bucket, plus a directory the sweep must walk past rather than judge."""
    _kernel(tmp_path, "kneemri-ok", "KernelRun(names=('v2member',)).run()\n", [DINOV3])
    _kernel(tmp_path, "kneemri-refused", "KernelRun(names=('v3member',)).run()\n", [])
    skipped = tmp_path / "kneemri-skipped"
    skipped.mkdir()
    (skipped / "kneemri-skipped.py").write_text("print('no KernelRun here')\n", encoding="utf-8")
    (skipped / "kernel-metadata.json").write_text(json.dumps({"id": "dimiturnt/s"}), encoding="utf-8")
    (tmp_path / "notakernel").mkdir()
    return MountSweep.over(MountCheck(_package(tmp_path)), tmp_path)


def test_mount_sweep_over(tmp_path: Path):
    """Every entry judged, and the WALKED count read from the directory rather than from the verdicts.

    Those two numbers are the point of the class: the buckets are computed from the verdicts and so can
    never disagree with them, while `walked` can — it is the only number that could have shown dark7's
    silence, a directory the sweep passed over without ever becoming an entry.
    """
    sweep = _swept(tmp_path)

    assert set(sweep.verdicts) == {"kneemri-ok", "kneemri-refused", "kneemri-skipped"}
    assert (sweep.checked, sweep.walked) == (3, 5), "the members dir and notakernel are walked, not checked"


def test_refused(tmp_path: Path):
    """The bucket that decides the exit code, keyed on the word the verdict starts with."""
    assert _swept(tmp_path).refused == ["kneemri-refused"]


def test_skipped(tmp_path: Path):
    """Unanswerable is not ok — a kernel building no KernelRun picks its members some other way."""
    assert _swept(tmp_path).skipped == ["kneemri-skipped"]


def test_passed(tmp_path: Path):
    """The honest denominator: checked and cleared, never the count of verdicts printed."""
    assert _swept(tmp_path).passed == 1, "one ok of three verdicts - the skip is not a pass"


def test_lines(tmp_path: Path):
    """Each verdict on its own row, then a tally that names what was walked past."""
    lines = _swept(tmp_path).lines(tmp_path)

    assert len(lines) == 4, "three verdicts and one tally"
    assert lines[-1].startswith("1 ok, 1 skipped, 1 refused of 3 kernel directories")
    assert "(2 others" in lines[-1], "the directories the sweep declined to judge are counted out loud"


def test_encoder_mounts_over():
    """A mount is unresolved when its dataset does not publish the subpath the mount names.

    The flat case is `farseat_v1`'s: the dataset exists, the kernel declares it, and the leaf the
    encoder resolves on was never published — so the member is dropped at run time and the entry
    scores as a smaller ensemble than the one that was designed.
    """
    nested = "name,size,creationDate\nvits16/config.json,785,2026-08-23 12:42:39\n"
    flat = "name,size,creationDate\nconfig.json,785,2026-08-24 12:42:39\n"

    assert EncoderMounts.over(["dimiturnt/dinov3-hf/vits16"], lambda _: nested).missing == {}
    assert EncoderMounts.over(["dimiturnt/vjepa2-hf"], lambda _: flat).missing == {}
    assert EncoderMounts.over(["dimiturnt/vjepa2-hf/vjepa2-vitl"], lambda _: flat).missing == {
        "dimiturnt/vjepa2-hf/vjepa2-vitl": "dimiturnt/vjepa2-hf"
    }
