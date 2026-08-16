"""Predicting a whole split and writing the submission — the part both kernels do identically.

The two kernels differ in exactly one thing: which weights package they mount. Everything after that
— loading members, decoding studies in chunks, predicting, rank-combining, writing the CSV — was
duplicated line for line between them, which is two places to fix a bug found once and two places for
them to drift silently apart.

What CANNOT move here is the bootstrap: locating the mounted kit and installing its wheels has to
happen before this module can be imported at all. That part stays in the kernel, and it is the only
part that should.
"""

import importlib
import logging
import multiprocessing
import os
import sys
import time
from collections.abc import Iterator
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor

from kneemri.corpus.slot_reader import SlotReader
from kneemri.corpus.study import SlotPipeline, Study
from kneemri.inference.ensemble import Ensemble, Read
from kneemri.models.model import TARGETS

logger = logging.getLogger(__name__)

STUDY_CHUNK = 48
"""Studies decoded before a prediction pass. Bounds peak memory: a chunk of 48 at twelve slices and
336 pixels is about 390 MB, against a Kaggle host with 32 GB and four CPUs."""

BYTES_PER_GB = 1024**3

KIB_PER_GB = 1024**2
"""`ru_maxrss` is kibibytes on Linux, which is the only host this instrument reports from."""

rusage = importlib.import_module("resource") if sys.platform != "win32" else None
"""The host-memory instrument, absent on the dev machine and present on every host that runs a kernel."""

POOL_FLOOR = 20
"""Below this many studies, spawning the pool costs more than it saves. DERIVED, not picked: the Kaggle
run of 2026-08-14 decoded 3 visible studies in 6s against 3s for the serial path, so four spawned
interpreters cost about 3s; the pool saves about 0.15s per study (0.27 -> 0.12, `decode_bench`). Three
seconds buys twenty studies. A real split is thousands and always wins; the three-study visible pass and
a dry run are exactly the cases that would silently get slower."""

DECODE_WORKERS = 0
"""Decode processes; 0 means one per core. MEASURED, not assumed (`kneemri.corpus.decode_bench`, 96 studies,
warm, returning the real bags): serial 25.7s, threads(4) 19.0s, threads(8) 26.1s — the setting this
replaces, SLOWER than no pool at all — processes(4) 11.2s, processes(8) 12.8s. pydicom parses headers in
Python, so decode is GIL-bound and threads contend; processes reach 2.3x and peak at the core count,
which is what a Kaggle host has four of."""


@dataclass(frozen=True)
class Split:
    """Which tables a submission reads, and where that split's pixels live.

    One object because the two are never chosen apart, and because they arrived as a fifth and sixth
    argument to a constructor — which is the shape `Window` was created to fix elsewhere in this
    package. `test` is the competition's own split and the only one a kernel names; `train` lets the
    SAME path run over studies whose labels we hold, which is how a submission full of nonsense can be
    told apart from a crash without spending an entry.
    """

    name: str = "test"
    series_root: Path | None = None

    def series(self, competition: Path) -> Path:
        """Where the DICOM directories are — beside the tables unless a dry run says otherwise."""
        return self.series_root or competition / f"{self.name}_series"


class Submission:
    """Runs an ensemble over a split's studies and writes `submission.csv`."""

    def __init__(
        self,
        ensemble: Ensemble,
        competition: Path,
        working: Path,
        *,
        overlap: bool = True,
        split: Split | None = None,
    ) -> None:
        self.ensemble = ensemble
        self.device = ensemble.device
        self.competition = competition
        self.working = working
        self.overlap = overlap
        self.split = split or Split()

    def decode(
        self, identifiers: list[str], planes: dict[str, str], read: Read
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Decode a chunk of studies ONCE and return the batched bag at every size asked for.

        Sizes rather than one size, because the resize is the only size-dependent step: `SlotReader`
        stops before it, so a chunk read once serves 448 and 336 together and each is byte-identical to
        a native read. That turns N resolutions from N decodes into one — decode being the I/O-bound
        half of a submission on four CPUs — and it is what lets our strongest member share an entry with
        members fitted at another resolution.

        The scheme still comes from the members, never a default: a nine-slot member routed under the
        canonical six is fed three slots it never saw.

        SERIAL, and deliberately so. `chunks` is the path a submission runs and it owns the pool, one
        for the whole read; this is the direct form a test or a dry run calls for ONE chunk, and it is
        the reference the equality test measures the pooled path against. A second pool here would have
        been a second decode path to keep in step, which is what the two kernels taught.
        """
        pipeline = self.pipeline(read)
        root = self.split.series(self.competition)
        return self.batched([self.decoded_at(root / name, planes, pipeline, read) for name in identifiers], read)

    def pipeline(self, read: Read) -> SlotPipeline:
        """The reader ONE decode needs, built from what the members recorded rather than a constant.

        The scheme carries the TIE RULE, so this needs no extra field: `Ensemble.decoded_scheme` refuses
        a group whose members disagree about it, and `reads` groups by it, so a mixed entry is two
        decodes rather than one silently wrong one.

        `crop_mm` reached `SlotReader` as its module default no matter what a member's manifest said, so
        a member fitted on a different physical extent was served under 130 mm and nothing anywhere
        would have said so — every fingerprint still passes, because each is checked against its own
        reading. `Member.pixel_group` has always separated such members; `Ensemble.reads` now groups by
        the crop too, and this is where that answer finally reaches the decode.
        """
        return SlotPipeline(
            scheme=read.scheme,
            reader=SlotReader(
                crop_mm=read.crop.mm,
                image_size=max(read.sizes),
                centre=read.crop.located(),
                band=read.band,
            ),
        )

    @staticmethod
    def decoded_at(
        directory: Path, planes: dict[str, str], pipeline: SlotPipeline, read: Read
    ) -> dict[int, tuple[Tensor, Tensor]]:
        """One study's bag from its DIRECTORY, or zeros with an empty mask if it cannot be read.

        Static, and taking the directory rather than a name plus this object's split, so a PROCESS pool
        can call it: a bound method would drag the whole `Submission` — its ensemble and every loaded
        model — through a pickle for each study. `decoded` is the instance-level convenience that
        resolves a name against the split and calls this.
        """
        try:
            return Study.from_directory(directory, planes, pipeline).bags(read.slices, read.sizes)
        # Broad on purpose: pydicom, the filesystem and the router each fail in their own way, and any
        # failure to read ONE study must cost one row rather than the entry. Logged with the traceback,
        # which is what keeps this from being a silent swallow — and what BLE001 asks for.
        except Exception:
            logger.exception("study %s could not be decoded — contributing zeros and an empty mask", directory.name)
            slots = len(pipeline.scheme.slots)
            return {
                size: (torch.zeros(slots, read.slices, size, size, dtype=torch.uint8), torch.zeros(slots))
                for size in read.sizes
            }

    def decoded(
        self, name: str, planes: dict[str, str], pipeline: SlotPipeline, read: Read
    ) -> dict[int, tuple[Tensor, Tensor]]:
        """One study's bag, or zeros with an empty mask if it cannot be read at all.

        ONE UNREADABLE STUDY MUST NOT COST THE ENTRY. `pool.map` re-raises, so a single study that
        `Study.from_directory` cannot open took down the whole run — and a run that fails writes no
        submission at all, by design. Across three visible studies that never fires; across a hidden
        test set of thousands it is close to certain, which is exactly the asymmetry that makes it
        expensive to discover.

        Zeros with a ZERO MASK is the same answer `bag` already gives for a slot whose series will not
        decode: the head renormalises over what the study actually has, so an unreadable study is one
        weak row rather than a wrong one. A SYSTEMATIC failure is still caught — every study zero means
        every prediction ties, and `check` refuses an all-constant table.
        """
        return self.decoded_at(self.split.series(self.competition) / name, planes, pipeline, read)

    def batched(self, bags: list[dict[int, tuple[Tensor, Tensor]]], read: Read) -> dict[int, tuple[Tensor, Tensor]]:
        """A chunk's per-study bags stacked into one batch per decoded size."""
        return {
            size: (torch.stack([bag[size][0] for bag in bags]), torch.stack([bag[size][1] for bag in bags]))
            for size in read.sizes
        }

    def chunks(
        self, identifiers: list[str], planes: dict[str, str], read: Read
    ) -> Iterator[dict[int, tuple[Tensor, Tensor]]]:
        """Decoded chunks, with the NEXT chunk already decoding while this one is on the GPU.

        THE OVERLAP THIS DOCSTRING USED TO CLAIM DID NOT EXIST. The body was `sized = self.decode(...)`
        followed by `yield`, and `Ensemble.predict_members` consumes it with a plain `for`: so the run
        was decode(1), forward every member on (1), decode(2), forward on (2). The GPU idled through
        every decode and the CPUs idled through every forward, for the whole hidden test set.

        One pool, alive for the entire read rather than rebuilt per chunk, gives both halves at once.
        The next chunk's studies are SUBMITTED before this one is awaited, so the workers chew on it
        while the main thread is inside the forward pass — and a pool rebuilt per chunk would have paid
        process spawn plus a torch import every 48 studies, which is how this optimisation turns into a
        pessimisation.

        PROCESSES, NOT THREADS, and measured rather than assumed: `kneemri.corpus.decode_bench` puts 8 threads
        at 0.98x of running SERIALLY on 96 studies while 4 processes reach 2.29x. pydicom parses in
        Python, so decode is GIL-bound; the previous setting's stated reason — "dominated by waiting on
        the mount" — predicts the opposite of what happens.
        """
        started = time.time()
        pipeline = self.pipeline(read)
        root = self.split.series(self.competition)
        groups = [identifiers[begin : begin + STUDY_CHUNK] for begin in range(0, len(identifiers), STUDY_CHUNK)]

        def submit(pool: Executor, names: list[str]) -> list[Future[dict[int, tuple[Tensor, Tensor]]]]:
            return [pool.submit(self.decoded_at, root / name, planes, pipeline, read) for name in names]

        with self.pool(len(identifiers)) as pool:
            pending: list[Future[dict[int, tuple[Tensor, Tensor]]]] = submit(pool, groups[0]) if groups else []
            for index, group in enumerate(groups):
                # QUEUED BEFORE THE WAIT, which is the whole point: by the time the main thread returns
                # from the GPU, these have been decoding rather than waiting to be asked.
                following: list[Future[dict[int, tuple[Tensor, Tensor]]]] = (
                    submit(pool, groups[index + 1]) if index + 1 < len(groups) else []
                )
                decode_started = time.time()
                sized = self.batched([future.result() for future in pending], read)
                pending = following
                begin = index * STUDY_CHUNK
                # THE PROJECTION IS THE ENSEMBLE'S TO MAKE, not this loop's. It used to be computed here
                # as elapsed / chunks-done x chunks, which is right for ONE read group and wrong for an
                # entry that decodes twice — it projected the first decode as if it were the whole run,
                # and a fused entry serving two routers is exactly the case that runs longest.
                #
                # `index` and not `index + 1`: this loop is a generator, so by the time it resumes the
                # chunks BEFORE this one have been forwarded and this one has only been decoded. Counting
                # it as finished would credit the run with a forward pass it has not made yet, which
                # under-projects exactly when the forward is the expensive half.
                budget = self.ensemble.budget
                if budget:
                    budget.chunk(index, len(groups))
                projection = budget.report() if budget else ""
                logger.info(
                    f"  studies {begin + 1}-{begin + len(group)}: decode {time.time() - decode_started:.0f}s "
                    f"at {list(read.sizes)}, elapsed {time.time() - started:.0f}s, "
                    f"{projection}peak {self.peak_vram():.1f} GB vram, {self.peak_rss():.1f} GB host"
                )
                yield {size: (images.to(self.device), mask.to(self.device)) for size, (images, mask) in sized.items()}

    def peak_vram(self) -> float:
        """The highest allocation the card has held so far, in GB, or zero off the GPU.

        THREE ENTRIES DIED WITHOUT NAMING WHAT RAN OUT. An out-of-memory kill and a nine-hour timeout
        leave the same log — the last chunk line, then nothing — so the two are indistinguishable
        afterwards unless the run says how close it was to each ceiling while it was still alive.
        """
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.max_memory_allocated() / BYTES_PER_GB

    def peak_rss(self) -> float:
        """The highest HOST memory this process and its reaped decode workers have held, in GB.

        The other ceiling, and the one `peak_vram` alone cannot see. A host out-of-memory kill is not
        an exception the kernel can catch and log — the process is killed — so it leaves exactly the
        trace three lost entries left: the last chunk line, then nothing. Reporting only the VRAM peak
        answers the wrong half of "how close was it", since a chunk of 48 lives in host memory before
        it reaches the card and the decode pool holds its own copies besides.

        Children count because the pool is where the pixels are decoded, and `RUSAGE_CHILDREN` reports
        only workers that have EXITED — a pool that lives for the whole read contributes nothing until
        it is shut down. So this under-reports while the pool runs and is honest once it does not; it
        is a floor on the true peak, never a ceiling.
        """
        if rusage is None:
            return 0.0
        held = rusage.getrusage(rusage.RUSAGE_SELF).ru_maxrss + rusage.getrusage(rusage.RUSAGE_CHILDREN).ru_maxrss
        return held / KIB_PER_GB

    def pool(self, studies: int = POOL_FLOOR) -> Executor:
        """The decode pool: SPAWNED processes where they help, threads where they cannot be used.

        SPAWN, EXPLICITLY, and this is not a detail. Kaggle is Linux, where the default start method is
        fork, and by the time a submission decodes anything the parent has loaded every member onto the
        GPU — forking a process with an initialised CUDA context is undefined behaviour. These workers
        never touch CUDA, so a fork would PROBABLY be fine, and "probably fine" in this exact path is
        what three lost entries were made of. Spawn costs one interpreter start per worker, paid once
        because the pool lives for the whole read rather than per chunk.

        Threads are the fallback on a single core AND on a run too small to amortise the spawn — they
        produce IDENTICAL bags, which the equality test in `test_chunks` measures against the serial
        path, so the fallback costs speed and never correctness.
        """
        workers = DECODE_WORKERS or (os.cpu_count() or 1)
        if workers <= 1 or studies < POOL_FLOOR:
            return ThreadPoolExecutor(max_workers=1)
        return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))

    def predict(self, identifiers: list[str], planes: dict[str, str]) -> NDArray[np.float64]:
        """Every member's predictions over every study, rank-combined into one table.

        The grouping and combining live on `Ensemble`, which knows which members can share pixels; this
        supplies the only thing it cannot know, which is how to turn a set of studies into chunks.
        """
        return self.ensemble.predict_groups(lambda read: self.chunks(identifiers, planes, read), overlap=self.overlap)

    def run(self) -> pd.DataFrame:
        """The whole submission: decode, predict, write — and NOTHING if any of it fails.

        THERE WAS A FALLBACK HERE AND IT COST A SUBMISSION. It copied `sample_submission.csv` out
        first, "so a crash later still leaves a scoreable file", and on 2026-08-12 that is exactly what
        it left: fusion v2 did not finish on the hidden test set, the sample's constant values were
        scored, and the leaderboard reported 0.500 — chance. A slot was spent to learn nothing, and the
        number looked like a model result rather than a crash.

        A kernel that fails must produce NO submission.csv. Kaggle then fails the submission itself,
        which costs no slot and says plainly that the run broke. A fallback converts a loud failure into
        a quiet wrong answer, and on a five-a-day budget that is the more expensive of the two.

        The slice depth comes from the members, not from a default. It used to be `slices: int = 12`,
        and every kernel called `run()` — so a member fitted on a deeper stack would have been scored on
        twelve slices without a word.
        """
        test = pd.read_csv(self.competition / f"{self.split.name}.csv")
        series = pd.read_csv(self.competition / f"{self.split.name}_series.csv")
        planes = series.set_index("SeriesInstanceUID")["Anatomical_Plane"].to_dict()
        identifiers = test["StudyInstanceUID"].tolist()
        logger.info(f"{len(identifiers)} studies, {len(series)} series")

        combined = self.predict(identifiers, planes)
        frame = pd.DataFrame(self.ensemble.submission(identifiers, combined))
        self.check(frame, identifiers)
        frame.to_csv(self.working / "submission.csv", index=False)
        logger.info(f"wrote {len(frame)} rows")
        return frame

    @staticmethod
    def check(frame: pd.DataFrame, identifiers: list[str]) -> None:
        """Refuse a table that cannot be a real answer, before it is written and scored.

        Two failures are indistinguishable from a working submission by eye and both score around
        chance: a table missing studies, and a table with no ordering in it at all.

        EVERY target constant, not any of them. A single flat column is possible on a small split
        where predictions tie — and the visible test set inside a kernel is three studies, so a
        per-column rule would fail the run that has to succeed before anything can be submitted. All
        twelve flat is not a tie, it is the sample submission, which is what the removed fallback wrote
        and what scored 0.500.
        """
        if len(frame) != len(identifiers):
            message = f"submission has {len(frame)} rows for {len(identifiers)} studies"
            raise ValueError(message)
        if len(frame) > 1 and all(frame[target].nunique() == 1 for target in TARGETS):
            message = "every target is constant — that scores as chance, not as a model"
            raise ValueError(message)
