"""One member's probabilities for a fixed set of studies — the half of an analysis that needs a GPU.

Separated from `member_curve` because the fitness gate is right that they are two subjects, and because
the same loop was already open-coded in a throwaway slice-count script. An analysis that loads members
and reads them over test-time windows is a recurring shape here, not a one-off.
"""

import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy.typing import NDArray
from torch import Tensor

from core.findings import TARGETS
from kneemri.inference.holdout_slab import HoldoutSlab
from kneemri.inference.members import Member, Window
from kneemri.inference.norm_adaptation import Adaptation, NoAdaptation
from kneemri.inference.prediction_cache import PredictionCache
from kneemri.inference.test_time import SpatialViews
from kneemri.inference.trainer_record import TrainerRecord
from kneemri.models.backbone import ENCODERS
from kneemri.models.model import StudyModel

logger = logging.getLogger(__name__)

BATCH = 8


@dataclass(frozen=True)
class ServedModel:
    """A loaded member with the window it reads and the arithmetic it was fitted in.

    The three travel together everywhere a member is READ rather than trained, and none of them is
    optional to the reading: the window decides which slices the encoder sees, and the precision is the
    member's OWN - a bf16 member read in fp16 is a different function of the same pixels.
    """

    model: StudyModel
    window: Window
    precision: torch.dtype = torch.float16

    def autocast(self, device: torch.device) -> torch.autocast:
        """The member's own arithmetic on the card; a no-op anywhere else."""
        return torch.autocast("cuda", dtype=self.precision, enabled=device.type == "cuda")

    def crop(self, view: UInt8[Tensor, "b n s h w"], start: int) -> UInt8[Tensor, "b n k h w"]:
        """The window's slices from `start` - the stack this member reads at one window start."""
        return view[:, :, start : start + self.window.width]

    def logits(
        self, view: UInt8[Tensor, "b n s h w"], flags: Float[Tensor, "b n"], start: int
    ) -> Float[Tensor, "b labels"]:
        """One forward at one window start, at the member's own image size."""
        return self.model(self.crop(view, start), flags, self.window.size)


class HoldoutPredictions:
    """Each checkpoint's probabilities for one fixed set of studies.

    Separate from the arithmetic below because it is the half that needs a GPU, a cache and 90 seconds;
    keeping the curve itself a pure function of arrays is what lets the curve be tested at all.
    """

    def __init__(
        self, device: torch.device, views: "SpatialViews | None" = None, adaptation: Adaptation | None = None
    ) -> None:
        self.device = device
        # The default changes no number and costs nothing: `NoAdaptation.on` yields immediately and its
        # `episode` of 0 leaves the batch size at BATCH, so a caller that names no adaptation runs the
        # loop that produced every figure on the board.
        self.adaptation: Adaptation = adaptation or NoAdaptation()
        # The SAME views the submission path uses, or none. A whole-stack member gets one window, so
        # window averaging - worth a measured +0.0156 to the windowed members - is unavailable to it and
        # rotation is the only test-time average left. The ruler has to apply whatever deployment will,
        # or it ranks members under a policy they will not ship under.
        self.views = views or SpatialViews(degrees=(0.0,), shifts=((0.0, 0.0),))
        self.record = TrainerRecord(self.views)

    def load(self, checkpoint: Path, backbone: torch.nn.Module, dim: int, slots: int) -> StudyModel:
        """One member, rebuilt in the shape its own config records.

        The shape comes from the checkpoint's own config rather than from this method's assumptions: a
        member trained with per-slice max pooling is a different function from one trained with channel
        stacking, and loading either under the wrong rule would silently produce a model whose weights
        no longer mean what they did.

        The backbone is passed in rather than fetched here. Constructing it internally would have meant
        this method could only run with a network and a 90 MB download, which is to say it could not be
        tested at all — the mirror gate said so, and it was right.
        """
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        # THE SHARED REBUILD, not this method's own reading of the config. Every switch was read back
        # here except the two that decide the head's WIDTH and whether the encoder carries adapters, so
        # this could not load a focally pooled or a LoRA member at all - and the serve path, which read
        # those two, was the half missing the prior. One rebuild has one set of blind spots.
        model = StudyModel.from_config(saved.get("config", {}), backbone, dim, slots)
        model.load_trained(saved["model"])
        return model.to(self.device).eval()

    @staticmethod
    def fits(window: "Window", depth: int) -> None:
        """Refuse a window the served stack cannot carry, instead of scoring whatever it overlaps.

        The window is sized by the CALLER against the depth it believes the slab serves, and a slab is
        allowed to serve a different one — so a caller that asks the wrong source produces a read that
        is short rather than wrong-looking. Off the end it raises deep inside the encoder as a reshape
        of zero elements, which names neither the window nor the slab; short of the end it raises
        nothing at all and returns a number. The second is the dangerous one and it is why this checks
        every start rather than only the last: six members were scored on eight slices of twenty and
        the only symptom was a plausible drop.
        """
        over = [start for start in window.starts if start + window.width > depth]
        if window.width <= 0 or over:
            message = (
                f"a window of width {window.width} at starts {window.starts} does not fit a stack "
                f"{depth} slices deep ({len(over)} start(s) run past the end) - the window was sized "
                f"against a different depth than the slab serves"
            )
            raise ValueError(message)

    def predict(
        self,
        model: StudyModel,
        slab: HoldoutSlab,
        window: "Window",
        precision: torch.dtype = torch.float16,
    ) -> NDArray[np.float64]:
        """Probabilities for every study, averaged over the given test-time windows.

        Reads the studies a batch at a time out of the slab rather than being handed the whole holdout:
        the array this used to take was 12.8 GB at 336 px and 22.8 at 448, resident for a walk that
        never needed more than eight studies of it. `HoldoutSlab.of` wraps arrays a caller already
        holds, so a caller that genuinely has pixels rather than a corpus is unaffected.

        `width` is how many slices each window covers, and it defaults to the three-slice group only
        because that is what a windowed member reads. A WHOLE-STACK member's slot embedding is the
        elementwise maximum over its entire stack, so it must be given one window of full width — this
        was hardcoded to GROUP, which windowed such a member by three and quietly scored a different
        function. The same defect shipped to the leaderboard through `Ensemble` (knee-y8y); this is the
        analysis path, where it decides which members we choose to submit.
        """
        served = ServedModel(model, window, precision)
        chunks = []
        for images, mask in slab.batches(self.adaptation.episode or BATCH):
            pixels: UInt8[Tensor, "b n s h w"] = torch.from_numpy(np.ascontiguousarray(images)).to(self.device)
            flags: Float[Tensor, "b n"] = torch.from_numpy(mask).to(self.device)
            graded = partial(self._logits, served, pixels, flags)
            with self.adaptation.on(model, graded):
                chunks.append(self._average(served, pixels, flags))
        return np.concatenate(chunks).astype(np.float64)

    def _average(
        self,
        served: ServedModel,
        pixels: UInt8[Tensor, "b n s h w"],
        flags: Float[Tensor, "b n"],
    ) -> NDArray[np.float32]:
        """The test-time average that produced every published figure, unchanged and still ungraded."""
        # The member's OWN arithmetic, not autocast's cuda default of float16: this is the tool that
        # decides which members to submit, so scoring a bf16 member in fp16 ranks it under a function
        # it was never fitted as. Same defect as Ensemble.predict_member carried until today.
        with torch.no_grad(), served.autocast(self.device):
            total = torch.zeros(pixels.shape[0], len(TARGETS), device=self.device)
            passes = 0
            for view in self.views.apply(pixels):
                self.fits(served.window, view.shape[2])
                for start in served.window.starts:
                    total = total + torch.sigmoid(served.logits(view, flags, start).float())
                    passes += 1
            return (total / passes).cpu().numpy()

    def _logits(
        self,
        served: ServedModel,
        pixels: UInt8[Tensor, "b n s h w"],
        flags: Float[Tensor, "b n"],
    ) -> Float[Tensor, "studies labels"]:
        """One GRADED read for the adapt pass — the first view at the first start, not the average.

        NOT THE AVERAGE, and that is a cost decision stated rather than hidden: the average is
        views x starts forwards, and putting a backward through all of them would multiply the adapt
        pass by that product for a signal that is the same function of the same weights. The unrotated
        view at the first start is the one every member is fitted on; the rotations are a test-time
        average, not a different reading.
        """
        view = next(iter(self.views.apply(pixels)))
        self.fits(served.window, view.shape[2])
        with served.autocast(self.device):
            return served.logits(view, flags, served.window.starts[0])

    def cached(
        self, reading: Member, slab: HoldoutSlab, cache: PredictionCache, window: Window
    ) -> Float[np.ndarray, "studies targets"]:
        """One member's probabilities — from the cache, then the run's own record, and only then the card.

        THREE SOURCES IN COST ORDER. The cache answers in milliseconds. `TrainerRecord` answers from the
        array training already wrote, in the same order of magnitude and with no card at all, whenever
        the reading asked for is the one the trainer used. The encoder runs only when neither can, which
        for this pool is 38 members of 200 rather than all of them.

        The encoder, its download, the holdout's pixels and the GPU are ALL reached only on a miss. That
        is the difference between an analysis that can run while the card belongs to something else and
        one that cannot, and every paired interval, oracle and subset search here is arithmetic over what
        this returns.

        `load` takes its backbone as an argument so it can be tested without a network; this method is
        the caller that does the fetching, and it does it inside the miss branch for the same reason.
        """
        return cache.remembered(
            reading.file,
            window,
            reading.precision,
            lambda studies: self._uncached(reading, slab, window, studies),
        )

    def _uncached(
        self, reading: Member, slab: HoldoutSlab, window: Window, studies: list[str]
    ) -> Float[np.ndarray, "studies targets"]:
        """The two sources the cache does not hold: the run's own record, else the encoder."""
        recorded = self.record.instead_of(reading, slab, window, studies)
        if recorded is not None:
            logger.info("%s: read from the run's own holdout record, not the card", reading.file.name)
            return recorded
        # A MISS SAYS SO. The hit path returns in milliseconds and the miss path is a full inference
        # pass over the holdout — minutes on the card, hours on a CPU. Silent, the two are one
        # unexplained wait, and on 2026-08-19 two CPU subset searches were killed as hung while they
        # were computing: the pool's three best members have no cache entry, because the search that
        # scored them was `record_mix`, which reads the arrays training already wrote and never fills
        # this store. Naming the member and the device turns that wait into a decision.
        logger.info("%s: cache MISS, computing on %s", reading.file.name, self.device)
        encoder = ENCODERS[reading.variant]
        model = self.load(reading.file, encoder.load(), encoder.dim, reading.slot_count)
        return self.predict(model, slab, window, reading.precision)
