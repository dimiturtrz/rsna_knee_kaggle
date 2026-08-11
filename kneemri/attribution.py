"""What is the model actually using — measured by taking things away, not by drawing heatmaps.

This model is attention-based multiple-instance learning: six acquisition slots are the instances, the
study-level label is the bag, and `SlotHead`'s per-diagnosis attention is the pooling. Nothing in the
supervision ever says WHERE a finding is, so a saliency map has no reason to look right and cannot be
checked against anything. An occlusion is different: it produces a NUMBER on the held-out score, and
the number answers a question that matters — *is the model reading anatomy, or reading an artefact?*

Three diagnostics, each a measured delta rather than a picture:

- **Slot importance.** Drop one acquisition and re-score. The masked softmax renormalises over what
  remains, so this is exactly the "study arrived without that sequence" case the mask exists for. A
  finding that only moves when its own sequence is removed is being read where a radiologist reads it;
  a model that collapses onto one slot for everything is the degenerate-attention failure this
  architecture is known for (`knee-e6s`: attention/anatomy correlation +0.127).
- **Border vs centre.** The crop is 130 mm around the joint, so the outer ring is mostly soft tissue,
  coil and background — plus any burned-in laterality marker or vendor text. If blanking the border
  costs nothing and blanking the centre costs everything, the model is reading the joint. The reverse
  would be a shortcut worth finding.
- **Depth profile.** Score each window of the slice stack alone. Where along the stack a label's signal
  lives is the one localisation this supervision CAN support, and it bears directly on how thick a
  three-slice group should be.

Saliency is deliberately not here. The sibling cardiac project keeps captum behind an optional extra
for the same reason: the always-on diagnostic has to be the one that yields a measurement.
"""

from dataclasses import dataclass, replace

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy.typing import NDArray

from kneemri.holdout_predictions import HoldoutPredictions
from kneemri.model import TARGETS, StudyModel
from kneemri.scores import LabelScores

CENTRE_FRACTION = 0.5
"""Side of the kept centre square as a fraction of the image — a quarter of the area, the joint itself."""


@dataclass(frozen=True)
class Holdout:
    """The studies a diagnostic scores on, and how to read them.

    One object because these five always travel together and an occlusion varies exactly one of them:
    `replace(holdout, images=...)` is what taking something away looks like. Passing them separately
    put every diagnostic at six arguments and tripped the argument-count gate, which was right that an
    object was missing.
    """

    images: UInt8[NDArray[np.uint8], "b n s h w"]
    mask: Float[NDArray[np.float32], "b n"]
    truth: Float[NDArray[np.float32], "b o"]
    starts: list[int]
    size: int


class Attribution:
    """Occlusion diagnostics for a trained study model, reported as deltas on the held-out score."""

    def __init__(self, predictions: HoldoutPredictions, scores: LabelScores | None = None) -> None:
        self.predictions = predictions
        self.scores = scores or LabelScores()

    def _mean(
        self,
        model: StudyModel,
        holdout: Holdout,
    ) -> float:
        """The held-out mean AUC under whatever occlusion the caller has already applied."""
        scored = self.predictions.predict(model, holdout.images, holdout.mask, holdout.starts, holdout.size)
        return float(np.nanmean(list(self.scores.per_label(scored, holdout.truth).values())))

    def slot_importance(
        self,
        model: StudyModel,
        holdout: Holdout,
    ) -> dict[str, float]:
        """How much the score falls when each acquisition slot is withheld, one at a time.

        Withheld through the PRESENCE MASK rather than by blanking pixels, so the head renormalises its
        attention over the remaining slots — the same path a study that never contained that sequence
        takes. Blanking the pixels instead would feed the encoder a black image and measure the model's
        response to an input it never sees in the wild.
        """
        base = self._mean(model, holdout)
        drops = {}
        for slot in range(holdout.mask.shape[1]):
            withheld = holdout.mask.copy()
            withheld[:, slot] = 0.0
            drops[str(slot)] = base - self._mean(model, replace(holdout, mask=withheld))
        return drops

    def region_importance(
        self,
        model: StudyModel,
        holdout: Holdout,
    ) -> dict[str, float]:
        """The score with only the centre kept, and with only the border kept.

        The interesting outcome is the asymmetry. Centre-only holding up while border-only collapses
        means the model reads the joint; the reverse means it has found something in the periphery —
        a burned-in marker, a coil, a vendor annotation — that correlates with the label.
        """
        images = holdout.images
        pixels = images.shape[-1]
        keep = int(pixels * CENTRE_FRACTION)
        low = (pixels - keep) // 2
        high = low + keep

        centre_only = np.zeros_like(images)
        centre_only[..., low:high, low:high] = images[..., low:high, low:high]

        border_only = images.copy()
        border_only[..., low:high, low:high] = 0

        return {
            "whole": self._mean(model, holdout),
            "centre_only": self._mean(model, replace(holdout, images=centre_only)),
            "border_only": self._mean(model, replace(holdout, images=border_only)),
        }

    def depth_profile(
        self,
        model: StudyModel,
        holdout: Holdout,
    ) -> dict[str, float]:
        """Each window of the slice stack scored ALONE, so the signal can be located along the depth.

        This is the only localisation a study-level label can support: the model is never told where a
        finding is, but it can be asked which depth it needed in order to find one.
        """
        return {str(start): self._mean(model, replace(holdout, starts=[start])) for start in holdout.starts}

    def report(self, slots: dict[str, float], regions: dict[str, float], depth: dict[str, float]) -> list[str]:
        """The three diagnostics as lines, with the verdict each one supports stated rather than implied."""
        lines = ["slot importance (score lost when the slot is withheld):"]
        for slot, drop in sorted(slots.items(), key=lambda item: -item[1]):
            lines.append(f"  slot {slot}: {drop:+.4f}")
        spread = max(slots.values()) - min(slots.values())
        lines.append(
            f"  spread {spread:.4f} — a large spread means one acquisition carries everything, "
            "which is the degenerate-attention failure of attention-MIL"
        )

        lines.append("")
        lines.append(
            f"region: whole {regions['whole']:.4f} | centre-only {regions['centre_only']:.4f} "
            f"| border-only {regions['border_only']:.4f}"
        )
        lines.append(
            "  border-only should be near chance. Anything else is the model reading the periphery — "
            "a marker, a coil, an annotation — rather than the joint"
        )

        lines.append("")
        best = max(depth, key=lambda start: depth[start])
        lines.append(f"depth: best single window starts at slice {best} ({depth[best]:.4f})")
        lines.append(f"  range {min(depth.values()):.4f} to {max(depth.values()):.4f} over {len(depth)} windows")
        return lines

    def targets(self) -> list[str]:
        """The label order these diagnostics are averaged over."""
        return list(TARGETS)


def main() -> None:
    """Run the occlusion diagnostics for one member on its fold's holdout."""
    import argparse  # noqa: PLC0415 - CLI-only, and this module is imported as a library by analyses
    import logging  # noqa: PLC0415 - as above
    from pathlib import Path  # noqa: PLC0415 - as above

    from transformers import AutoModel  # noqa: PLC0415 - a 90 MB import only this entrypoint needs

    from core.paths import DataRoot  # noqa: PLC0415 - as above
    from kneemri.backbone import ENCODERS, SMALL  # noqa: PLC0415 - as above
    from kneemri.folds import ReportFolds  # noqa: PLC0415 - as above
    from kneemri.holdout_predictions import GROUP  # noqa: PLC0415 - as above
    from kneemri.slot_reader import IMAGE_SIZE  # noqa: PLC0415 - as above
    from kneemri.slots import SLOTS  # noqa: PLC0415 - as above
    from kneemri.study_cache import StudyCache  # noqa: PLC0415 - as above
    from kneemri.targets import Targets  # noqa: PLC0415 - as above

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--slices", type=int, default=12)
    parser.add_argument("--studies", type=int, default=400, help="holdout studies sampled, for wall-clock")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    root = DataRoot.from_config(arguments.paths)
    corpus, identifiers = StudyCache(root.processed(), slices=arguments.slices).load()
    folds = ReportFolds(arguments.folds, arguments.split_seed)
    _, holdout = folds.holdout(folds.assign(str(root.raw() / "train.csv"), identifiers), arguments.fold)
    holdout = holdout[: arguments.studies]

    truth = Targets.load(arguments.labels, root.raw() / "train.csv", identifiers).scores[holdout]
    saved = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    encoder = ENCODERS[str(saved.get("config", {}).get("variant", SMALL))]

    predictions = HoldoutPredictions(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model = predictions.load(
        arguments.checkpoint, AutoModel.from_pretrained(encoder.repository), encoder.dim, len(SLOTS)
    )
    images, mask = corpus.images[holdout], corpus.mask[holdout]
    starts = list(range(arguments.slices - GROUP + 1))

    attribution = Attribution(predictions)
    holdout_data = Holdout(images, mask, truth, starts, IMAGE_SIZE)
    logger.info("slots: %s", [slot.name for slot in SLOTS])
    for line in attribution.report(
        attribution.slot_importance(model, holdout_data),
        attribution.region_importance(model, holdout_data),
        attribution.depth_profile(model, holdout_data),
    ):
        logger.info("%s", line)


if __name__ == "__main__":
    main()
