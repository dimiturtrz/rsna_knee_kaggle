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

from kneemri.holdout_predictions import HoldoutPredictions, Window
from kneemri.model import TARGETS, StudyModel
from kneemri.scores import LabelScores

CENTRE_FRACTION = 0.5
"""Side of the kept centre square as a fraction of the image — a quarter of the area, the joint itself."""

RING_WIDTHS = (0.05, 0.10, 0.15, 0.25)
"""Ring widths to try, as a fraction of the image side measured inward from the edge.

`region_importance` measured border-only at 0.8006 against a whole-image 0.8448, which looks like the
model reading the periphery — but its "border" is everything outside the central half, which is 75% of
the AREA and still full of joint. A 0.05 ring on a 130 mm crop is the outer ~6 mm: skin, coil, air and
whatever the scanner burned in. If THAT still scores, the periphery reading is real; if it collapses,
the earlier number only ever said the signal is spread out.
"""

CHANCE = 0.5
"""ROC AUC of a model that knows nothing — the value a ring holding no anatomy should approach."""


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
    window: Window
    """HOW the member is read — starts, WIDTH and size together, from `Member.window`.

    It was `starts` and `size` as two loose fields, and the width was then supplied by `score` as the
    constant GROUP. So a whole-stack member, whose window is its full depth, was occluded three slices
    at a time — the knee-y8y defect surviving in the analysis path after inference and `member_curve`
    were both fixed for it. Carrying the whole `Window` makes the width impossible to supply from
    somewhere else."""


@dataclass(frozen=True)
class Diagnostics:
    """Everything the occlusions measured, so the report reads one object rather than five arguments.

    Passing them separately put `report` at six parameters and tripped the argument-count gate — the
    same signal that produced `Holdout`, and right for the same reason: these five are one finding
    about one model, and they are never assembled apart.
    """

    slots: dict[str, float]
    regions: dict[str, float]
    rings: dict[str, float]
    depth: dict[str, float]
    depth_by_label: dict[str, dict[str, float]]


class Attribution:
    """Occlusion diagnostics for a trained study model, reported as deltas on the held-out score."""

    def __init__(self, predictions: HoldoutPredictions, scores: LabelScores | None = None) -> None:
        self.predictions = predictions
        self.scores = scores or LabelScores()

    def _labels(
        self,
        model: StudyModel,
        holdout: Holdout,
    ) -> dict[str, float]:
        """Per-label held-out AUC under whatever occlusion the caller has already applied."""
        scored = self.predictions.predict(model, holdout.images, holdout.mask, holdout.window)
        return self.scores.per_label(scored, holdout.truth)

    def _mean(
        self,
        model: StudyModel,
        holdout: Holdout,
    ) -> float:
        """The held-out mean AUC under whatever occlusion the caller has already applied."""
        return float(np.nanmean(list(self._labels(model, holdout).values())))

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

    def ring_importance(
        self,
        model: StudyModel,
        holdout: Holdout,
        widths: tuple[float, ...] = RING_WIDTHS,
    ) -> dict[str, float]:
        """Score with ONLY a narrow ring at the edge of the field kept, for several ring widths.

        This is the test `region_importance` could not do. Its "border" was everything outside the
        central half — 75% of the area, most of it still knee — so a high border-only score was
        consistent both with a model reading the periphery and with a signal that is simply spread out.
        A narrow ring contains no joint at all, so it separates them: a ring that scores well above
        chance means the model found something outside the anatomy, and that is a shortcut which will
        not survive a different scanner population.

        Widths shrink toward the edge, so the scores should FALL toward chance as the ring narrows.
        A flat profile is the alarming outcome.
        """
        pixels = holdout.images.shape[-1]
        scores = {}
        for width in widths:
            kept = int(pixels * width)
            ring = holdout.images.copy()
            ring[..., kept : pixels - kept, kept : pixels - kept] = 0
            scores[f"{width:.2f}"] = self._mean(model, replace(holdout, images=ring))
        return scores

    def averaged(self, by_label: dict[str, dict[str, float]]) -> dict[str, float]:
        """Mean over labels at each depth — the profile, derived rather than measured a second time.

        This used to be its own method that re-ran every window through the model, which doubled the
        cost of the depth diagnostic to compute a number already contained in `depth_by_label`. Ten
        GPU passes for an arithmetic mean.
        """
        return {start: float(np.nanmean(list(labels.values()))) for start, labels in by_label.items()}

    def depth_by_label(
        self,
        model: StudyModel,
        holdout: Holdout,
    ) -> dict[str, dict[str, float]]:
        """Each window of the slice stack scored ALONE, per diagnosis.

        Depth is the only localisation a study-level label can support: the model is never told where
        a finding is, but it can be asked which depth it needed in order to find one. Kept per label
        because twelve findings have no reason to share a depth.


        The averaged profile says the best single window is worth 0.036 more than the worst, which is
        larger than every method change measured in this project. What it cannot say is whether ONE
        window could serve every label: a meniscal tear and a joint effusion live in different places,
        and a mean over twelve labels would hide two opposite preferences as a flat middle.

        This is what gates acting on the profile at all. If the labels agree, a single re-weighting of
        the sampled windows helps all twelve; if they disagree, the only correct response is to read
        more of the stack rather than to pick a better window.
        """
        return {
            str(start): self._labels(model, replace(holdout, window=replace(holdout.window, starts=[start])))
            for start in holdout.window.starts
        }

    def report(self, diagnostics: Diagnostics) -> list[str]:
        """The diagnostics as lines, with the verdict each one supports stated rather than implied."""
        lines = ["slot importance (score lost when the slot is withheld):"]
        for slot, drop in sorted(diagnostics.slots.items(), key=lambda item: -item[1]):
            lines.append(f"  slot {slot}: {drop:+.4f}")
        spread = max(diagnostics.slots.values()) - min(diagnostics.slots.values())
        lines.append(
            f"  spread {spread:.4f} — a large spread means one acquisition carries everything, "
            "which is the degenerate-attention failure of attention-MIL"
        )

        lines.append("")
        regions = diagnostics.regions
        lines.append(
            f"region: whole {regions['whole']:.4f} | centre-only {regions['centre_only']:.4f} "
            f"| border-only {regions['border_only']:.4f}"
        )
        lines.append(
            "  a high border-only is ambiguous on its own: the border is 75% of the area and still "
            "holds joint. The rings below are what separates 'spread out' from 'reading the periphery'"
        )

        lines.append("")
        lines.append("outer ring only (fraction of the side, inward from the edge):")
        for width, score in sorted(diagnostics.rings.items()):
            lines.append(f"  ring {width}: {score:.4f}  ({score - CHANCE:+.4f} over chance)")
        narrowest = min(diagnostics.rings, key=float)
        lines.append(
            f"  the narrowest ring holds no joint, so {diagnostics.rings[narrowest]:.4f} is the number "
            "that matters. Near chance clears the model; well above it is a shortcut"
        )

        lines.append("")
        depth = diagnostics.depth
        best = max(depth, key=lambda start: depth[start])
        lines.append(f"depth: best single window starts at slice {best} ({depth[best]:.4f})")
        lines.append(f"  range {min(depth.values()):.4f} to {max(depth.values()):.4f} over {len(depth)} windows")
        lines.extend(self._depth_agreement(diagnostics.depth_by_label))
        return lines

    def _depth_agreement(self, by_label: dict[str, dict[str, float]]) -> list[str]:
        """Whether the twelve labels want the same window, which is what gates acting on the profile."""
        preferred = {
            label: max(by_label, key=lambda start: by_label[start][label]) for label in next(iter(by_label.values()))
        }
        chosen = sorted(set(preferred.values()), key=float)
        lines = [f"  per-label best window: {len(chosen)} distinct depths over {len(preferred)} labels"]
        for start in chosen:
            wanting = [label for label, best in preferred.items() if best == start]
            lines.append(f"    slice {start}: {', '.join(wanting)}")
        lines.append(
            "  one shared depth means a re-weighting of the sampled windows helps every label; "
            "several means the only correct answer is to read more of the stack"
        )
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
    from kneemri.members import Member  # noqa: PLC0415 - as above
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
    # THE MEMBER'S OWN READING, never a constant. This CLI loaded the default 336 px cache, windowed by
    # GROUP whatever the member was, passed IMAGE_SIZE as the read size and assumed the canonical six
    # slots — so a whole-stack member was occluded three slices at a time, a 448-cache member could not
    # be read at all, and an extended-scheme member would have been rebuilt with the wrong slot count.
    # member_curve carried the identical defect and was fixed; this, the other analysis CLI, was not.
    # It is the tool that produced the border-only and depth-spread findings, so a wrong reading here
    # does not fail — it publishes. The crop and the band were the next two fields to be dropped on the
    # way here, which is why the member names its own cache now instead of a caller spelling it out:
    # the border ring this CLI blanks is measured in fractions of a crop it was not reading.
    member = Member.from_checkpoint(arguments.checkpoint)
    corpus, identifiers = member.cache(root.processed()).load()
    folds = ReportFolds(arguments.folds, arguments.split_seed)
    _, holdout = folds.holdout(folds.assign(str(root.raw() / "train.csv"), identifiers), arguments.fold)
    holdout = holdout[: arguments.studies]

    truth = Targets.load(arguments.labels, root.raw() / "train.csv", identifiers).scores[holdout]
    saved = torch.load(arguments.checkpoint, map_location="cpu", weights_only=True)
    encoder = ENCODERS[str(saved.get("config", {}).get("variant", SMALL))]

    predictions = HoldoutPredictions(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model = predictions.load(
        arguments.checkpoint, AutoModel.from_pretrained(encoder.repository), encoder.dim, member.slot_count
    )
    images, mask = corpus.images[holdout], corpus.mask[holdout]
    # `overlap=True` gives a windowed member every start and a whole-stack member the ONE centred window
    # of its full depth, which is the whole point of routing this through Member.window.
    window = member.window(corpus.slices, overlap=True)

    attribution = Attribution(predictions)
    holdout_data = Holdout(images, mask, truth, window)
    logger.info("reading: %s", f"{len(window.starts)} window(s) of {window.width} at {window.size} px")
    logger.info("labels: %s", list(TARGETS))
    by_label = attribution.depth_by_label(model, holdout_data)
    diagnostics = Diagnostics(
        slots=attribution.slot_importance(model, holdout_data),
        regions=attribution.region_importance(model, holdout_data),
        rings=attribution.ring_importance(model, holdout_data),
        depth=attribution.averaged(by_label),
        depth_by_label=by_label,
    )
    for line in attribution.report(diagnostics):
        logger.info("%s", line)


if __name__ == "__main__":
    main()
