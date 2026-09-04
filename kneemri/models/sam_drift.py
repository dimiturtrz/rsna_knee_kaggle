"""Where a SAM fine-tune actually moved — per-region parameter drift between two releases.

THE QUESTION THIS ANSWERS IS A READ-OUT POINT, and it is worth a measurement because the literature
has no answer and a card would spend forty minutes learning one badly. MedSAM's vision encoder runs
twelve ViT-B blocks at 768 wide and then a convolutional NECK that projects to 256 channels for the
mask decoder. `sam_trunk` currently hands the head the neck output, and three published results argue
against that: SAM's neck output linear-probes at 11.1% ImageNet top-1 against DINOv2's 77.4%
(SAMantics, arXiv:2411.15288), a medical report-generation model drops the same neck explicitly to
read patch embeddings (FODA-PG, arXiv:2409.03947), and the neck is a 3x compression designed for a
decoder this project does not build.

NONE OF THAT SETTLES IT, because of one thing MedSAM's own paper says: the fine-tune "updated all
trainable parameters in the image encoder", the neck included, and it publishes no layer-wise
breakdown. So if the medical adaptation CONCENTRATED in the neck, reading raw tokens would discard
precisely what makes the checkpoint medical and leave a plain SAM wearing MedSAM's name — and the
resulting arm would be read as "a dense segmentation objective does not transfer" when what was
actually measured is a trunk the fine-tune barely touched. That is the RAD-DINO failure again: a real
number supporting a false answer to the question the arm was queued to settle.

The checkpoints answer it directly. `facebook/sam-vit-base` is the initialisation MedSAM started from,
so the per-region relative movement between the two IS the layer-wise breakdown the paper omits. It
costs two CPU downloads and no card.

READ IT AS A RATIO BETWEEN REGIONS, never as an absolute. A relative L2 of 0.1 means nothing on its
own; the neck moving several times more than the blocks means the adaptation lives in the part a
classifier would throw away, and the blocks moving comparably or more means raw tokens keep it.
"""

import logging
from collections import defaultdict

import torch
from torch import nn

from core import cli

logger = logging.getLogger(__name__)

SAM_VIT_B = "facebook/sam-vit-base"
"""The initialisation MedSAM fine-tuned from, which is what makes the difference a MEASUREMENT.

Named here rather than passed as a default string in the parser so that the claim "this is the
baseline the fine-tune started at" sits next to the reason it must be true: a drift computed against
any other release measures two independent trainings and means nothing.
"""


class SamDrift:
    """How far each region of one released encoder sits from the release it was fine-tuned from."""

    @staticmethod
    def moved(before: nn.Module, after: nn.Module) -> dict[str, float]:
        """Per-parameter relative movement, `||after - before|| / ||before||`, keyed by parameter path.

        RELATIVE rather than absolute because the regions being compared are not the same size or the
        same scale: a 3x3 convolution's weights and a 768-wide attention projection's have different
        norms before anything is trained, so a raw distance would rank regions by how big they are.

        A parameter the two releases do not share is SKIPPED rather than raising. The two are the same
        architecture, so a missing name means a genuine structural difference between the releases, and
        that is a fact for the caller to see in the census rather than a crash — a drift report that
        refuses to run tells the reader less than one that says which regions it could compare.
        """
        original = dict(before.named_parameters())
        moved: dict[str, float] = {}
        for name, fitted in after.named_parameters():
            start = original.get(name)
            if start is None or start.shape != fitted.shape:
                continue
            base = torch.linalg.vector_norm(start.detach().float())
            if base == 0:
                continue
            moved[name] = (torch.linalg.vector_norm(fitted.detach().float() - start.detach().float()) / base).item()
        return moved

    @staticmethod
    def region(name: str) -> str:
        """The region a parameter path belongs to — its top component, plus the block index inside one.

        `layers` is split per block and nothing else is, because the whole point of the comparison is
        whether adaptation sits at the END of the stack (the neck) or through it, and a single `layers`
        row averages a depth gradient away.
        """
        parts = name.split(".")
        return f"{parts[0]}.{parts[1]}" if parts[0] == "layers" and len(parts) > 1 else parts[0]

    @classmethod
    def grouped(cls, moved: dict[str, float]) -> dict[str, float]:
        """The same movement per REGION, each region's parameters pooled as a root-mean-square.

        Pooled rather than averaged so a region is not diluted by its many small biases: the mean over
        parameter paths weights a one-element LayerNorm bias exactly as heavily as an attention matrix,
        which is how a region that barely moved can be made to look like one that did.
        """
        pooled: dict[str, list[float]] = defaultdict(list)
        for name, value in moved.items():
            pooled[cls.region(name)].append(value * value)
        return {region: float(torch.tensor(values).mean().sqrt()) for region, values in pooled.items()}

    @classmethod
    def lines(cls, moved: dict[str, float]) -> list[str]:
        """The per-region report, ordered by movement so the adaptation's home is the first row."""
        grouped = cls.grouped(moved)
        ranked = sorted(grouped.items(), key=lambda row: row[1], reverse=True)
        census = f"{len(moved)} parameters compared across {len(grouped)} regions"
        return [census, *(f"{region}: {value:.4f}" for region, value in ranked)]


def main() -> None:
    """Report where a fine-tuned SAM release moved away from the release it started from."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--baseline", default=SAM_VIT_B, help="the release the fine-tune started from")
    command.parser.add_argument("--against", required=True, help="the fine-tuned release to measure")
    arguments = command.parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=cli.FORMAT)

    # Imported here, not at module scope, for `sam_trunk`'s reason: a Kaggle kernel reaches transformers
    # only once the kit is on `sys.path`, and the models package is imported before that happens.
    from transformers import SamModel  # noqa: PLC0415 - see above

    before = SamModel.from_pretrained(arguments.baseline).vision_encoder
    after = SamModel.from_pretrained(arguments.against).vision_encoder
    command.report(SamDrift.lines(SamDrift.moved(before, after)))


if __name__ == "__main__":
    main()
