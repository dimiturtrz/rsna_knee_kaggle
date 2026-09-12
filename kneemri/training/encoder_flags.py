"""What the ENCODER half of a training run accepts — the trunk flags, apart from the rest of the vocabulary.

Split out of `FlagVocabulary` on the 750-line god-file ceiling, and this is the group that comes out
whole: every flag here is a statement about the trunk, and none of them is read by anything that does
not already hold an `Adaptation`. The split is the same shape as the one that took this vocabulary out
of `TrainingOptions` — the declaration grows when a switch is added, the meaning does not.
"""

import argparse
from pathlib import Path

from kneemri.models.backbone import ENCODERS, SMALL
from kneemri.training.adaptation import Adaptation
from kneemri.training.init_scope import INIT_SCOPES
from kneemri.training.pretrained_encoder import ALL_BLOCKS


class EncoderFlags:
    """The trunk's flags, addable to any parser that wants them."""

    @staticmethod
    def add(parser: argparse.ArgumentParser) -> None:
        """What the encoder IS, what it reads, and what of it adapts — the trunk half of the recipe.

        Split out of `parser` rather than left inline because that method crossed the statement ceiling,
        and this is the group that comes out whole: every flag here is a statement about the trunk, and
        none of them is read by anything that does not already hold an `Adaptation`.
        """
        parser.add_argument(
            "--encoder-layer",
            type=int,
            default=-1,
            help="which encoder block the head reads. -1 is the last, which is what every member so far "
            "was fitted on. The final block of a self-supervised ViT is the most specialised toward its "
            "PRETRAINING objective, and our findings are 1-3 mm - both DINOv2 and DINOv3 ship "
            "get_intermediate_layers for dense work. Costs no extra forward pass.",
        )
        parser.add_argument(
            "--through-plane",
            action="store_true",
            help="give each slice its two depth neighbours as the encoder's other two input channels "
            "instead of three copies of itself. Same slice axis, same pooling, same token / parameter / "
            "FLOP count as the control — the only change is what the three channels hold, so it is the "
            "one arm that puts through-plane information INTO a slice's feature vector rather than "
            "mixing depth after the encoder has already reduced each slice to one.",
        )
        parser.add_argument(
            "--anatomical-prior",
            action="store_true",
            help="start each diagnosis's slot attention from how the finding is actually read",
        )
        parser.add_argument(
            "--variant",
            choices=tuple(ENCODERS),
            default=SMALL,
            help="which pretrained encoder. DINOv2-small is the committed default: DINOv3 is refuted "
            "as an improvement at -0.038 and retained only as a diversity member (see "
            "interpretations/kneemri/2026-08-13_dinov2-stays-and-why-dinov3-loses-the-focal-findings.md).",
        )
        parser.add_argument(
            "--unfreeze-last",
            type=int,
            default=Adaptation().unfreeze_last,
            help="how many of the encoder's trailing blocks train. The restraint is a HYPOTHESIS, not "
            "a measurement: with 58 image-read labels the early generic blocks may have too little "
            "supervision to improve and enough to damage. Capacity is the only axis that has ever "
            "cleared our noise floor, so the hypothesis is worth testing rather than inheriting.",
        )
        parser.add_argument(
            "--unfreeze-offset",
            type=int,
            default=Adaptation().unfreeze_offset,
            help="how many blocks at the TOP of the trunk stay FROZEN, which turns --unfreeze-last from "
            "a suffix into a window. Zero is every run before 2026-09-05. Lee et al. (ICLR 2023) choose "
            "the block by the KIND of shift, and the input-level case - natural images to greyscale MRI - "
            "wants EARLY blocks, which a suffix cannot express.",
        )
        parser.add_argument(
            "--init",
            type=Path,
            default=None,
            help="start from a member already on disk instead of from the encoder's published weights "
            "plus a fresh head. The one flag LP-FT needs: stage one trains a head over a FROZEN trunk "
            "(--unfreeze-last 0), stage two points --init at that checkpoint and unfreezes at a low "
            "backbone rate. Kumar et al. (arXiv:2202.10054) measure fine-tuning from a fresh head "
            "distorting pretrained features ~20x more than fine-tuning from a fitted one, which is the "
            "shape radnet_family produced here: 0.8197 frozen, 0.6716 fine-tuned, same weights. The "
            "checkpoint's DESIGN is not read back - this loads weights into the model these flags "
            "describe, so a mismatched architecture raises here rather than training something else.",
        )
        parser.add_argument(
            "--init-scope",
            choices=sorted(INIT_SCOPES),
            default="all",
            help="how much of --init is taken. 'all' is LP-FT's case, where stage one's head IS the head "
            "stage two wants. 'trunk' is the CROSS-CORPUS case: a checkpoint fitted on MRNet has three "
            "series slots against this corpus's six, so head.slot_emb is (3, 256) against (6, 256) and a "
            "whole-model load raises - after stage one has already spent its card. The head is not the "
            "part that transfers anyway; feature reuse in the early and middle blocks is (Yosinski 2014, "
            "Raghu 2019, Matsoukas 2022). Everything the scope does take is still shape-checked.",
        )
        parser.add_argument("--head-lr", type=float, default=Adaptation().head_lr)
        parser.add_argument(
            "--backbone-lr",
            type=float,
            default=Adaptation().backbone_lr,
            help="how fast the ENCODER adapts. The 125:1 ratio to the head was inherited, never "
            "measured, and at 8e-6 the encoder barely moves - which is why our depth bracket came out "
            "neutral: unfreezing more blocks cannot help at a rate where no block moves.",
        )
        parser.add_argument(
            "--layer-decay",
            type=float,
            default=Adaptation().layer_decay,
            help="scale the encoder rate DOWN by this factor per block below the top one (0.65-0.75 is "
            "the usual range; 1.0, the default, is the flat rate every ledger number was measured at). "
            "Without it depth and rate can only move together: --unfreeze-last 12 at the rate that makes "
            "the top blocks move applies that same rate to the encoder's most general layers, so a "
            "deeper arm is not a deeper version of the shallow one, it is a different perturbation.",
        )
        parser.add_argument(
            "--whole-stack",
            action="store_true",
            help="read EVERY cached slice each step instead of one sampled 3-slice window. Needs a "
            "per-slice pooling or a tubelet trunk, and for a tubelet trunk it is the only path that "
            "reaches a window wider than 3 at all. A step currently sees 18 of a study's ~180 slices, "
            "while averaging "
            "ten windows at INFERENCE is worth +0.0156 - the largest single effect measured here - so "
            "training was discarding information it already had. Costs about 4x a step.",
        )
        parser.add_argument(
            "--report-weight",
            type=float,
            default=0.0,
            help="also predict each study's REPORT embedding, at this weight (0 = off). Our supervision "
            "is twelve numbers where the report has a median of 129 words in twelve languages; this "
            "term supervises the representation with the rest of it. Training-only - test.csv has no "
            "Report column, so inference is unchanged.",
        )
        parser.add_argument(
            "--encoder-weights",
            type=Path,
            default=None,
            help="start the encoder from a domain-adapted checkpoint written by "
            "kneemri.training.pretrain instead of the stock natural-image weights. Everything else in "
            "the recipe is unchanged, so a run with this flag and its matched control differ in one "
            "variable: what the encoder knew before supervision began.",
        )
        parser.add_argument(
            "--encoder-adapt-depth",
            type=int,
            default=ALL_BLOCKS,
            help="take only the patch embedding and the first N blocks from --encoder-weights, leaving "
            "the rest stock. 0 takes the whole adaptation. The argument for on-domain pretraining only "
            "ever concerned the blocks --unfreeze-last leaves frozen, which supervision cannot write into "
            "and "
            "so cannot repair; the top blocks the head reads are the ones a reconstruction objective "
            "rewrites hardest, and two full loads measured on 2026-08-20 lost 0.17 with block 11 moved "
            "55%% and 21%% of its own norm.",
        )
