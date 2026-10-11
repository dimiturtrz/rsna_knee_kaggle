"""How much of the encoder moves, and how fast — the entangled triple, apart from the loop that uses it.

It lived inside `trainer.py` until 2026-09-12, when that file crossed the god-file ceiling and the split
had to be made somewhere. This is the seam that costs nothing: `Adaptation` is a frozen record of four
numbers with no behaviour, it is CONSTRUCTED by five modules that have no other business with `Trainer`
(the flag vocabularies read its defaults to print them), and the trainer holds one rather than being one.
"""

from dataclasses import dataclass
from pathlib import Path

from kneemri.models.backbone import ENCODERS
from kneemri.models.encoder_surgery import NO_LAYER_DECAY, NO_OFFSET
from kneemri.models.study_design import StudyDesign
from kneemri.models.variants import SMALL
from kneemri.training.config_leaf import ConfigLeaf
from kneemri.training.init_scope import INIT_SCOPES
from kneemri.training.pretrained_encoder import ALL_BLOCKS

WEIGHT_DECAY = 0.02


@dataclass(frozen=True)
class Adaptation:
    """How much of the encoder moves, and how fast — one object because they cannot be read apart.

    The head is random at initialisation and has everything to learn; the encoder starts from a good
    solution and needs only to be moved off it. That argument sets the 125:1 default ratio, and it was
    inherited from the reference rather than measured.

    Depth and rate are ENTANGLED, which is why they stopped being three constants. Our bracket found
    unfreezing every block worth +0.0008, and a null result at a backbone rate of 8e-6 cannot
    distinguish "more trainable blocks do not help" from "no block was going to move anyway". Asking
    how many blocks should train is only a real question at a rate where a block moves.
    """

    head_lr: float = 1e-3
    backbone_lr: float = 8e-6
    weight_decay: float = WEIGHT_DECAY
    """How hard AdamW pulls every trained weight back toward zero — a member of the same entangled set.

    It was a module constant until 2026-08-31 and no arm had ever varied it. 0.02 came in with the
    reference recipe at a backbone rate of 8e-6; the rate sweep then moved the trunk five-fold to 4e-5
    without re-deriving it, so the decay every current number was measured under is the one chosen for a
    trunk that barely moved. Defaults to that constant, which is what keeps every existing arm identical.
    """

    unfreeze_last: int = 6
    """Six of twelve blocks — inherited from the reference, and only meaningful beside the rate above."""

    unfreeze_offset: int = NO_OFFSET
    """How many blocks at the TOP of the trunk stay frozen, which turns the suffix into a WINDOW.

    Zero for every run before 2026-09-05, and that is the suffix every ledger number was measured
    under. It exists because the block a run should open is decided by the KIND of distribution shift
    (Lee et al., ICLR 2023) and the input-level case — which natural-image pretraining to greyscale MRI
    is — wants EARLY blocks, the one thing `unfreeze_last` alone cannot express. See `BlockWindow`.
    """

    unfreeze_stem: bool = False
    """Whether the encoder's pre-block parameters — patch embedding, cls token, position table — train.

    False for every run this project has made, `--unfreeze-last 12` included: `last_blocks` builds its
    trainable set out of the window's BLOCKS, so the stem freezes at every depth, and `encoder_drift`
    reads it at 0.000000 across all fifteen supervised members swept (knee-p6fr8). It is a separate
    switch from the depth because it is a different KIND of layer: the blocks read tokens, and the stem
    is the only place the trunk reads pixels, which is where natural-image RGB against 336 px greyscale
    MRI is a difference in the data rather than in a representation of it. The position table is
    resampled 37x37 to 24x24 on every forward by `Dinov2Embeddings`, so at this read size the stem is
    both frozen and interpolated. See `BlockWindow.stem`.
    """

    layer_decay: float = NO_LAYER_DECAY
    """How much slower each block below the top one moves — the third member of the entangled triple.

    Defaults to no decay, which is what every ledger number was measured under. It belongs here rather
    than beside the schedule because it is the term that makes the OTHER two separable: at one flat rate
    a depth result and a rate result are the same measurement read twice.
    """


@dataclass(frozen=True)
class EncoderConfig:
    """Which trunk a run builds, what it starts from, and how much of it moves."""

    encoder_layer: int = ConfigLeaf.of(
        default=-1,
        says=(
            "which encoder block the head reads. -1 is the last, which is what every member so far was fitted"
            " on. The final block of a self-supervised ViT is the most specialised toward its PRETRAINING "
            "objective, and our findings are 1-3 mm - both DINOv2 and DINOv3 ship get_intermediate_layers for"
            " dense work. Costs no extra forward pass."
        ),
    )
    fuse_blocks: int = ConfigLeaf.of(
        default=0,
        says=(
            "how many blocks before the last add their normalised tokens to the last block's, each through"
            " a zero-initialised per-channel gate, so the run starts exactly at the control. 0 is every "
            "member so far. Costs no extra forward pass."
        ),
    )
    slice_shift: int = ConfigLeaf.of(
        default=0,
        says=(
            "how many of the last blocks first add, per channel, the previous and next slice's tokens to each"
            " slice's own, through zero-initialised gates, so the run starts exactly at the control. 0 is every"
            " member so far. Costs no extra forward pass."
        ),
    )
    through_plane: bool = ConfigLeaf.of(
        default=StudyDesign().through_plane,
        says=(
            "give each slice its two depth neighbours as the encoder's other two input channels instead of "
            "three copies of itself. Same slice axis, same pooling, same token / parameter / FLOP count as "
            "the control — the only change is what the three channels hold, so it is the one arm that puts "
            "through-plane information INTO a slice's feature vector rather than mixing depth after the "
            "encoder has already reduced each slice to one."
        ),
    )
    anatomical_prior: bool = ConfigLeaf.of(
        default=False, says="start each diagnosis's slot attention from how the finding is actually read"
    )
    variant: str = ConfigLeaf.of(
        default=SMALL,
        says=(
            "which pretrained encoder. DINOv2-small is the committed default: DINOv3 is refuted as an "
            "improvement at -0.038 and retained only as a diversity member (see "
            "interpretations/kneemri/2026-08-13_dinov2-stays-and-why-dinov3-loses-the-focal-findings.md)."
        ),
        choices=tuple(ENCODERS),
    )
    unfreeze_last: int = ConfigLeaf.of(
        default=Adaptation().unfreeze_last,
        says=(
            "how many of the encoder's trailing blocks train. The restraint is a HYPOTHESIS, not a "
            "measurement: with 58 image-read labels the early generic blocks may have too little supervision "
            "to improve and enough to damage. Capacity is the only axis that has ever cleared our noise "
            "floor, so the hypothesis is worth testing rather than inheriting."
        ),
    )
    unfreeze_offset: int = ConfigLeaf.of(
        default=Adaptation().unfreeze_offset,
        says=(
            "how many blocks at the TOP of the trunk stay FROZEN, which turns --unfreeze-last from a suffix "
            "into a window. Zero is every run before 2026-09-05. Lee et al. (ICLR 2023) choose the block by "
            "the KIND of shift, and the input-level case - natural images to greyscale MRI - wants EARLY "
            "blocks, which a suffix cannot express."
        ),
    )
    unfreeze_stem: bool = ConfigLeaf.of(
        default=Adaptation().unfreeze_stem,
        says=(
            "train the encoder's patch embedding, cls token and position table, which --unfreeze-last "
            "cannot reach at any depth: the trainable set is built out of the window's BLOCKS, so the stem "
            "freezes at 12 as it does at 6. It is the only layer whose input is PIXELS, so it is where "
            "ImageNet RGB against 336 px greyscale MRI is a difference in the DATA. The position table is "
            "additionally resampled 37x37 to 24x24 every forward at this read size, and has never been "
            "allowed to adapt to the grid it is resampled onto (knee-p6fr8)."
        ),
    )
    init: Path | None = ConfigLeaf.of(
        default=None,
        says=(
            "start from a member already on disk instead of from the encoder's published weights plus a fresh"
            " head. The one flag LP-FT needs: stage one trains a head over a FROZEN trunk (--unfreeze-last "
            "0), stage two points --init at that checkpoint and unfreezes at a low backbone rate. Kumar et "
            "al. (arXiv:2202.10054) measure fine-tuning from a fresh head distorting pretrained features ~20x"
            " more than fine-tuning from a fitted one, which is the shape radnet_family produced here: 0.8197"
            " frozen, 0.6716 fine-tuned, same weights. The checkpoint's DESIGN is not read back - this loads "
            "weights into the model these flags describe, so a mismatched architecture raises here rather "
            "than training something else."
        ),
    )
    init_scope: str = ConfigLeaf.of(
        default="all",
        says=(
            "how much of --init is taken. 'all' is LP-FT's case, where stage one's head IS the head stage two"
            " wants. 'trunk' is the CROSS-CORPUS case: a checkpoint fitted on MRNet has three series slots "
            "against this corpus's six, so head.slot_emb is (3, 256) against (6, 256) and a whole-model load "
            "raises - after stage one has already spent its card. The head is not the part that transfers "
            "anyway; feature reuse in the early and middle blocks is (Yosinski 2014, Raghu 2019, Matsoukas "
            "2022). Everything the scope does take is still shape-checked."
        ),
        choices=sorted(INIT_SCOPES),
    )
    head_lr: float = ConfigLeaf.of(default=Adaptation().head_lr)
    backbone_lr: float = ConfigLeaf.of(
        default=Adaptation().backbone_lr,
        says=(
            "how fast the ENCODER adapts. The 125:1 ratio to the head was inherited, never measured, and at "
            "8e-6 the encoder barely moves - which is why our depth bracket came out neutral: unfreezing more"
            " blocks cannot help at a rate where no block moves."
        ),
    )
    layer_decay: float = ConfigLeaf.of(
        default=Adaptation().layer_decay,
        says=(
            "scale the encoder rate DOWN by this factor per block below the top one (0.65-0.75 is the usual "
            "range; 1.0, the default, is the flat rate every ledger number was measured at). Without it depth"
            " and rate can only move together: --unfreeze-last 12 at the rate that makes the top blocks move "
            "applies that same rate to the encoder's most general layers, so a deeper arm is not a deeper "
            "version of the shallow one, it is a different perturbation."
        ),
    )
    weight_decay: float = ConfigLeaf.of(
        default=Adaptation().weight_decay,
        says=(
            "AdamW's decoupled weight decay, applied to every parameter group. 0.02 is the inherited default "
            "and was never re-derived: it was chosen beside a backbone rate of 8e-6, and the rate sweep has "
            "since moved the trunk 5x to 4e-5, so the same decay now competes with steps five times larger. "
            "Decay and rate are not independent — the equilibrium norm a weight settles at is set by their "
            "ratio — so a rate change with the decay held fixed has already changed the regularisation, in a "
            "direction nobody chose."
        ),
    )
    whole_stack: bool = ConfigLeaf.of(
        default=False,
        says=(
            "read EVERY cached slice each step instead of one sampled 3-slice window. Needs a per-slice "
            "pooling or a tubelet trunk, and for a tubelet trunk it is the only path that reaches a window "
            "wider than 3 at all. A step currently sees 18 of a study's ~180 slices, while averaging ten "
            "windows at INFERENCE is worth +0.0156 - the largest single effect measured here - so training "
            "was discarding information it already had. Costs about 4x a step."
        ),
    )
    report_weight: float = ConfigLeaf.of(
        default=0.0,
        says=(
            "also predict each study's REPORT embedding, at this weight (0 = off). Our supervision is twelve "
            "numbers where the report has a median of 129 words in twelve languages; this term supervises the"
            " representation with the rest of it. Training-only - test.csv has no Report column, so inference"
            " is unchanged."
        ),
    )
    encoder_weights: Path | None = ConfigLeaf.of(
        default=None,
        says=(
            "start the encoder from a domain-adapted checkpoint written by kneemri.training.pretrain instead "
            "of the stock natural-image weights. Everything else in the recipe is unchanged, so a run with "
            "this flag and its matched control differ in one variable: what the encoder knew before "
            "supervision began."
        ),
    )
    encoder_adapt_depth: int = ConfigLeaf.of(
        default=ALL_BLOCKS,
        says=(
            "take only the patch embedding and the first N blocks from --encoder-weights, leaving the rest "
            "stock. 0 takes the whole adaptation. The argument for on-domain pretraining only ever concerned "
            "the blocks --unfreeze-last leaves frozen, which supervision cannot write into and so cannot "
            "repair; the top blocks the head reads are the ones a reconstruction objective rewrites hardest, "
            "and two full loads measured on 2026-08-20 lost 0.17 with block 11 moved 55%% and 21%% of its own"
            " norm."
        ),
    )
    lora_rank: int = ConfigLeaf.of(
        default=0,
        says=(
            "adapt every block along a rank-r subspace with the pretrained weights FROZEN, instead of "
            "rewriting the last few blocks. 0 disables it. The base weight stays bit-identical, so what a "
            "small noisy dataset can do to a 1.7B-image representation is bounded by the rank rather than by "
            "the learning rate."
        ),
    )
    recompute_activations: bool = ConfigLeaf.of(
        default=False,
        says=(
            "gradient checkpointing: 19%% less throughput (measured here) for nearly all of the encoder's "
            "activation memory. Needed once a step puts many images through the encoder - whole-stack reading"
            " and higher resolutions both overflow 32 GB, and an overflow THRASHES rather than failing, which"
            " looks like 100%% utilisation at a quarter of the power budget."
        ),
    )
