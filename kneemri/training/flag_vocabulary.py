"""What the training command line ACCEPTS, apart from what a flag MEANS.

`TrainingOptions` held both and was measured doing it: 693 lines against a 750-line god-file ceiling,
of which two thirds were this argparse declaration. Only this half grows when a switch is added, which
is why that file kept walking toward the gate - the same reason `run_setup.py` was split out of it
before, and the same shape of split.

WHAT DOES NOT MOVE: `TrainingOptions.parse` stays the public entry, because reading argv into the value
is what every caller actually wants. This module is what argv is allowed to say.
"""

import argparse
from pathlib import Path

from core.data.corpus.cache_flags import CacheFlags
from kneemri.models.backbone import ENCODERS, SMALL
from kneemri.models.model import StudyDesign
from kneemri.models.slice_pooling import CHANNELS, SLICE_POOLINGS
from kneemri.models.slot_head import ATTENDED, SLOT_POOLINGS
from kneemri.models.token_pooling import CLS_MEAN, POOL_PARTS
from kneemri.training.gradient_clip import UNCLIPPED
from kneemri.training.init_scope import INIT_SCOPES
from kneemri.training.mixup import OFF
from kneemri.training.pretrained_encoder import ALL_BLOCKS
from kneemri.training.trainer import ONE_CYCLE_WARMUP, Adaptation


class FlagVocabulary:
    """Every flag one training run accepts, and the defaults argv gives them."""

    @classmethod
    def parser(cls) -> argparse.ArgumentParser:
        """The flags, separable from reading them, so a test can ask what argv accepts.

        Every field here has a default in TWO homes — the dataclass and the flag — and a drift between
        them is silent: the tests construct this object directly, so a parser default that moves means
        the command line fits a recipe no test has ever checked. Handing the parser out is what lets
        that be asserted rather than assumed.
        """
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "--fold",
            type=int,
            default=0,
            help="-1 trains on EVERY study: no fold is ever assigned -1, so the holdout comes out "
            "empty and nothing is held back. That is the final-fit mode the public recipes use, and it "
            "cannot report a score by construction - the last epoch is kept rather than the best one.",
        )
        parser.add_argument("--folds", type=int, default=5)
        parser.add_argument("--epochs", type=int, default=8)
        parser.add_argument("--batch", type=int, default=12)
        parser.add_argument(
            "--accumulate",
            type=int,
            default=1,
            help="micro-batches summed into one optimiser step. The effective batch is --batch times "
            "this, at the activation memory of --batch alone, which is what lets a wider encoder hold "
            "the settled recipe's batch: dinov2-base at 336 px and 20 slices asks 56.3 GB at batch 12 "
            "and 28.3 at 6. Costs one forward/backward per micro-batch, and restores the GRADIENT only "
            "- anything pooled within a step still sees --batch studies.",
        )
        parser.add_argument(
            "--mixup",
            type=float,
            default=OFF,
            help="blend each batch with a permutation of itself, pixels and TARGETS together, at a "
            "weight drawn from Beta(a, a). 0.1-0.4 is the usual range; 0, the default, is off and is "
            "what every ledger number was measured under. This is the first label-space regulariser "
            "here: --intensity-aug changes how a study looks and leaves its label alone, which is why "
            "it fired and the overfitting did not care - band_dense_long peaked at epoch 6 and lost "
            "0.0182 over eighteen more while its train loss halved.",
        )
        parser.add_argument(
            "--slot-dropout",
            type=float,
            default=0.0,
            help="withhold each PRESENT slot from a training study with this probability, pixels and "
            "presence mask together. 0, the default, is off and is what every ledger number was "
            "measured under. The first SUPERVISION-space regulariser here: --mixup perturbs the label "
            "and --intensity-aug perturbs the appearance, and neither changes which acquisitions a "
            "study is made of. This corpus varies on exactly that axis - the axial fluid slot fills "
            "100%% of studies while both T1 slots fill 55-60%% - so nearly half of all studies reach "
            "inference with slots the model has only ever seen filled. A study is never emptied.",
        )
        # EVERY FLAG THAT NAMES A CACHE, from the one place that knows the set. These were eight
        # blocks here and eight more in `build_cache`, and the two had already diverged on the band.
        CacheFlags.add_to(parser)
        parser.add_argument(
            "--train-size",
            type=int,
            default=0,
            help="resolution the model actually READS, resampled from the cache. 0 means the cache's "
            "own. Decouples resolution from decoding: a 448 px cache can be read at 392 without a "
            "second 45-minute build, and without changing the batch or the slice count to afford it.",
        )
        parser.add_argument(
            "--studies",
            type=int,
            default=0,
            help="cap the corpus positions this run touches (0 = all of them). A SMOKE knob: it proves "
            "a recipe wires end to end — model built, loss stepped, checkpoint written — in minutes "
            "instead of holding the card for forty to find out a flag was miswired. The holdout keeps "
            "the fold's ratio so both halves of the loop run, and the score it reports means nothing.",
        )
        parser.add_argument("--seed", type=int, default=2026, help="model init and data order ONLY")
        parser.add_argument(
            "--split-seed",
            type=int,
            default=2026,
            help="fold assignment. SEPARATE from --seed on purpose: deriving the split from the "
            "training seed gives every seed a different holdout, so two runs are not comparable and "
            "cross-scoring them leaks training data.",
        )
        parser.add_argument(
            "--calibrated-targets",
            action="store_true",
            help="replace each report level with what it was MEASURED to be worth against the 57 "
            "annotated studies: 0.08->0.041, 0.28->0.136, 0.68->0.378, 0.82->0.688, 0.94->0.958. Four of "
            "the five assigned values sit outside their own 95%% interval, all too high. Applies to the "
            "TRAINING table only - recalibrating the ruler would raise the AUC by construction.",
        )
        parser.add_argument(
            "--finding-calibration",
            action="store_true",
            help="on top of --calibrated-targets, shift each finding's level by its own measured "
            "offset. The pooled map assumes a 0.68 is worth 0.378 on every finding; over the reader's "
            "positive calls the radiologist agrees 44.4%% of the time on Contusion and 81.2%% on "
            "Synovitis, and the two fail in OPPOSITE directions. Offsets are empirical-Bayes shrunk "
            "(tau=0.294), so a thin finding keeps the pooled value. REQUIRES --calibrated-targets.",
        )
        cls._curriculum_arguments(parser)
        parser.add_argument(
            "--vram-gb",
            type=float,
            default=0.0,
            help="hard cap on this run's share of the card, in GB (0 = 80%% of it, the band the arms "
            "converged on; naming a number also makes the run WAIT for that much to be free). Two "
            "reasons for capping at all. It lets "
            "another project hold the rest of the GPU. And it converts the WORST failure mode we have "
            "into the best one: Windows pages GPU memory over PCIe rather than raising OOM, so a run "
            "that does not fit crawls at a quarter of the power budget while reporting 100%% "
            "utilisation. Under a cap, PyTorch raises OOM immediately instead.",
        )
        # `--compile-encoder` is GONE the way `--bf16` went: compilation is the default, and the flag
        # that used to switch it on now accepts a value nothing reads. Twenty-two of thirty-six arms
        # were running eager for no reason but not naming `$THROUGHPUT` — 21% of every step, on the
        # bulk of the arithmetic, thrown away by an omission rather than by a decision.
        parser.add_argument(
            "--eager-encoder",
            dest="compile_encoder",
            action="store_false",
            help="skip torch.compile on the encoder. THE DEFAULT IS COMPILED: measured -21%% per step "
            "(150 ms eager against 119 ms) for one 2.2 s compilation per process, and score-neutral — "
            "the throughput pair scored 0.7604 plain against 0.7606 compiled, a fifth of the noise "
            "floor. Compiled in place (`nn.Module.compile`), so parameter names are unchanged and the "
            "checkpoint loads through the inference path. This escape exists for the throughput "
            "CONTROLS, which have to run eager to measure what compiling is worth, and for a machine "
            "with no Triton — upstream ships it everywhere except Windows, where `triton-windows` is "
            "pinned in pyproject for exactly this.",
        )
        parser.add_argument(
            "--lora-rank",
            type=int,
            default=0,
            help="adapt every block along a rank-r subspace with the pretrained weights FROZEN, "
            "instead of rewriting the last few blocks. 0 disables it. The base weight stays "
            "bit-identical, so what a small noisy dataset can do to a 1.7B-image representation is "
            "bounded by the rank rather than by the learning rate.",
        )
        # `--bf16` is GONE, not deprecated. It existed only so fourteen arms written before bf16 became
        # the default would still parse; those arms now take the recipe from experiments/recipe.sh and
        # none of them names it. A flag that accepts the default and does nothing is a thing to keep in
        # step forever, and this repository has one dev and no live system — the old invocation is in
        # git if it is ever wanted.
        parser.add_argument(
            "--fp16",
            dest="bf16",
            action="store_false",
            help="train in fp16 with a loss scaler instead of the bf16 default. THE DEFAULT IS bf16: it "
            "matches fp16's matmul throughput on this card (native from sm_80) and removes scale "
            "collapse, since bf16 carries fp32's exponent range and needs no scaler. Serving is a "
            "SEPARATE choice and is fp16 — a T4 has fp16 tensor cores and none for bf16, 65 TFLOPS "
            "against 8.1, and fp16 has MORE mantissa than bf16, so the only thing given up is range. "
            "Measured on real studies, our members peak at 3%% of fp16's ceiling, and `Ensemble.arithmetic` "
            "does the swap while `predict_member` redoes any non-finite pass in fp32.",
        )
        parser.add_argument(
            "--recompute-activations",
            action="store_true",
            help="gradient checkpointing: 19%% less throughput (measured here) for nearly all of the encoder's "
            "activation memory. Needed once a step puts many images through the encoder - whole-stack "
            "reading and higher resolutions both overflow 32 GB, and an overflow THRASHES rather than "
            "failing, which looks like 100%% utilisation at a quarter of the power budget.",
        )
        parser.add_argument(
            "--ema",
            action="store_true",
            help="score and save an exponential moving average of the weights rather than the live "
            "ones. Universal in the public notebooks, and aimed at the 0.0062 seed spread we measured: "
            "it stops the answer depending on which step training happened to stop at.",
        )
        parser.add_argument("--intensity-aug", action="store_true", help="gamma/blur/contrast/noise too")
        parser.add_argument("--weight-floor", action="store_true", help="an unmentioned finding pulls at 0.25")
        parser.add_argument("--normalise-by-count", action="store_true", help="loss .mean() not /weight-sum")
        parser.add_argument(
            "--severity-weight",
            action="store_true",
            help="a cell pulls at how often its GRADE was confirmed against the image annotations "
            "(0.378 mild / 0.688 moderate / 0.958 severe) rather than at the reader's __conf, which is "
            "0.95 on all three. Values are untouched — this is the weight, not the target.",
        )
        parser.add_argument(
            "--hold-annotations",
            action="store_true",
            help="withhold the 58 image-read annotations from the TRAINING table, so this member's "
            "predictions on those studies never saw their radiologist labels. That is the one input "
            "`kneemri.evaluation.supervision_gap` needs and no member we fit can give it: the override "
            "weights them at 3.0, so every other member is scored against a ruler it trained on. The "
            "holdout ruler is re-read WITH them, so this arm stays comparable to every ledger row.",
        )
        cls._label_quality_arguments(parser)
        parser.add_argument(
            "--per-label-loss",
            action="store_true",
            help="normalise the loss PER LABEL so each of the twelve contributes equally, as the metric "
            "does. Pooled, a label's share of the gradient tracks how often it is mentioned — Effusion "
            "is positive in 60%% of studies and Fracture in 7%% — while the score averages the twelve "
            "the same.",
        )
        parser.add_argument(
            "--rank-weight",
            type=float,
            default=0.0,
            help="add a pairwise ranking term at this strength (0 = cross-entropy only). The metric is "
            "a mean of per-label AUCs, which reads ORDER and discards calibration; this term pays only "
            "for pairs of studies ordered the wrong way round.",
        )
        cls._teacher_arguments(parser)
        parser.add_argument(
            "--contrast-weight",
            type=float,
            default=0.0,
            help="add a per-finding supervised-contrastive term at this strength over what each "
            "diagnosis READ from the study, before the readout collapsed it to a logit (0 = off). Every "
            "other term here is a function of the logits — a dot product against one learned direction — "
            "so two studies read for different reasons can share it. This one asks the representation "
            "itself to put torn beside torn and intact beside intact, per finding, which is the only "
            "phrasing this repo has for 'is the tissue CONNECTED' rather than 'how bright, and where'. "
            "Composes with --rank-weight, which the house recipe always passes, so a contrastive arm "
            "stays a single-variable change against its control; not composed with --distill-weight, "
            "--dissent-weight or --pair-memory, whose dispatch returns a different objective.",
        )
        parser.add_argument(
            "--auc-margin",
            type=float,
            default=0.0,
            help="REPLACE cross-entropy with the AUC-margin surrogate at this margin width (0 = off, "
            "1.0 = the published default and full separation on a sigmoid scale). Cross-entropy "
            "optimises calibration and the metric is twelve rank statistics, so this drops the part of "
            "the answer the scoreboard cannot read rather than adding a term to compete with it. Won "
            "CheXpert 2020 (Yuan et al., arXiv 2012.03173). Composed with none of the terms above, "
            "which all add themselves to the cross-entropy this removes.",
        )
        parser.add_argument(
            "--pair-memory",
            type=int,
            default=0,
            help="keep this many past steps' readouts so the ranking term can form pairs ACROSS batches "
            "(0 = within-batch only). The metric orders the whole test set, while a batch of 12 can only "
            "ask for the ordering of the twelve studies that happened to be loaded together; batch size "
            "here is set by VRAM, so without this the pair pool is chosen by the card. Requires "
            "--rank-weight, since it changes that term and nothing else.",
        )
        parser.add_argument(
            "--side-weight",
            type=float,
            default=0.0,
            help="add a within-study term at this strength asking the two columns of a compartment pair "
            "(medial/lateral meniscus, medial/lateral OA) to be ordered as the reports grade them. "
            "Cross-entropy scores the twelve columns independently, so a head can minimise it on a pair "
            "by emitting the pair's marginal twice; side_blindness measures the Lateral OA column doing "
            "exactly that, at chance on every member.",
        )
        cls._focus_arguments(parser)
        cls._sampling_arguments(parser)
        parser.add_argument(
            "--slot-pooling",
            choices=tuple(SLOT_POOLINGS),
            default=ATTENDED,
            help="how the twelve diagnoses read the six slots. 'attend' gives each diagnosis its own "
            "query over the present slots; 'mean' pools them uniformly and is the ABLATION - it exists "
            "to give the attention a denominator, since every number in the ledger was measured on a "
            "recipe that already contained it. 'attend_fixed' is the attention again with the "
            "anatomical prior held as a BUFFER rather than as a warm start, so a gradient step cannot "
            "erase it - the form aprior_dense's washout says was never actually tested.",
        )
        parser.add_argument(
            "--pool",
            choices=tuple(POOL_PARTS),
            default=CLS_MEAN,
            help="how a slice's patch tokens become one vector. 'cls_mean' is the class token beside "
            "the patch mean and is what every member on disk was fitted with; 'cls_mean_focal' adds "
            "the mean of each channel's top eighth over the patch grid, which a plain mean dilutes - "
            "the small findings (meniscus, ACL, MCL) are the ones that occupy a fraction of the field.",
        )
        parser.add_argument(
            "--slice-pooling",
            choices=tuple(SLICE_POOLINGS),
            default=CHANNELS,
            help="how a group of slices becomes one feature. 'channels' stacks three slices as RGB, "
            "which couples a group's physical thickness to the slice count; 'max' encodes each slice "
            "on its own and takes the elementwise max, which does not, and is what the .903 reference "
            "does. 'attend' encodes each slice alone as 'max' does and then pools them by learned "
            "attention instead, which keeps WHICH slice mattered - the operation the post-MRNet "
            "literature uses. 'max' and 'attend' cost three encoder passes per group.",
        )
        parser.add_argument(
            "--slice-regions",
            type=int,
            default=StudyDesign().slice_regions,
            help="how many depth regions 'max' pooling takes its maximum within before averaging over "
            "them. 1 is one maximum over the whole band, which is what every member on disk was fitted "
            "with; higher values give each part of the depth its own maximum, so a quiet peripheral "
            "finding stops competing with a loud central one (band_wide and band_dense each bought one "
            "end of the band at the other's expense). Free - no parameters, no extra encoder passes.",
        )
        parser.add_argument("--labels", type=Path, required=True, help="label table to TRAIN on")
        parser.add_argument(
            "--eval-labels",
            type=Path,
            default=None,
            help="table to SCORE against (default: --labels). Hold this fixed across arms when the "
            "training labels are the variable, or the arms are measured with different rulers.",
        )
        cls._encoder_arguments(parser)
        cls._optimiser_arguments(parser)
        parser.add_argument(
            "--train-fraction",
            type=float,
            default=1.0,
            help="share of the TRAINING studies to fit on, holdout untouched — the data-scaling knob. "
            "Separate from --studies, which cuts both halves so a smoke still exercises the whole loop; "
            "this cuts one side only, so the score is comparable to every full-data row in the ledger.",
        )
        parser.add_argument("--report-vectors", type=Path, default=None, help="npz from report_embeddings")
        parser.add_argument("--out", type=Path, default=None)
        parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
        return parser

    @staticmethod
    def _label_quality_arguments(parser: argparse.ArgumentParser) -> None:
        """What the loss is asked to BELIEVE about a cell, apart from how the objective is shaped.

        Grouped because each rewrites a supervision value before any term reads it — one re-reads the
        reader's levels as an order, the other withdraws belief from named cells entirely — rather than
        adding a term to the objective. `parser` reached its statement ceiling and this is the seam it
        pointed at.
        """
        parser.add_argument(
            "--grade-ladder",
            action="store_true",
            help="read the reader's five levels as four ORDERED THRESHOLDS crossed by one latent score "
            "rather than as five values a logit must land on. Order is all an AUC scores, so the plain "
            "loss spends the axis it reads on absolute probabilities the metric never sees. Thresholds "
            "are derived from the measured agreement rates, and nothing downstream changes: the same "
            "per-label logit is stored and served.",
        )
        parser.add_argument(
            "--label-errors",
            type=Path,
            default=None,
            help="a two-column table of (StudyInstanceUID, target) cells an independent signal calls "
            "mislabelled; each is given zero loss weight. Written by "
            "`kneemri.evaluation.label_errors --flags`, whose confident joint SIZES its own flag set, so "
            "no threshold is fitted here. An annotated cell is never pruned - the radiologist's reading "
            "is the truth the detector was gated against.",
        )
        parser.add_argument(
            "--unread-targets",
            type=Path,
            default=None,
            help="a three-column table of (StudyInstanceUID, target, value) replacing the SCORE on cells "
            "the report never read - the 27.3% of pairs carrying the constant 0.28 at verdict UNK. The "
            "weight is untouched, so this changes what those cells pull toward and not how hard. The "
            "complement of --label-errors, which reaches the cell the reader got wrong rather than the "
            "cell the reader left silent. An annotated cell is never overridden.",
        )

    @staticmethod
    def _curriculum_arguments(parser: argparse.ArgumentParser) -> None:
        """How the epoch loop is shaped OVER TIME — the rate's path, where the loop stops, and how fast
        the label table is trusted.

        Grouped because each one answers the same question at a different layer: what an epoch is worth
        relative to the epochs around it. A ramp is a curriculum over weights the way one-cycle is a
        curriculum over the rate, and patience is where both are allowed to end.
        """
        parser.add_argument("--one-cycle", action="store_true", help="attach a OneCycleLR schedule")
        parser.add_argument(
            "--patience",
            type=int,
            default=0,
            help="stop after this many epochs without improving the holdout (0 = run every epoch). "
            "The twelve-epoch cycle peaks at epoch 6 and never recovers, and the best epoch is the one "
            "whose weights are kept, so the tail produces checkpoints that are discarded. The SCHEDULE "
            "is still built for the full count - this ends the loop, it does not compress the anneal.",
        )
        parser.add_argument(
            "--certainty-ramp",
            type=int,
            default=0,
            help="epochs over which per-cell weights rise from the confidence floor to their full trust "
            "(0 = off). Early epochs therefore treat every cell as equally uncertain, and later epochs "
            "recover the full certainty-weighted table. The total weight mass is held constant so the "
            "loss scale matches the un-ramped control.",
        )

    @staticmethod
    def _focus_arguments(parser: argparse.ArgumentParser) -> None:
        """WHICH of the twelve findings the loss leans on, and how hard — one policy in two halves.

        Grouped away from the recipe flags for the same reason the sampling pair is: a strength with no
        findings named leans on nothing, and named findings at strength 1.0 are the plain recipe rather
        than a second one. Neither half is readable without the other.
        """
        parser.add_argument(
            "--focus-label",
            default=None,
            help="lean the loss on one or more of the twelve findings, joined by '+' in ONE quoted "
            'token (a finding\'s name holds a space): --focus-label "MCL+Lateral OA". Every unnamed '
            "column's loss weight is multiplied by --unfocused-weight, which at its default of 0 is the "
            "hard mask. The head still emits twelve logits, so the checkpoint stays a member; only the "
            "loss leans. An unknown name RAISES rather than falling through to the plain recipe.",
        )
        parser.add_argument(
            "--unfocused-weight",
            type=float,
            default=0.0,
            help="what an unnamed finding's loss weight is multiplied by, 0 masking it outright. The "
            "mask has been run and it failed in a named way: focus_mcl regressed its own target label "
            "by -0.0576 paired and collapsed to a 0.5332 twelve-column mean, because one supervisory "
            "signal per study is not enough to fit an encoder on. A weight strictly inside (0, 1) keeps "
            "all twelve signals and only moves the gradient's balance.",
        )

    @staticmethod
    def _sampling_arguments(parser: argparse.ArgumentParser) -> None:
        """WHICH studies an epoch descends on, as opposed to what it reads or how it is scored.

        Grouped away from the recipe flags because they are one policy in two halves — an alpha without
        a spread file weights nothing, and a spread file at alpha zero is the matched control rather
        than a second recipe.
        """
        parser.add_argument(
            "--sample-spread",
            type=Path,
            default=None,
            help="draw each epoch's studies in proportion to how much a reference pool DISAGREED about "
            "them, from a file written by kneemri.evaluation.train_disagreement. Every arm on the ledger "
            "draws a uniform permutation instead; residual_split put ~70%% of the pool's errors in the "
            "band the family splits on, and this is the only lever aimed at it. Loss-ranked mining is "
            "the conventional alternative and is wrong here: ~87%% of the residual is aleatoric, so it "
            "mines label noise. The epoch keeps its length, so the compute is matched.",
        )
        parser.add_argument(
            "--sample-alpha",
            type=float,
            default=0.5,
            help="the share of the draw --sample-spread controls; the rest is a uniform floor. 0.0 is "
            "flat weights and is how the arm runs its own matched control — NOT the ledger's uniform "
            "policy, since the draw is with replacement at every alpha.",
        )
        parser.add_argument(
            "--sample-ramp",
            type=int,
            default=0,
            help="reach --sample-alpha over this many epochs instead of at the first step, which turns "
            "the tilt into a curriculum: the epoch starts near-uniform and spends its later passes on "
            "the studies the reference pool could not resolve. 0 is the immediate tilt every sampled arm "
            "so far ran. The static tilt COSTS on this corpus (disagreed_full 0.8471 against its own "
            "alpha-zero control at 0.8507) while a schedule over label trust PAYS (certainty_ramp "
            "0.8584), which is the contrast this flag exists to test on the disagreement signal.",
        )
        parser.add_argument(
            "--corpus-oversample",
            type=float,
            default=0.0,
            help="up-weight non-target corpora by this factor when drawing each epoch (0 = uniform draw). "
            "Requires the labels table to carry a `corpus` column, as written by build_merged. Mutually "
            "exclusive with --sample-spread: both are epoch-level draw policies and cannot both steer the "
            "same batch.",
        )
        parser.add_argument(
            "--target-corpus",
            type=str,
            default="canonical",
            help="the corpus tag that mixed-corpus sampling treats as the target domain. Default matches "
            "the scheme name build_merged uses for the RSNA competition corpus.",
        )

    @staticmethod
    def _optimiser_arguments(parser: argparse.ArgumentParser) -> None:
        """HOW the weights are moved, as opposed to what they read or what they are asked to predict.

        These three are the last constants in `trainer.py` that no arm has ever varied: every number in
        the ledger was measured at weight decay 0.02, at 15% warmup, and with no gradient ceiling at all.
        None of the three was chosen against this corpus — they came in with the reference recipe, at a
        backbone rate 5x lower than the one the rate sweep settled on. They are grouped because they
        describe the OPTIMISER and its schedule, and because each defaults to the inherited value, so
        every existing arm parses to exactly the run it already measured.
        """
        parser.add_argument(
            "--grad-clip",
            type=float,
            default=UNCLIPPED,
            help="ceiling on the global gradient norm of one optimiser step (0 = unbounded, which is "
            "what every number in the ledger was measured under). The failure it addresses is measured: "
            "native224_long slid -0.0165 over twelve extra epochs while its training loss kept falling, "
            "which is the shape a few large late steps produce. Clipping bounds how far one batch may "
            "move the weights; it does not change where the walk is allowed to end up, which is "
            "--weight-decay. 1.0 is the usual value in the ViT recipes this project's trunk came from.",
        )
        parser.add_argument(
            "--weight-decay",
            type=float,
            default=Adaptation().weight_decay,
            help="AdamW's decoupled weight decay, applied to every parameter group. 0.02 is the "
            "inherited default and was never re-derived: it was chosen beside a backbone rate of 8e-6, "
            "and the rate sweep has since moved the trunk 5x to 4e-5, so the same decay now competes "
            "with steps five times larger. Decay and rate are not independent — the equilibrium norm a "
            "weight settles at is set by their ratio — so a rate change with the decay held fixed has "
            "already changed the regularisation, in a direction nobody chose.",
        )
        parser.add_argument(
            "--warmup",
            type=float,
            default=ONE_CYCLE_WARMUP,
            help="share of the one-cycle schedule spent warming UP before the anneal begins (0.15 is "
            "the inherited default and what every ledger number was measured at). Over twelve epochs "
            "that is 1.8 epochs climbing to the peak rate on a run whose best epoch is 7, so a seventh "
            "of the budget is spent below the rate the recipe was tuned for. Read with --one-cycle only "
            "— without a schedule there is no warmup to size, and the flag is then inert.",
        )

    @staticmethod
    def _teacher_arguments(parser: argparse.ArgumentParser) -> None:
        """What a TEACHER teaches and where its readings come from — every flag that needs another model.

        Split out of `parser` for the same reason `_encoder_arguments` was, on the same ceiling: adding
        the representation pair crossed it. The group comes out whole — each weight here is dead without
        one of the two files, and neither file is read by anything else.
        """
        parser.add_argument(
            "--teacher-targets",
            type=Path,
            default=None,
            help="npz of an ensemble's scores for the TRAINING studies, read by --distill-weight and by "
            "--dissent-weight. "
            "Its members must not have seen the holdout this arm is scored on: a --fold -1 member saw "
            "every study, so teaching from one leaks the evaluation labels into the student.",
        )
        parser.add_argument(
            "--teacher-features",
            type=Path,
            default=None,
            help="npz of a teacher's pooled REPRESENTATIONS of the TRAINING studies, read by "
            "--feature-weight. A different file from --teacher-targets and under a different array name, "
            "so handing one where the other belongs raises before the card is touched. "
            "Its members must not have seen the holdout this arm is scored on: a --fold -1 member saw "
            "every study, so teaching from one leaks the evaluation labels into the student.",
        )
        parser.add_argument(
            "--distill-weight",
            type=float,
            default=0.0,
            help="add a pairwise term at this strength asking the student to reproduce an ENSEMBLE's "
            "ordering of these studies (0 = off). Not the usual KL on softened probabilities, which "
            "targets calibration the metric discards. The reader emits five levels, so a pair of studies "
            "on the SAME level teaches the label term nothing — and a finding present in 7%% of studies "
            "strands roughly 87%% of its pairs that way. A continuous teacher orders all of them, so the "
            "gain should concentrate on the rarest findings. Requires --teacher-targets.",
        )
        parser.add_argument(
            "--feature-weight",
            type=float,
            default=0.0,
            help="add a cosine term at this strength pulling what the student READ from a study onto "
            "what a TEACHER read from it, through a learned projection between the two widths (0 = off). "
            "Distillation at the representation rather than at the verdict: --distill-weight teaches the "
            "teacher's twelve-way ORDERING, which is its head, and swin_distill_fit measured that at "
            "-0.0077 against a cold control. A large value here makes the label term a probe rather than "
            "a target, which is what turns a stage into a PRETRAIN whose product is an encoder. "
            "Requires --teacher-features.",
        )
        parser.add_argument(
            "--dissent-weight",
            type=float,
            default=0.0,
            help="add a pairwise ranking term at this strength whose pairs are re-weighted toward the "
            "ones an ENSEMBLE orders WRONG (0 = off). The same teacher file --distill-weight reads, for "
            "the opposite purpose: not to reproduce the pool's ordering but to spend the member's "
            "capacity where that ordering fails, which is the one way to raise ensemble ambiguity "
            "without raising member error. Anchored to the labels throughout — a pair is only ever paid "
            "for in the direction the reports order it. Requires --teacher-targets.",
        )

    @staticmethod
    def _encoder_arguments(parser: argparse.ArgumentParser) -> None:
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
