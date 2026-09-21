"""What a run is fitted TO — the label node of the recipe tree.

Split out of `config_tree` because it is the one node whose leaves are all about the TARGETS rather
than the run: which table, weighted how, with which cells held back. `supervision` is the value
`Targets.pair` takes, so this module is the boundary between a recipe and what the loss sees.
"""

from dataclasses import dataclass
from pathlib import Path

from core.data.labels.targets import CONFIDENCE_FLOOR, Supervision
from kneemri.training.config_leaf import ConfigLeaf


@dataclass(frozen=True)
class TargetConfig:
    """What the model is fitted TO — which labels, weighted how, under which extra losses."""

    calibrated_targets: bool = ConfigLeaf.of(
        default=False,
        says=(
            "replace each report level with what it was MEASURED to be worth against the 57 annotated "
            "studies: 0.08->0.041, 0.28->0.136, 0.68->0.378, 0.82->0.688, 0.94->0.958. Four of the five "
            "assigned values sit outside their own 95%% interval, all too high. Applies to the TRAINING table"
            " only - recalibrating the ruler would raise the AUC by construction."
        ),
    )
    finding_calibration: bool = ConfigLeaf.of(
        default=False,
        says=(
            "on top of --calibrated-targets, shift each finding's level by its own measured offset. The "
            "pooled map assumes a 0.68 is worth 0.378 on every finding; over the reader's positive calls the "
            "radiologist agrees 44.4%% of the time on Contusion and 81.2%% on Synovitis, and the two fail in "
            "OPPOSITE directions. Offsets are empirical-Bayes shrunk (tau=0.294), so a thin finding keeps the"
            " pooled value. REQUIRES --calibrated-targets."
        ),
    )
    severity_weight: bool = ConfigLeaf.of(
        default=False,
        says=(
            "a cell pulls at how often its GRADE was confirmed against the image annotations (0.378 mild / "
            "0.688 moderate / 0.958 severe) rather than at the reader's __conf, which is 0.95 on all three. "
            "Values are untouched — this is the weight, not the target."
        ),
    )
    hold_annotations: bool = ConfigLeaf.of(
        default=False,
        says=(
            "withhold the 58 image-read annotations from the TRAINING table, so this member's predictions on "
            "those studies never saw their radiologist labels. That is the one input "
            "`kneemri.evaluation.supervision_gap` needs and no member we fit can give it: the override "
            "weights them at 3.0, so every other member is scored against a ruler it trained on. The holdout "
            "ruler is re-read WITH them, so this arm stays comparable to every ledger row. It withholds the "
            "annotator's VALUE and NOT the study: all 58 rows still train at their report-derived values, 45 "
            "of them in the fold-0 training split. For a gold read that is actually out of sample, this flag "
            "is not enough — add --hold-annotation-studies."
        ),
    )
    hold_annotation_studies: bool = ConfigLeaf.of(
        default=False,
        says=(
            "withhold the annotated STUDIES themselves: every cell of all 58 is zeroed in the WEIGHT array, "
            "so they contribute nothing to the loss. Requires --hold-annotations, which it implies rather "
            "than replaces. This is what every header claiming a 'clean' or 'held out' gold reading said it "
            "was doing and was not (knee-x4muo); it costs 1.66%% of the training corpus, and at 13 usable "
            "studies the ruler it buys may still be too thin to read."
        ),
    )
    grade_ladder: bool = ConfigLeaf.of(
        default=False,
        says=(
            "read the reader's five levels as four ORDERED THRESHOLDS crossed by one latent score rather than"
            " as five values a logit must land on. Order is all an AUC scores, so the plain loss spends the "
            "axis it reads on absolute probabilities the metric never sees. Thresholds are derived from the "
            "measured agreement rates, and nothing downstream changes: the same per-label logit is stored and"
            " served."
        ),
    )
    label_errors: Path | None = ConfigLeaf.of(
        default=None,
        says=(
            "a two-column table of (StudyInstanceUID, target) cells an independent signal calls mislabelled; "
            "each is given zero loss weight. Written by `kneemri.evaluation.label_errors --flags`, whose "
            "confident joint SIZES its own flag set, so no threshold is fitted here. An annotated cell is "
            "never pruned - the radiologist's reading is the truth the detector was gated against."
        ),
    )
    unread_targets: Path | None = ConfigLeaf.of(
        default=None,
        says=(
            "a three-column table of (StudyInstanceUID, target, value) replacing the SCORE on cells the "
            "report never read - the 27.3%% of pairs carrying the constant 0.28 at verdict UNK. The weight is"
            " untouched, so this changes what those cells pull toward and not how hard. The complement of "
            "--label-errors, which reaches the cell the reader got wrong rather than the cell the reader left"
            " silent. An annotated cell is never overridden."
        ),
    )
    weight_floor: bool = ConfigLeaf.of(default=False, says="an unmentioned finding pulls at 0.25")
    normalise_by_count: bool = ConfigLeaf.of(default=False, says="loss .mean() not /weight-sum")
    per_label_loss: bool = ConfigLeaf.of(
        default=False,
        says=(
            "normalise the loss PER LABEL so each of the twelve contributes equally, as the metric does. "
            "Pooled, a label's share of the gradient tracks how often it is mentioned — Effusion is positive "
            "in 60%% of studies and Fracture in 7%% — while the score averages the twelve the same."
        ),
    )
    rank_weight: float = ConfigLeaf.of(
        default=0.0,
        says=(
            "add a pairwise ranking term at this strength (0 = cross-entropy only). The metric is a mean of "
            "per-label AUCs, which reads ORDER and discards calibration; this term pays only for pairs of "
            "studies ordered the wrong way round."
        ),
    )
    contrast_weight: float = ConfigLeaf.of(
        default=0.0,
        says=(
            "add a per-finding supervised-contrastive term at this strength over what each diagnosis READ "
            "from the study, before the readout collapsed it to a logit (0 = off). Every other term here is a"
            " function of the logits — a dot product against one learned direction — so two studies read for "
            "different reasons can share it. This one asks the representation itself to put torn beside torn "
            "and intact beside intact, per finding, which is the only phrasing this repo has for 'is the "
            "tissue CONNECTED' rather than 'how bright, and where'. Composes with --rank-weight, which the "
            "house recipe always passes, so a contrastive arm stays a single-variable change against its "
            "control; not composed with --distill-weight, --dissent-weight or --pair-memory, whose dispatch "
            "returns a different objective."
        ),
    )
    auc_margin: float = ConfigLeaf.of(
        default=0.0,
        says=(
            "REPLACE cross-entropy with the AUC-margin surrogate at this margin width (0 = off, 1.0 = the "
            "published default and full separation on a sigmoid scale). Cross-entropy optimises calibration "
            "and the metric is twelve rank statistics, so this drops the part of the answer the scoreboard "
            "cannot read rather than adding a term to compete with it. Won CheXpert 2020 (Yuan et al., arXiv "
            "2012.03173). Composed with none of the terms above, which all add themselves to the cross-"
            "entropy this removes."
        ),
    )
    pair_memory: int = ConfigLeaf.of(
        default=0,
        says=(
            "keep this many past steps' readouts so the ranking term can form pairs ACROSS batches (0 = "
            "within-batch only). The metric orders the whole test set, while a batch of 12 can only ask for "
            "the ordering of the twelve studies that happened to be loaded together; batch size here is set "
            "by VRAM, so without this the pair pool is chosen by the card. Requires --rank-weight, since it "
            "changes that term and nothing else."
        ),
    )
    side_weight: float = ConfigLeaf.of(
        default=0.0,
        says=(
            "add a within-study term at this strength asking the two columns of a compartment pair "
            "(medial/lateral meniscus, medial/lateral OA) to be ordered as the reports grade them. Cross-"
            "entropy scores the twelve columns independently, so a head can minimise it on a pair by emitting"
            " the pair's marginal twice; side_blindness measures the Lateral OA column doing exactly that, at"
            " chance on every member."
        ),
    )
    focus_label: str | None = ConfigLeaf.of(
        default=None,
        says=(
            "lean the loss on one or more of the twelve findings, joined by '+' in ONE quoted token (a "
            "finding's name holds a space): --focus-label \"MCL+Lateral OA\". Every unnamed column's loss "
            "weight is multiplied by --unfocused-weight, which at its default of 0 is the hard mask. The head"
            " still emits twelve logits, so the checkpoint stays a member; only the loss leans. An unknown "
            "name RAISES rather than falling through to the plain recipe."
        ),
    )
    unfocused_weight: float = ConfigLeaf.of(
        default=0.0,
        says=(
            "what an unnamed finding's loss weight is multiplied by, 0 masking it outright. The mask has been"
            " run and it failed in a named way: focus_mcl regressed its own target label by -0.0576 paired "
            "and collapsed to a 0.5332 twelve-column mean, because one supervisory signal per study is not "
            "enough to fit an encoder on. A weight strictly inside (0, 1) keeps all twelve signals and only "
            "moves the gradient's balance."
        ),
    )

    @property
    def supervision(self) -> Supervision:
        """What the label switches mean, as the value `Targets.pair` takes.

        These change what the model is FITTED ON while every recorded config keeps agreeing with itself,
        and they map to a data argument rather than a policy object, so `options_to_policies` cannot see
        them — this property is where a test reaches them.
        """
        return Supervision(
            floor=CONFIDENCE_FLOOR if self.weight_floor else 0.0,
            calibrated=self.calibrated_targets,
            finding=self.finding_calibration,
            severity=self.severity_weight,
            annotations=not self.hold_annotations,
            withheld=self.hold_annotation_studies,
            suspect=self.label_errors,
            unread=self.unread_targets,
        )
