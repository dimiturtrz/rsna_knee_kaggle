# Q-H: Published evidence on repairing small, mechanistic label defects

Question from Claude (2026-09-01 18:30Z): Is there a published result where repairing a *small, mechanistically identified* label defect (single-digit percent of cells) moves a downstream classifier more than its share of the data predicts? Context: ~1,240 corpus-wide missed-negation cells (~1% of corpus), with models already moving 0.211 of the way toward the empirical rate unsupervised.

## Short answer

Direct published evidence for a sub-2% mechanistic repair producing a *disproportionate* classifier gain is scarce. The literature strongly supports that **some mislabeled examples are far more harmful than others**, but the experiments that isolate this effect operate at 10-40% synthetic noise, not at the 1% scale of the knee-MRI negation defect. The honest prior is still count-scaling, with a modest upward adjustment for defects that are (a) learned early and (b) look like representative members of the wrong class.

## Evidence that harm is non-uniform (relevant to "more than count-scaling")

### Yuan et al. — "Enhancing Sample Selection by Cutting Mislabeled Easy Examples" (arXiv 2502.08227, 2025)

Key finding: mislabeled examples that the model learns **early** are disproportionately harmful. The authors call these Mislabeled Easy Examples (MEEs).

- Experimental design: CIFAR-10 with 40% instance-dependent label noise (20,000 mislabeled examples). They rank mislabeled examples by the epoch at which a reference model first learns them, then divide into five groups of 4,000.
- Result: a model trained on the **earliest-learned 4,000** mislabeled examples plus 30,000 clean examples generalizes significantly worse than a model trained on the **latest-learned 4,000** mislabeled examples plus the same 30,000 clean examples. Both datasets have the same noise rate (~12%), so the count-scaling prediction would be equal harm; the early/late split breaks that prediction.
- Mechanism: MEEs cluster near the center of their *mislabeled* class in early feature space (median distance ratio r = 0.830, with 53.8% closer to the wrong class center than their true class center). Non-MEEs have r = 3.923.
- Implication for negation repair: missed-negation examples that look like clean positives (the model already predicts positive with high confidence) are the analogue of MEEs. The pre-test finding that models move 0.211 of the way toward the empirical rate unsupervised is consistent with these examples being "easy" in the MEE sense — the model has already partially learned them. This shrinks headroom but also says the residual fixable signal is concentrated in the most harmful subset.

### Hao et al. — "Inaccurate labels in weakly-supervised deep learning" (IEEE JBHI, 2020)

Key finding: identifying and correcting mislabels improves classifier performance in real medical-imaging tasks.

- Tasks: mammographic BI-RADS density classification and DDSM breast cancer diagnosis.
- Scale: experiments inject 10-100% flipped labels; at 10% noise, the method identifies up to 98% of flipped labels by inspecting the top 30% of the dataset, and correcting them improves test AUC.
- Relevance: this is a correction result, but at 10% noise — an order of magnitude above the ~1% negation defect. It does not test whether a 1% repair produces a >1% effect.

## What is NOT in the literature

- No study was found that injects or repairs a 1-2% label defect and reports a classifier gain larger than the defect's frequency share.
- No medical-imaging study specifically repairs missed-negation extraction errors and measures downstream multilabel AUC change.

## Pricing for the knee-MRI negation defect

Given:

- ~1,240 cells, ~1% of corpus.
- 12 of 108 confident FPs are missed-negation, i.e. these examples are currently pushing the model toward false positive.
- Models already move 0.211 of the way toward the empirical rate without any label repair, suggesting a large fraction of the signal is learnable from the raw reports.

Honest estimate:

- **Naive count-scaling**: +0.001 to +0.003 AUC if the 1% defect currently costs 1% of the remaining headroom.
- **MEE-adjusted**: perhaps 2-3x count-scaling for the subset that is both confidently wrong and representative of the wrong class. That still lands in the +0.002 to +0.006 AUC range, i.e. below the 1.5% bar the project treats as a real result.
- The main value of the repair is not a score jump; it is removing a mechanistic bias that can distort every future supervision experiment (including the certainty-ramp arm and any cleanlab-gated reweighting).

## References

- Yuan et al., "Enhancing Sample Selection by Cutting Mislabeled Easy Examples", arXiv:2502.08227, 2025.
- Hao et al., "Inaccurate labels in weakly-supervised deep learning", IEEE Journal of Biomedical and Health Informatics 24(9):2701-2710, 2020.
