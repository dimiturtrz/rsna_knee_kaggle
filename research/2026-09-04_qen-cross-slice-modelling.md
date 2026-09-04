# Q-EN — what does cross-slice / 3D modelling buy over per-slice-then-pool?

Claude's question: every current member reads slices independently through a 2D trunk and pools them (`--slice-pooling max --whole-stack`). Nothing models the volume. What is the measured gain of cross-slice modelling over this design, specifically in knee MRI, and what is the cheapest way to add it without a new trunk?

## 1. Matched knee-MRI comparisons: 3D gains are real but modest

The cleanest public matched comparison I could find is in the KneeXNet paper (Sun et al., 2025), which trains a 2D CNN and a 3D CNN from scratch on the same MRNet data (1,370 exams, 1,130 training). Both use the same preprocessing and the same three diagnostic tasks:

| Task | 2D CNN per-slice | 3D CNN | Δ |
|---|---|---|---|
| Abnormality | 0.923 ± 0.008 | 0.937 ± 0.007 | +0.014 |
| ACL tear | 0.896 ± 0.010 | 0.912 ± 0.009 | +0.016 |
| Meniscal tear | 0.889 ± 0.009 | 0.905 ± 0.008 | +0.016 |

Source: Table 3 of ["Leveraging spatial dependencies and multi-scale features for automated knee injury detection on MRI diagnosis"](https://pmc.ncbi.nlm.nih.gov/articles/PMC12088959/).

That is a consistent but small gain (~0.015 AUC) from switching to a volumetric trunk. Caveats:
- The 2D and 3D architectures are not identical depth/channel ladders; the comparison is "matched pipeline" rather than "same backbone with/without depth".
- MRNet is only 1,130 training studies, smaller than our ~4,400.

CoPAS (Qiu et al., 2024) reports a larger gain (0.812 vs MRNet 0.781 average AUC on 12 abnormalities, 1,748 subjects), but the lift bundles 3D ResNet branches, multi-sequence co-plane attention, and multi-task learning together. The ablation confirming the isolated contribution of the cross-slice attention is in the supplementary material, which I did not retrieve. Source: ["Learning co-plane attention across MRI sequences assists radiologists in diagnosing twelve types of knee abnormalities"](https://pmc.ncbi.nlm.nih.gov/articles/PMC11368947/).

## 2. Data scaling: 3D capacity has the adverse corpus-size dependence Claude expected

AnyMC3D (Liu et al., Siemens Healthineers, 2025) is the strongest general evidence. They benchmark 12 classification tasks across CT and MRI and compare adapted 2D foundation models against from-scratch 3D CNNs, 3D medical FMs, and 2.5D baselines. Their headline finding is:

> "2D-based methods surpass 3D architectures for 3D classification" when the 2D FM is properly adapted (LoRA) and slice embeddings are fused with query-based attention pooling.

Source: ["Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification"](https://arxiv.org/abs/2512.12887).

Key data-regime result: on one task, AnyMC3D (DINOv3) with 20% of the data (only 39 positive cases) reaches 0.924 AUC, while a 3D DenseNet with 60% of the data reaches 0.741. With 20% data they beat the 3D model trained on three times as much data. This is exactly the pattern Claude suspected: 3D conv needs data to pay off, and at small-to-medium corpus sizes a strong 2D backbone plus lightweight slice reasoning wins.

## 3. Cheapest cross-slice mechanism: learned attention over slice embeddings

AnyMC3D's slice-fusion ablation on task T5 (trauma CT) is the most direct mechanism comparison:

| Fusion method | Trainable params | AUC |
|---|---|---|
| Query-based attention pooling (Ours) | 0.001 M | 0.962 |
| Average pooling | 0 | 0.958 |
| Transformer encoder | 7.0 M | 0.950 |
| Max pooling | 0 | 0.946 |
| Median pooling | 0 | 0.944 |
| LSTM | 5.5 M | 0.903 |

Source: Table 5 of the AnyMC3D paper.

Takeaways:
- A learned query over slice embeddings beats every simple pooling rule and every sequential model.
- The margin over max pooling is 0.016 AUC; the margin over average is 0.004 AUC.
- Sequential models (LSTM, Transformer) add millions of parameters and underperform because medical volumes have variable coverage and anisotropic spacing.

This maps directly to our pipeline: keep the DINO trunk, produce per-slice embeddings, add one or a few learned query vectors, and attend over the slice sequence before the head. This is the cheapest trunk-free cross-slice mechanism currently backed by measurement.

BIPON (Wang et al., 2026) independently reaches a similar design for knee MRI specifically: a "multi-scale attention aggregator" applies parallel temporal pooling at multiple kernel sizes over slice embeddings, then reweights slices by attention. Their ablation reports that removing this aggregator and reverting to mean pooling degrades discrimination, calibration, and decision quality. I could not extract the exact AUC drop from the PDF text, but the architectural direction aligns with AnyMC3D. Source: ["Forensic-oriented injury and abnormality assessment in sports medicine via a biomechanically-informed predictive framework"](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1759763/full).

## 4. Does the pooling rule itself matter?

Yes, but the best rule depends on how localized the pathology signal is.

- MRNet (Bien et al., 2018) chose **max pooling across slices** because knee abnormalities are small and sparse; the most abnormal slice carries the signal. Source: ["Deep-learning-assisted diagnosis for knee magnetic resonance imaging"](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/).
- MRPyrNet (Dunnhofer et al., 2021) finds that for ACL tears, max and average pooling inside their pyramidal detail module are similar (ACL produces high signal), but for meniscal tears **average pooling is better** because meniscal signal is more balanced across slices. Source: ["Improving MRI-based Knee Disorder Diagnosis with Pyramidal Feature Details"](https://proceedings.mlr.press/v143/dunnhofer21a/dunnhofer21a.pdf).
- AnyMC3D's table above ranks average above max for slice-embedding aggregation.

No knee-MRI paper reports a direct ablation of max-vs-mean-vs-attention *across slices* with the same 2D backbone. The strongest evidence is therefore AnyMC3D (across-slice, cross-modality) and MRPyrNet (within-slice sub-region pooling).

## 5. Bottom line

1. **In knee MRI, a full 3D CNN trunk gives about +0.015 AUC over a matched from-scratch 2D CNN** (MRNet-scale data). It is not zero, but it is small.
2. **The corpus-size dependence favors 2D-plus-attention over 3D conv** at our scale: AnyMC3D shows a LoRA-adapted 2D FM with query-based slice attention outperforms 3D architectures and is 3× more data-efficient.
3. **The cheapest trunk-free cross-slice mechanism is a learned query attention pooling over slice embeddings** (AnyMC3D: +0.004 over average, +0.016 over max, with only 0.001M parameters). A multi-scale attention aggregator over slices is a knee-MRI-specific variant of the same idea.
4. **The pooling rule matters**, but attention removes the need to commit to max vs mean; it learns to up-weight the slices that carry the finding.

Practical recommendation: instead of replacing the DINO trunk with a 3D conv net, run an arm that keeps the trunk and replaces `--slice-pooling max` with a lightweight attention-based slice aggregator (one learned query per label, or a small multi-scale temporal-attention block). The expected gain is small but the cost is tiny, and it directly tests the one structural axis that has not been varied.
