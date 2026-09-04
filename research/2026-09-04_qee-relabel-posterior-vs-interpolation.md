# Q-EE — Re-labelling detected-noisy cells: posterior directly, or interpolated with the original label?

Claude's Q-EE (2026-09-04T1600Z) asks two things about re-labelling methods for detected label noise:

1. Do they write the model's posterior directly onto a noisy cell, or do they interpolate between the posterior and the original label — and is there a measured comparison?
2. Is there any published case where re-labelling is applied and the per-class prevalence change is reported alongside the AUC?

## Short answer

- **Interpolation is the norm; pure posterior is rare.** Most methods write a convex combination `ŷ = α·posterior + (1−α)·noisy_label`. The research is about how to set `α`, not about whether to keep the original label at all.
- **Measured comparisons of mixing weight exist, but they are not a direct "posterior vs. original" ablation.** They compare fixed vs. dynamic weights, or different fixed weights, on the same detected set.
- **Per-class prevalence shift is essentially unreported.** BoMD shows a before/after label-distribution figure and a per-label AUC table, but does not tabulate prevalence change next to AUC. I found no other paper that does.

## (1) What the literature actually does

### Canonical interpolation form

The standard label-correction target is:

```
ŷ_i = α_i · p̂_i + (1 − α_i) · ỹ_i
```

where `p̂_i` is a (often sharpened or temporally averaged) model prediction and `ỹ_i` is the original noisy label. This is explicit in the DULC review of the field ([Xu et al., AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/35489/37644)).

### Bootstrapping (Reed et al. 2014)

The earliest instance: `α` is a **fixed scalar** (e.g. 0.6) for every sample. ([Reed et al., 2014](https://arxiv.org/abs/1412.6596), cited in DULC).

### M-correction / DivideMix family (Arazo et al. 2019; Li et al. 2020)

`α` is made **sample-wise dynamic** from a beta mixture model on the loss distribution. Still an interpolation; the original label is never fully discarded for the detected set.

### SELC (Lu & He 2022)

Temporal ensemble prediction is mixed with the original label. `α` is fixed within an epoch, dynamic across epochs. ([SELC](https://arxiv.org/abs/2205.01156), cited in ALR and DULC).

### BoMD (Chen et al., ICCV 2023)

The closest medical multi-label example. For a detected noisy sample it writes:

```
ỹ_i = (1 − λ) · y_i + λ · (γ · 1_|Y| + (1 − γ) · ȳ_i) ⊙ m
```

where `ȳ_i` is the average label of K nearest neighbours in a learned descriptor graph, `1_|Y|` is a uniform smoothing vector, `m` masks confident negatives, and `λ, γ` are fixed mix coefficients. The **original label is retained at weight `1−λ`**. ([BoMD paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_BoMD_Bag_of_Multi-label_Descriptors_for_Noisy_Chest_X-ray_Classification_ICCV_2023_paper.pdf))

### DULC (Xu et al., AAAI 2025)

`α_i` is set to a **class-normalised Jensen–Shannon divergence** between prediction and noisy label, with a linear decay in early epochs. Still interpolation. DULC explicitly argues that the weight should be sample-wise dynamic and class-wise uniform, and provides the cleanest ablation of weight design.

### ALR (Zhang et al., Neural Networks 2025)

A two-phase method: warm-up on original labels, then update targets as a momentum blend of the previous soft target and the current prediction:

```
t_i(e) = α · t_i(e−1) + (1 − α) · p_i(e)
```

with an entropy regulariser that hardens high-confidence soft labels. This is closer to "write the posterior" over time, but it is still a temporal interpolation and the original label influence decays through the momentum chain rather than being replaced outright. ([ALR paper](https://arxiv.org/html/2502.00386v1))

## (2) Measured comparisons of the mixing choice

| study | what is compared | finding |
| :-- | :-- | :-- |
| **BoMD** | fixed `λ = 0.2`, `0.6`, `1.0` on the same KNN-relabelled set | `λ = 0.6` wins; relying mostly on the pseudo-label (`λ = 0.2`) or keeping the original label (`λ = 1.0`) both lose. This is the nearest medical multi-label measurement. |
| **DULC** | dynamic JSD-based `α` vs. static/fixed `α` vs. no normalisation/decay | Dynamic weight beats static by up to 32.8 accuracy points on CIFAR-10 symmetric-90% noise; normalisation and early-epoch decay are also critical. |
| **ALR** | full ALR vs. ablated entropy-loss and warm-up | The entropy-hardening term matters; the paper does not report an ablation that replaces the soft target with the raw posterior. |

**No paper holds the detector fixed and directly compares "write posterior only (α=1)" against "interpolate with original label (α<1)".** The closest are the BoMD `λ` ablation and the DULC static-vs-dynamic ablation; both still live inside the interpolation family.

The theoretical argument against pure posterior is Nodet et al.'s observation that corrected samples contribute full gradients, while reweighted samples contribute little — but this assumes the correction is *accurate*. A wrong correction at full weight injects a clean-looking mislabel, which is why methods keep the original label in the mix (or add uniform smoothing / entropy hardening) as a regulariser.

## (3) Per-class prevalence change reported with AUC

**None found as a numerical table.**

- **BoMD** reports per-label AUC in Tables 4/7 and shows a histogram of label distribution before and after re-labelling (Fig. 4). The caption states the re-labelled distribution is "closer to the original clean label distribution", which is a prevalence claim, but it is visual, not tabulated next to AUC.
- The systematic benchmark papers (DULC, ALR, DivideMix, etc.) report accuracy/mAP/AUC only; prevalence shift is not a reported metric.
- In medical imaging, the standard evaluation uses a clean test-set AUC. The possibility that a denoising method changes class prevalences in the training set — and that this shift might help or hurt ensemble behaviour — is not a reported quantity in the surveyed literature.

## Implication for the project's re-label arm (`knee-f8hv5`)

- If the arm writes the held-out posterior **directly** onto the 5,360 flagged cells, it will drive down positive rates for all findings because the dominant gold disagreement mode is report over-call (108 false positives vs. 32 misses). That is the intended treatment, but it is also a prevalence shift.
- If it **interpolates**, the magnitude of the prevalence shift is governed by `α`. BoMD's `λ = 0.6` suggests keeping a substantial fraction of the original label is beneficial; DULC says the weight should be sample- and class-adaptive rather than global.
- Because the project already has `unread_teacher` as evidence that prevalence shifts transmit between heads, the re-label arm should report **both** the AUC delta and the per-class prevalence change in the new training targets. The literature does not supply a canonical mixing weight for this; it will have to be measured on the 58-study clean subset or on a held-out clean ruler.

## Sources

- Xu et al., *Revisiting Interpolation for Noisy Label Correction*, AAAI 2025. https://ojs.aaai.org/index.php/AAAI/article/view/35489/37644
- Chen et al., *BoMD: Bag of Multi-label Descriptors for Noisy Chest X-ray Classification*, ICCV 2023. https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_BoMD_Bag_of_Multi-label_Descriptors_for_Noisy_Chest_X-ray_Classification_ICCV_2023_paper.pdf
- Zhang et al., *Efficient Adaptive Label Refinement for Label Noise Learning*, Neural Networks 2025. https://arxiv.org/html/2502.00386v1
- Reed et al., *Training Deep Neural Networks on Noisy Labels with Bootstrapping*, 2014. https://arxiv.org/abs/1412.6596
- Nodet et al., *Contrastive Representations for Label Noise Require Fine-Tuning*, IAL 2021 / CEUR-WS. https://ceur-ws.org/Vol-3079/ial2021_paper8.pdf
