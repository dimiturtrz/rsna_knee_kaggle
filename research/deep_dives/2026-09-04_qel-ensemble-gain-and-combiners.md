# Q-EL — Ensemble gain size and combiners in medical imaging / knee MRI

2026-09-04

## Bottom line for Claude

- A **+0.016 AUC** ensemble gain over the best single member is **well inside the published range** for medical-image classification, though it sits on the smaller side when the best single is already strong.
- Reported gains are **not a fixed constant**: they are large when single models are mid-quality and make different errors, and they shrink toward **~+0.001 AUC** once single models are near the board ceiling.
- On this specific competition, public evidence says a **genuinely cross-family blend also buys only ~+0.001 AUC**; our near-homogeneous DINO pool is therefore not the only explanation for the flatness.
- **Learned combiners** (stacking, weighted averaging, input-conditioned weights) can help when base models have *complementary, predictable* strengths, but in medical imaging simple mean / soft-vote usually matches or beats them. A card-free offline test on the 58-study gold set is the right next step.

## What the literature reports for ensemble-over-best-single gains

### Breast mammography — a clean +0.03 AUC example

Shen et al. (2019) trained four all-convolutional networks for breast-cancer detection on screening mammography. On CBIS-DDSM the best single model reached AUC **0.88**, and averaging four models raised it to **0.91** (+0.03). On INbreast the best single reached **0.95**, and the four-model average reached **0.98** (+0.03) [[Shen et al., Nature Medicine 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6715802/)].

This is the strongest directly comparable number: heterogeneous-ish CNNs, image classification, AUC metric, and a gain three times larger than the +0.016 in question.

### Heterogeneous CNN stacking on four medical-image datasets — up to +13% F1

Müller et al. (IEEE Access 2022) ran a reproducible study across CHMNIST (histology), COVID (chest X-ray), ISIC (dermoscopy), and DRD (diabetic retinopathy). They stacked **nine different CNN architectures** (DenseNet, EfficientNet, InceptionResNetV2, MobileNetV2, ResNeXt, ResNet, VGG, Xception, Vanilla).

Stacking produced the largest gains on the two hardest datasets: **+13% F1 on ISIC** and **+12% F1 on DRD**, versus **+2% on COVID** and **+1% on CHMNIST**. Five-fold bagging of a single architecture (EfficientNetB4) was close but smaller: **+11% ISIC, +9% DRD, +2% COVID, +1% CHMNIST** [[Müller et al., IEEE Access 2022](https://arxiv.org/abs/2201.11440)].

The authors also measured AUC for the pooling functions: the naïve "Best Model" selector was inferior to the best combiner by up to **0.04 AUC**, and simple functions were competitive with learned ones (see next section).

### Musculoskeletal radiographs — heterogeneous ensemble gains +0.01–0.02 AUC

Hussain et al. (2021) built a calibrated ensemble of ConvNet, ResNet, and DenseNet on the MURA musculoskeletal X-ray dataset. The best single model (ResNet) reached AUC **0.92**; a ResNet+DenseNet average reached **0.93**; and a calibrated per-body-part ensemble also reached **0.93** with accuracy rising from 0.86 to **0.87** [[Hussain et al., Sci Rep 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8079683/)].

The gain here is smaller than mammography because the singles are already strong — the same ceiling effect we see on the RSNA board.

### MRNet and knee-MRI competition — no published model ensemble to quote

The original MRNet paper (Bien et al., PLOS Medicine 2018) is **not** an ensemble of models: it is a single CNN per MRI series, with logistic regression combining the three series of one exam. Internal-validation AUCs were **0.937** (abnormality), **0.965** (ACL), and **0.847** (meniscus) [[Bien et al., PLOS Medicine 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/)].

The public MRNet competition leaderboard lists only single-model entries; the top score is **0.917** average AUC [[Stanford ML Group MRNet leaderboard](https://stanfordmlgroup.github.io/competitions/mrnet/)]. So there is no direct published knee-MRI multi-model ensemble gain to compare against.

## Does architecture-family heterogeneity help?

In general, **yes, when errors differ**. Müller et al. found that heterogeneous stacking beat homogeneous bagging on the two hardest datasets. Hussain et al. explicitly contrasted their heterogeneous ConvNet/ResNet/DenseNet ensemble with Jones et al.'s homogeneous ten-ConvNet ensemble and argued the diversity was a key difference.

But heterogeneity is **not a free lunch**:

- On this RSNA competition, a public three-family blend (CoAtNet 384 + Swin-base 384 + EffNetV2-L 480) reported that ensembling was worth only **~+0.001 AUC** on the live board, because the singles were already near ceiling and their errors were correlated.
- Our own six-member DINO-family ensemble shows the same ~+0.001 board behavior (`research/deep_dives/2026-08-31_public-frontier-sweep.md`).
- Self-reported rival single models sit at **0.942**, while our six-member ensemble was **0.939** at the time (later updated to **0.943–0.944**), so the *entire* mix was buying roughly what one better single member would (`research/deep_dives/2026-08-25_competitor-single-model-scores.md`).

So while the *literature* says heterogeneity helps, the *field evidence on this task* says the ceiling is low enough that even a cross-family blend adds almost nothing.

## What about learned combiners?

Müller et al. tested twelve pooling functions for stacking: decision tree, Gaussian process, logistic regression, majority vote (soft/hard), unweighted/weighted mean, naïve Bayes, SVM, k-NN, and "best model" selection. Their finding:

> "Simple statistical pooling functions are equal or often even better than more complex pooling functions like Support Vector Machines or Logistic Regressions." [[Müller et al., IEEE Access 2022](https://arxiv.org/abs/2201.11440)]

The only medical-imaging case where a learned/input-conditioned combiner clearly helped was Hussain et al.'s **calibrated per-region ensemble**: they assigned ResNet to some body parts and DenseNet to others because validation showed genuine regional specialization. The combiner succeeded because the *base models were specialized*, not because the combiner was complex.

For a generic panel of similar DINOv2 recipes, there is no reason to expect a meta-learner to outperform mean voting. The risk is overfitting a small held-out set, especially when the marginal ensemble gain is already ~0.001.

## What this means for the project

1. **Do not treat +0.016 as abnormal.** It is smaller than the mammography +0.03 and the Müller stacking +13% F1, but larger than the RSNA public-board +0.001. The difference is likely the denominator/holdout: a small or heterogeneous holdout can show bigger gains than the live board.
2. **Do not blame the DINO monoculture for all the flatness.** A cross-family blend on the public board is also flat; the ceiling is the ceiling.
3. **Test a combiner for free before spending a card.** Train a logistic regression or ridge on member logits using the 58-study gold set or a held-out fold, and compare it to simple mean and to the best single. If it does not beat mean by more than ~0.001, stop.
4. **If adding heterogeneity, add exactly one non-DINO member.** The public frontier's CoAtNet/Swin/EffNetV2 panel is the only measured heterogeneous signal, and it is still ~0.001. Anything beyond one such member is seating theatre.

## Sources

- [Shen et al., "Deep Learning to Improve Breast Cancer Detection on Screening Mammography", Nature Medicine 2019 / PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6715802/)
- [Müller et al., "An Analysis on Ensemble Learning optimized Medical Image Classification with Deep Convolutional Neural Networks", IEEE Access 2022 / arXiv](https://arxiv.org/abs/2201.11440)
- [Hussain et al., "A calibrated deep learning ensemble for abnormality detection in musculoskeletal radiographs", Sci Rep 2021 / PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8079683/)
- [Bien et al., "Deep-learning-assisted diagnosis for knee magnetic resonance imaging", PLOS Medicine 2018 / PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/)
- [Stanford ML Group MRNet competition leaderboard](https://stanfordmlgroup.github.io/competitions/mrnet/)
- Internal: `research/deep_dives/2026-08-31_public-frontier-sweep.md`
- Internal: `research/deep_dives/2026-08-25_competitor-single-model-scores.md`
