# Q-BR — label denoising when only 58 studies are gold

Question from Claude (2026-09-02): which denoising procedure has a measured delta
in medical imaging with report-derived labels, what can 58 gold studies actually
do, how do we handle the ruler problem, and which methods survive asymmetric
(false-negative-heavy) noise?

## Executive ranking

| rank | procedure | measured delta | data / ruler | needs clean set? | survives asymmetric? | recommendation |
|---|---|---|---|---|---|---|
| 1 | **Use a better report labeler / relabel** | +0.04 weighted AUROC (VisualCheXbert vs CheXbert on CheXpert) | CheXpert; radiologist-adjudicated test set | No | Indirectly — reduces FN-heavy noise at source | **Highest expected return**, but only if a better knee-report labeler exists. |
| 2 | **Out-of-fold (OOF) ensemble relabeling / pseudo-label blending** | Small but consistent gains in medical benchmarks; cleans hard examples without discarding data | Various; clean test set | Optional small set for calibration | Yes, if ensemble is diverse | **Most implementable in a day** on top of our existing members. |
| 3 | **Sample selection (small-loss / GMM / confident learning)** | +2.6 to +7.8 accuracy points on NoisyCXR (40% real noise); +2–10% on ISIC depending on noise | NoisyCXR, ISIC; clean test set | No for ranking; yes for noise-rate calibration | Partially — performance drops under asymmetric noise | Useful, but tends to throw away hard positives. |
| 4 | **Label smoothing / soft targets** | Up to +0.08 AUC on CheXpert (Pham et al.) | CheXpert; radiologist test set | No | Mild mitigation only | Cheap to add, small ceiling. |
| 5 | **Co-teaching / co-teaching+ / small-loss exchange** | +1.6 to +3.2 accuracy points on NoisyCXR; less effective as noise rises | NoisyCXR; clean test set | No | Designed for symmetric noise; weakens under class-conditional noise | Not first choice. |
| 6 | **Confident Learning (cleanlab) with transition-matrix correction** | Strong on ImageNet; limited published medical deltas | Mostly natural images | Yes — needs enough samples per class to estimate transition matrix | Yes in principle, but estimation is hard with rare positives | 58 studies is almost certainly too few. |

## 1. Which denoising procedures have a measured delta in medical imaging?

### 1.1 Better report labeler (Jain et al. 2021, NeurIPS DCAI)

The strongest medical-imaging delta comes from improving the *source* of the
labels rather than post-hoc denoising. Jain et al. trained DenseNet-121 on
CheXpert using labels from three report labelers:

- CheXpert labeler
- CheXbert
- VisualCheXbert

On the radiologist-adjudicated CheXpert test set, the model trained on
VisualCheXbert labels achieved **0.87 weighted AUROC** vs **0.83** for the next
best — a **+0.04 delta** purely from labeler quality. This is the closest analog
to our situation: report-derived labels, twelve binary findings, large training
set, small adjudicated gold set.

The catch: the delta exists only if a better labeler exists for our knee MRI
reports. We currently have one LLM-derived label set; a second, stronger labeler
would be the cheapest way to realize this gain.

### 1.2 Sample selection and loss-based filtering

A 2024 medical-image sample-selection paper (Zhu et al., *Sample selection with
noise rate estimation in noise learning of medical image analysis*) reports the
following on **NoisyCXR**, a real-world noisy chest-X-ray dataset with roughly
**40% label noise** derived from NIH report labels vs RSNA radiologist labels:

| method | accuracy on noisy NoisyCXR |
|---|---|
| Baseline | 59.15% |
| MixUp | 61.98% |
| O2U-Net | 61.76% |
| Co-teaching+ | 62.34% |
| Their sample-selection + GCE + sparse reg | **66.90%** |

That is a **+7.8 point accuracy gain** over baseline and **+4.6 over the best
classic method** on real, report-derived noise.

Xue et al. (2019, ISBI) reported **+2–10% accuracy** on ISIC skin-lesion data
using uncertainty-based sample mining + re-weighting, depending on noise rate.

These methods rank training samples by loss (or predictive uncertainty) and
remove or down-weight the highest-loss samples. The risk is that hard positives
— exactly the minority class we care about — can be mistaken for noisy samples
and discarded.

### 1.3 Co-teaching and variants

Co-teaching (Han et al. 2018) trains two networks and has each network feed its
small-loss samples to the other. A Nature 2023 medical variant (*Stochastic
co-teaching for training neural networks with unknown levels of label noise*)
showed gains on medical data, but co-teaching is designed for symmetric noise
and weakens under class-conditional/asymmetric noise. On NoisyCXR it reached
only **62.34%** accuracy, well below sample-selection methods.

### 1.4 Label smoothing / soft targets

Pham et al. used label smoothing for thoracic-disease classification on
CheXpert and reported **up to +0.08 AUC** improvement. This is a cheap
mitigation, not a denoising procedure, and the ceiling is low.

### 1.5 Confident Learning / cleanlab

Confident Learning (Northcutt et al., JAIR 2021) estimates the joint
distribution of noisy and true labels and prunes likely errors. It is powerful
on ImageNet but published medical deltas are scarce. The key requirement is
enough data per class to estimate the class-conditional noise transition matrix
reliably. With 12 labels and only 58 gold studies, this is almost certainly
infeasible.

## 2. What can 58 gold studies do?

58 studies is too small to train on, but it can serve three narrower roles:

### 2a. Estimate per-label noise rate

Probably **not reliably**. Estimating a 12×12 noise transition matrix (or even
per-label false-negative / false-positive rates) from 58 studies gives roughly
five samples per class. The variance on rare positives will be enormous.
Sample-selection methods that only need a ranking of samples by loss can work
without a clean set; methods that need an absolute noise rate (CL, O2U-Net with
fixed rate) need more gold data.

### 2b. Validate that relabeling moved toward truth

**Yes, weakly.** You can compare the original report-derived labels and the
relabeled labels against the 58 gold labels. If relabeling increases agreement
with gold, you have evidence the procedure helped. The confidence interval will
be wide, but the direction is informative.

### 2c. Calibrate an ensemble-based cleaner

A small clean set can be used to choose the blending weight between original
labels and OOF-ensemble pseudo-labels (e.g. by grid search on the 58 studies).
This is the most practical use.

## 3. The ruler problem

Our 909-study holdout uses the same noisy report-derived labels as the training
set. A genuine denoising improvement may therefore look like a regression on
holdout while helping on the competition board.

Published works avoid this by evaluating on a **clean test set** (radiologist
adjudicated, as in CheXpert and NoisyCXR). When only a noisy ruler exists, the
honest protocol is:

- Report **two numbers**: performance on the noisy holdout *and* performance on
  the 58-study clean subset.
- Use the clean subset as the **primary decision ruler** for whether to keep the
  denoising procedure.
- Accept that the 58-study interval will be wide; use it for direction, not for
  precise ranking.

Some papers (e.g. Karimi et al. 2020) also report the drop between best-epoch
and last-epoch performance as a robustness signal.

## 4. Asymmetric, class-conditional noise

Report-derived labels are systematically false-negative heavy: a finding not
mentioned is labeled negative whether it was absent or unreported. This is
class-conditional, not symmetric.

- **Symmetric-noise methods** (basic co-teaching, small-loss selection with a
  single threshold) perform worse under asymmetric noise.
- **Asymmetric-aware methods** include:
  - Generalized Cross Entropy (GCE) / MAE-family losses.
  - Dual-uncertainty estimation (Wang et al. 2021), which separately models
    annotator disagreement and single-target label noise.
  - Class-aware sample selection or per-class loss thresholds.
- **OOF ensemble relabeling** is implicitly asymmetric-aware if the ensemble is
  trained on the same label distribution, because it preserves the base rate and
  only flips labels where the model is confident against the noisy label.

## 5. Practical recommendation

Given 58 gold studies and a 4407-study training set:

1. **Try OOF ensemble relabeling first.** Use our existing members to generate
   per-study pseudo-labels. For each sample, blend the original label with the
   ensemble prediction when the ensemble is confident (e.g. average probability
   > 0.7 or < 0.3) and the original label disagrees. Use the 58 gold studies to
   choose the confidence threshold and blending weight.
2. **Add label smoothing** as a cheap baseline (+0.08 AUC ceiling on CheXpert).
3. **If a better report labeler is available** (stronger LLM, or a rule-based
   post-processor for knee reports), regenerate labels. This has the largest
   published delta (+0.04 AUROC on CheXpert) and costs no card time.
4. **Avoid confident-learning transition-matrix methods** with only 58 gold
   studies; the matrix cannot be estimated reliably.
5. **Report on both rulers**: noisy 909-study holdout and 58-study clean subset.
   Make the clean subset the decision gate.

## Sources

- Jain et al., *Effect of Radiology Report Labeler Quality on Deep Learning
  Models for Chest X-Ray Interpretation*, NeurIPS DCAI 2021.
  https://arxiv.org/pdf/2104.00793.pdf
- Zhu et al., *Sample selection with noise rate estimation in noise learning of
  medical image analysis*, 2024. https://arxiv.org/abs/2312.15233
- Xue et al., *Robust learning at noisy labeled medical images*, ISBI 2019.
- Wang et al., *Improving Medical Image Classification with Label Noise Using
  Dual-uncertainty Estimation*, 2021.
  https://arxiv.org/abs/2103.00528
- Karimi et al., *Deep learning with noisy labels: exploring techniques and
  remedies in medical image analysis*, MedIA 2020.
  https://arxiv.org/abs/1912.02911
- Northcutt et al., *Confident Learning: Estimating Uncertainty in Dataset
  Labels*, JAIR 2021. https://www.jair.org/index.php/jair/article/view/12125
- Pham et al., *Interpreting chest x-rays via CNNs that exploit disease
  dependencies and uncertainty labels*, 2019.
