# Q-AO — are the competition's test labels report-derived or radiologist-reviewed?

Question (Claude, 2026-09-02): determine whether the hidden test ground truth for the RSNA Knee
Abnormality Detection competition is derived from radiology reports (like most training labels) or
independently reviewed by radiologists.

## Bottom line

The hidden test labels are **most likely radiologist-adjudicated**, while the bulk of the training
labels are report-derived. This makes the vision-verification / label-correction axis a live lever
rather than an own goal.

## Evidence

### 1. What the competition says about training labels

RSNA's own challenge page ([RSNA Knee MRI AI Challenge](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge),
read 2026-09-02):

> "The challenge is the first to use both images and text from radiology reports to train AI models."

Kaggle data tab snippet ([RSNA Knee Abnormality Detection / Data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)):

> "We also provide the original text of the radiology report from which you may wish to derive the
> labels for the remaining studies."

This confirms that only the 58 publicly annotated studies are radiologist-labeled gold; the other
4,349 training studies get their labels from competitor-derived report extraction.

### 2. Direct evidence that the labels are "clinically adjudicated"

Kaggle discussion 733343 ([link](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343)):

> "The combination of multi-label MRI classification, multilingual reports, and **clinically
> adjudicated labels** makes this a ..."

The phrase is used to describe the challenge as a whole, not only the 58 gold training studies. It is
a community post, not an official host reply, but it is the closest thing to a direct statement found.

Kaggle discussion 733836 ([link](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733836)):

> "There's no proper ground truth 'label' here. A radiologist reads an MRI and writes a report, but
> scans are hard to read and two radiologists can disagree."

This reads as a description of how labels originate in practice, and it is consistent with
adjudication being needed to settle disagreements.

### 3. RSNA's institutional pattern

RSNA "Data for AI" page ([link](https://www.rsna.org/artificial-intelligence/data)):

> "RSNA curates ground-truth datasets that are diverse, fully anonymized and **triple-labeled by
> radiologists**—ensuring transparency and trust in every pixel."

Kitamura et al. 2024, *Institution Datasets and Hosting the RSNA AI Challenges*
([Radiology: Artificial Intelligence](https://pubs.rsna.org/doi/pdf/10.1148/ryai.230227)):

> "For example, in the bone age challenge, test set labels were averaged over six different
> radiologist labelers, making the ground truth a continuous ..."

The institutional norm is that RSNA challenge **test sets are radiologist-labeled/adjudicated**, not
report-extracted.

### 4. Why the ambiguity persists

No official page or host reply was found that states explicitly "the hidden test labels were reviewed
by radiologists." The Kaggle data description only speaks about deriving labels for the remaining
**training** studies. The "clinically adjudicated labels" phrase comes from a discussion post.

## What this means for the label-correction axis

If the test labels are radiologist-adjudicated, then correcting training labels toward the image
(radiologist-consensus truth) is aligned with the scored quantity. The measured 15.8% confident false
positives and 65.3% report→radiologist agreement on the 58 gold studies become a real lever, because
the hidden test follows the same radiologist-consensus standard.

If the test labels were report-extracted, the axis would be an own goal. The preponderance of evidence
points to adjudication, but it is not a verbatim quote.

## Recommended next step

Because the answer is not 100% explicit, design an experiment that is robust to both possibilities:

1. Train a second head that predicts "label is wrong / image contradicts report" on the 58 gold
   studies plus any other studies where image-report disagreement can be measured.
2. Do not overwrite the report-derived targets; instead, use the disagreement signal as an auxiliary
   loss or a sample weight.
3. Measure whether down-weighting high-disagreement studies helps or hurts the holdout. If it helps,
   the test labels are likely image-aligned (adjudicated). If it hurts, the test labels are likely
   report-aligned.

This costs no new labels and no board slots until the direction is proven.
