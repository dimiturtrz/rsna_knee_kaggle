# RSNA 2026 Knee Abnormality Detection — public landscape and 3D evidence

2026-09-01, Kimi. Snapshot of the current RSNA challenge, what the public leaderboard/models say, and how the modern multi-plane/3D literature (MRNet → CoPAS) fits the competition task.

## The competition

The [RSNA 2026 Knee Abnormality Detection AI Challenge](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge) is a study-level multi-label binary classification problem over 12 knee abnormalities (ACL, MCL, medial/lateral meniscus, medial/lateral/PF OA, effusion, synovitis, Baker's cyst, contusion, fracture). It is hosted on Kaggle as a code competition: submissions are notebooks, no internet at inference, and the GPU time budget is 9 h for the full ~1,300-study test set. The dataset contains >5,000 exams from 19 sites and associated radiology reports in multiple languages.

## The label problem

Only a small subset of training studies carry expert-derived gold labels; the remainder have report-derived labels. An independent community audit (referenced in the public repo `konradcinkusz/rsna-knee-abnormalities-efficiency-lb`) estimates **58 gold-labelled studies out of 4,407**, with report-derived labels disagreeing with gold labels in ~18% of cases. This makes the task weakly supervised and likely explains why very large backbones overfit.

## What the public Kaggle Models tab shows

The Kaggle competition [Models tab](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/models) reports real best public scores achieved by the community:

| Model | Best public score | Notes |
|---|---|---|
| DINOv2-small | **0.906** | best single public score; 40 users |
| EfficientNet-B3 | 0.701 | community checkpoint |
| dinov2-with-registers-base | 0.696 | community variant |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.656 | text encoder only |
| DINOv2-large | **0.500** | one user; effectively random, suggesting overfitting/bad fine-tuning on the tiny gold set |

Practical signal: **capacity does not buy performance here** — DINOv2-small beats base and large in the public results. The dominant effect is likely the small gold-label set and the weak-supervision setup, not backbone expressiveness.

## Public architecture direction

The public repo [`konradcinkusz/rsna-knee-abnormalities-efficiency-lb`](https://github.com/konradcinkusz/rsna-knee-abnormalities-efficiency-lb) documents a **2.5D DINOv2 vision branch + multilingual text branch** architecture. Because `test.csv` does not contain the `Report` column, the text branch can only be used during training (e.g., as auxiliary supervision or distillation); inference is vision-only. This is the same design the repo's `model.py` sketch proposes.

## Modern 3D/multi-plane evidence: CoPAS

[Qiu et al., *Nature Communications* 2024](https://doi.org/10.1038/s41467-024-51888-4) present **CoPAS**, a multi-sequence, multi-plane model for 12 knee abnormalities on a 1,748-patient clinical dataset. It uses:

- one 3D ResNet-18 branch per anatomical plane, with weight sharing;
- cross-plane attention to let each plane borrow spatial information from the orthogonal planes;
- co-plane cross-sequence attention between PDW and T1W/T2W contrasts;
- a plane-aware probability matrix that fuses branch predictions while learning abnormality-plane correlations.

On their internal test set CoPAS reached mean AUC-ROC 0.812, outperforming MRNet in 8 of 12 classes and matching or beating junior radiologists. Generalization to external datasets dropped to ~0.72, driven by missing sequences and scanner differences.

Key takeaway for this competition: **attention across planes and sequences can help**, but the gain depends on having the same sequences available at test time and on enough data. CoPAS also confirms that multi-task learning helps rare classes, while single-task MRNet can still win on a few classes.

## Implications for this project

1. **Backbone size**: public evidence says DINOv2-small is the strongest starting point in this challenge's public results; large models score worse, likely due to the tiny gold set. The project's DINOv2-large/v3-large arms should be treated as diversity bets, not as obviously stronger solo members.
2. **Text branch**: the report text is train-only, so any text model must distil into the vision branch. A multilingual MiniLM encoder is the public default.
3. **3D path**: a true 3D convolutional trunk is expensive under the 9 h inference limit and has no public score advantage in this challenge. CoPAS-style cross-plane attention is a more targeted 3D-ish idea, but it needs multi-plane/multi-sequence inputs that may not be consistently available.
4. **Weak supervision**: the 58-vs-4,407 label split is the dominant data fact; improvements in label quality, calibration, and report-to-label mapping likely dominate pure architecture changes.

## Sources

- RSNA challenge page: https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge
- Kaggle Models tab: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/models
- Public 2.5D + text repo: https://github.com/konradcinkusz/rsna-knee-abnormalities-efficiency-lb
- Qiu et al., CoPAS, *Nat Commun* 2024: https://doi.org/10.1038/s41467-024-51888-4
