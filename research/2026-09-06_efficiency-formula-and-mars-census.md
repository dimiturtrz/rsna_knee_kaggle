# 2026-09-06 — Efficiency Prize formula (official page, via API) and MARS census

## Efficiency Prize Evaluation — quoted from the competition's own overview page

Source: Kaggle `competition_list_pages('rsna-knee-abnormality-detection')` → page
"Efficiency Prize Evaluation" (no browser needed; full text archived at
`.agent-runtime/scratch/efficiency_prize_page.txt`). Key facts:

    Efficiency = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400    [minimize]

- Eligible: among the team's Leaderboard-Prize selections (two finals or auto-selected)
  AND private rank above the `sample_submission.csv` benchmark; a submission may win both.
- Benchmark ≈ 0.5 (constant prediction); maxAUC = best private AUC over ALL submissions
  (endogenous, settles at close).
- RuntimeSeconds = total evaluation wall-clock; 32400 s = 9 h = kernel cap.
- Exchange rate at Benchmark 0.5 / maxAUC 0.95: 1 h runtime ≈ 0.05 AUC-equivalent →
  runtime dominates the 0.0045 accuracy-visibility budget several-fold; the
  resolution entry (same members at 224 px) is the shape the formula rewards.
- Public efficiency LB is **rank-only by design** ("will show only the rank ... not the
  complete score") and **updates daily** → cheapwide_v1's rank appears at the next
  daily refresh; version history is not API-exposed.

## MARS (github.com/zqiuak/MARS) census — for the 3D tubelet seat price

- MRI foundation model (not CT): 3D Swin (MONAI SwinUNETR-style), SSL on 336,476
  multi-sequence MRI volumes / 64 datasets; **knee MRI (OAI, fastMRI Knee, private) is
  a named pretraining class**.
- Weights: single `MARS.pt`, 78.3 MiB, Google Drive (README link). Encoder-keyed
  `swinViT.*`; loader `main/get_vit_weight.py`.
- Licence: **Apache-2.0** repo LICENSE, §4 permits redistribution of Work and
  Derivative Works, no non-commercial clause; weights ride repo terms (inference, not
  an explicit weight licence).
- Input contract: **1.0 mm isotropic, 96³** (configurable per task), per-volume
  intensity norm (no fixed window), orientation-agnostic. **Hard 3D only** (5-D input,
  dims divisible by 32; no native 2D/thin-stack mode — thin stacks need zero-padding).
- Released config: patch/tubelet 2×2×2, embed 48, depths (2,2,2,2), window 7³, latent 768.
- No Kaggle mirror found → upload needed (trivial at 78 MiB; ship LICENSE with payload).
- Pricing caveat: the 1 mm isotropic contract destroys native thin-slice anisotropy —
  seating it means resampling away from the geometry our 2.5D stack reads natively.
