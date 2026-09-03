# Public baseline audit — RSNA Knee Abnormality Detection 2026

**Source:** `konradcinkusz/rsna-knee-abnormalities-efficiency-lb` on GitHub — a public solution repo with verified competition docs and a DINOv2 + multilingual text-branch skeleton.  
**Why it matters:** it is the *same* competition and confirms several things we had only inferred.

## Competition facts (verified against official Kaggle Data/Rules)

- **Task:** study-level multi-label binary classification over 12 knee findings:  
  `ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`.
- **Metric:** macro-averaged ROC AUC — rare labels count as much as common ones.
- **Data:** 4407 training studies, **only 58 have radiologist-adjudicated "gold" labels**; the other 4349 have report-derived labels that agree with gold only ~82 % of the time (systematic, not random error).
- **Test set:** ~1300 studies, **no report column** — any text model can only be used during training, never at inference.
- **Volume:** 819 640 DICOM files, ~570 GB.
- **Format:** code competition, 9 h GPU, no internet at submit, external models allowed only if freely/publicly available.
- **Regulation note:** sending report text to hosted LLM APIs is likely a Data-Security rule violation; text must be processed locally or inside the Kaggle notebook.

## What actually scores on the public leaderboard

Backbone "Best public score" from the Kaggle Models tab (single-model, not ensemble):

| Backbone | Public macro AUC | Note |
|---|---:|---|
| **DINOv2-small** | **0.906** | best public single model (source: public baseline repo) |
| DINOv2-base | 0.861 | — |
| EfficientNet-B3 | 0.701 | — |
| dinov2-with-registers-base | 0.696 | — |
| paraphrase-multilingual-MiniLM-L12-v2 | 0.656 | text-only baseline |
| EfficientNet-B0 | 0.622 | — |
| **DINOv2-large** | **0.500** | degenerate value — likely collapsed head, constant prediction, or label misalignment; not confirmed as overfitting |

Key takeaway: **bigger vision backbone is not better**.  DINOv2-small wins; DINOv2-large fails.  The 0.500 value is too clean to be a simple capacity story and should be verified before interpreting it as "overfits".

## Winning architecture template (from the repo's `model.py`)

- **Vision branch:** 2.5D DINOv2-small (`dinov2_vits14`, 384-dim).
  - Stack a few neighbouring slices as pseudo-RGB (B, 3, 224, 224).
  - Encode every slice of every sequence with the same DINOv2-small.
  - **Attention pooling per target** over all slice embeddings, producing a separate aggregated vector per class (so ACL and Effusion can attend to different slices).
- **Text branch (training only):** frozen/lightly-finetuned `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` on the report.
  - Used for extra signal / distillation on the 4349 non-gold studies.
  - Dropped at inference because test has no report.
- **Fusion + heads:** concatenate per-target vision vector with broadcast text vector, small MLP, then 12 independent sigmoid heads.
- **Loss:** `binary_cross_entropy_with_logits` with per-target `pos_weight` for class imbalance.

## What this means for our pipeline

1. **DINOv2-small should stay the vision trunk.**  Do not swap it for a larger DINOv2 or a generic VLM expecting automatic gains.
2. **A local text/vision model is useful, but only for training.**  It can:
   - Re-read reports to generate or clean weak labels for the 4349 non-gold studies.
   - Act as a teacher for distillation into the image-only model.
   - Sanity-check image-label consistency.
   It **cannot** be used at Kaggle test inference.
3. **3D CNN is unproven here.**  The public evidence says 2.5D + attention pooling is the practical winner; full 3D is compute-heavy and lacks a public score advantage.
4. **Efficiency track matters.**  Inference must fit in 9 h on Kaggle; oversized models or per-slice heavy VLM calls will fail the efficiency cut even if they scored well locally.
5. **Report privacy is a real constraint.**  Any bulk report processing must stay local (Qwen3.8-27B NVFP4 on the 5090, or a Kaggle-offline model), not external API.

## Open questions to verify against our own data

- Do our 58 gold studies line up with the repo's audit?  We should count non-empty labels in `train.csv` ourselves.
- Does DINOv2-small also win on our private validation?  Public score 0.906 is a strong prior, but our local OOF on the 58 gold is the signal to trust.
- Can a local VLM (Qwen3.8-27B or MedGemma-1.5-4B) improve the report-to-label mapping for the 4349 weakly-labeled studies compared with the current extractor?

## References

- Public repo + docs: [konradcinkusz/rsna-knee-abnormalities-efficiency-lb](https://github.com/konradcinkusz/rsna-knee-abnormalities-efficiency-lb)
- Skeleton `model.py` (2.5D DINOv2 + text branch): same repo, `model.py`
- RSNA announcement: [RSNA Knee Abnormality Detection AI Challenge](https://www.rsna.org/news/2026/august/ai-challenge-knee-mri)
