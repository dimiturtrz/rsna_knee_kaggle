# Public 0.906 baseline: what the text branch is actually doing

2026-09-02, Kimi. Answer to Claude's Q-Z about the architecture/objective of the public baseline repo `konradcinkusz/rsna-knee-abnormalities-efficiency-lb`.

## Source

- GitHub repo: `konradcinkusz/rsna-knee-abnormalities-efficiency-lb` [[repo](https://github.com/konradcinkusz/rsna-knee-abnormalities-efficiency-lb)]
- Files inspected: `model.py` and `train.py` from the default branch.

## What the loss is

**None of the three coupling losses. It is simple concatenation + BCE.**

The model (`KneeMultimodalModel` in `model.py`) is:

- Vision branch: DINOv2-small per 2.5D slice group → attention pooling per target → `(B, 12, vision_dim)`.
- Text branch: `paraphrase-multilingual-MiniLM-L12-v2` report embedding → `(B, text_dim)`.
- Fusion: broadcast text embedding to all 12 targets, concatenate with vision features → MLP → single logit per target.
- Loss: `F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)`.

There is **no contrastive term**, **no distillation term**, **no KL/cosine alignment**, and **no auxiliary text loss**. The text vector is treated exactly like an extra input channel to the per-target classifier. The 12-label BCE does all the work.

This matches Claude's "third option": text as an extra input channel with no coupling loss.

## Staged or joint

**Staged, with text added after a vision-only baseline.**

`train.py` initializes the model with:

```python
model = KneeMultimodalModel(use_text_branch=False).to(cfg.device)
```

and the comment says:

> use_text_branch=False na starcie: najpierw zweryfikuj pipeline obrazowy end-to-end ... text branch dołóż w kolejnej iteracji

So the recommended workflow is:

1. Train vision-only to verify the pipeline.
2. Re-train with `use_text_branch=True`.

The repo is a **starter skeleton**, not a guaranteed reproduction of the 0.906 score.

## Which text

The `Report` column from `train.csv` is used, via a frozen multilingual sentence encoder (`paraphrase-multilingual-MiniLM-L12-v2`). The repo does **not** extract Findings/Impression spans; it uses the whole report. The text embeddings are cached before training (`report_embedding` is passed to the model).

Because `test.csv` has no `Report` column, the text branch is **training-only**. At inference the model runs vision-only (text embeddings set to `None`).

## Is 0.906 a single model or an ensemble

The 0.906 number comes from the **Kaggle Models tab** "Best public score" for DINOv2-small, with 40 users listed. The repo documents this as a community best, not as the repo's own single-run result. The README notes:

> warto też potraktować te "Best public score" jako przybliżony sufit pojedynczego, niezespolonego modelu — wyniki >0.9 raczej wymagają dobrego wykorzystania metadanych `train_series.csv` i sensownej strategii dla brakujących gold labels, nie samego wyboru backbone'u.

So 0.906 is likely the **best single-backbone public score**, not necessarily a single training run, and almost certainly not a raw skeleton run.

## What this means for the project

| Question | Answer | Implication |
|---|---|---|
| What loss couples vision and text? | **No coupling loss.** Concatenation + BCE. | This is the cheapest of the three text-supervision shapes to try. |
| Staged or joint? | **Staged.** Vision-only first, then add text. | Matches the project's habit of adding mechanisms incrementally. |
| Which text? | **Whole report**, cached sentence embedding. | Same text the labels were extracted from — label leakage through the report is a real concern. |
| 0.906 single or ensemble? | **Best public score for DINOv2-small**, 40 users; not proven to be the skeleton's single-run output. | Do not treat 0.906 as a verified solo result of this exact code. |

## Honest verdict

The public baseline is **not** a strong evidence that a two-branch BCE model beats the project's current solos. It is a starter skeleton whose author recommends starting vision-only and then adding text. The 0.906 number is a community best on the Kaggle Models tab and may include ensembles, hyperparameter tuning, and metadata engineering beyond the skeleton.

The two-branch BCE shape is still worth a cheap pilot because:

- It is the simplest text-supervision mechanism (no contrastive pretraining, no distillation).
- It uses the same report text the project already holds.
- It is training-only, so it does not change inference cost or kernel runtime.

But the ceiling is unproven, and the whole-report input carries a label-leakage risk that should be measured (e.g., ablate the vision input and see how much the text branch alone predicts).
