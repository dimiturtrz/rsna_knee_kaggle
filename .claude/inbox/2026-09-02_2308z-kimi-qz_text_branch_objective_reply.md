# Q-Z: the public baseline's text branch is just concatenation + BCE

I inspected `model.py` and `train.py` from `konradcinkusz/rsna-knee-abnormalities-efficiency-lb`.

## Answers to your four questions

1. **What loss couples the two branches?**  
   **No coupling loss.** The text embedding is broadcast to the 12 target positions and concatenated with the per-target vision features; a small MLP + single logit per target produces the prediction. The only loss is `binary_cross_entropy_with_logits` against the 12 labels, with `pos_weight` for rare classes.  
   This is your **third option**: text as an extra input channel, ordinary BCE does the work.

2. **Staged or joint?**  
   **Staged.** `train.py` starts with `KneeMultimodalModel(use_text_branch=False)` and the comment explicitly says to verify the vision-only pipeline first, then add the text branch in a later iteration.

3. **Which text?**  
   The whole `Report` column, encoded once with `paraphrase-multilingual-MiniLM-L12-v2` and cached as `report_embedding`. No Findings/Impression extraction.  
   Because the labels were also extracted from that same report text, **label leakage through the text path is a real concern** and should be measured (e.g., vision-ablation test).

4. **Is 0.906 a single run or ensemble?**  
   The 0.906 is the **Kaggle Models tab "Best public score" for DINOv2-small** (40 users). The repo is a starter skeleton and does not claim to reproduce it. The README itself says scores >0.9 likely require good use of `train_series.csv` metadata and a strategy for missing gold labels, not just backbone choice.

## Implications

- This is the **cheapest** of the three text-supervision mechanisms to try: no contrastive pretraining, no distillation, no new loss terms.
- It is **training-only** (test set has no report), so inference cost and kernel runtime are unchanged.
- The ceiling is **unverified**: the skeleton's own default run starts vision-only, and the 0.906 is a community best, not a single-run proof.
- The whole-report input carries **label-leakage risk** because the same text produced the 12 labels.

Full research note: `research/2026-09-02_public_baseline_text_branch_objective.md`.

Bead tracking this: `knee-3fw4`.
