# Q-CI — What an LP-FT schedule has to be, concretely enough to arm

**Question (Claude, 2026-09-03):** For the `ortholarge` / `radnet_family` continued trunks, what is the concrete LP-FT recipe? Specifically:
1. LP phase stopping rule — convergence on what, and sensitivity to under-training the probe.
2. Trunk/head learning-rate ratio, and whether the head stays frozen once FT starts.
3. Any precedent for LP-FT when features are pooled from a volume / multiple slices.
4. Cheapest early signal that distinguishes "LP-FT will help" from "this trunk is just worse".

**TL;DR:** LP-FT is a head-initialization trick, not a separate architecture. The published recipe is: train a linear head on frozen features until convergence (or for a fixed epoch budget), then fine-tune the **whole** network with the head initialized from that solution. The head is **not frozen** during FT; the gain comes from the initialization, not from a special LR ratio. For our slice-pooled setting there is **no published precedent** — the pooling layer is best treated as part of the head, but this is our call to measure. The cheapest discriminator is a combination of **high frozen-LP accuracy + low one-epoch full-FT accuracy + large one-epoch feature drift**; if all three hold, LP-FT is a sensible bet.

---

## 1. LP phase stopping rule

### What Kumar et al. actually do

In **Kumar et al., “Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution” (ICLR 2022)** [arXiv:2202.10054](https://arxiv.org/abs/2202.10054), the LP-FT protocol is:

- **Phase 1 (LP):** Freeze the pretrained trunk, replace the head with a randomly initialized linear layer, and train the head on the target task.
- **Phase 2 (FT):** Initialize the head with the Phase-1 solution and fine-tune **all** parameters.

For the small/medium datasets (Living-17, Entity-30, DomainNet, etc.) the LP phase is run for the same epoch budget as FT (e.g., 20 epochs for Living-17 in their config) and the best model is chosen by **ID validation accuracy** (early stopping). For ImageNet, to keep total compute equal to vanilla FT, they run **5 epochs of LP + 5 epochs of FT** instead of 10 epochs of vanilla FT ([OpenReview version](https://openreview.net/forum?id=UYneFzXSJWh)).

The paper does **not** report an ablation on how under-training the LP head affects the final FT result. The theory only requires that the head be “good enough” (close to the optimal linear head on the pretrained features); it does not quantify sensitivity.

### What follow-up work uses

- **Tian et al. (2023)** use “LP for 25 epochs, then FT for another 25 epochs” ([arXiv:2303.10720](https://arxiv.org/abs/2303.10720)).
- **Asymmetric Duos** (Caltech-256 / iWildCam) use 5–8 LP epochs followed by 10–20 FT epochs ([arXiv:2505.18636](https://arxiv.org/abs/2505.18636)).
- The original code repo sweeps LP learning rates `[3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]` and runs the same `epochs` as FT ([AnanyaKumar/transfer_learning](https://github.com/AnanyaKumar/transfer_learning), `scripts/run_adaptation_experiments.py`).

### Practical recommendation for us

Because our trunk features are already cached to `.npy`, the LP phase is nearly free. We should:

1. Train the linear/pooling head on frozen cached features until **ID validation loss plateaus** (with a small patience, e.g., 2–3 epochs), OR for a conservative fixed budget of **5 epochs**.
2. Save the head checkpoint.
3. Use that checkpoint as the FT initialization and run the normal 12-epoch recipe.

A 5-epoch LP is the safest first arm because it matches the ImageNet-style budget split and costs minutes, not card hours.

---

## 2. Rate ratio and whether the head is frozen during FT

### Head initialization, not head freezing

In the original LP-FT implementation, the head is **initialized** from the LP solution and then **updated** during FT. The code uses a single optimizer over all parameters; there is no separate head LR multiplier by default ([AnanyaKumar/transfer_learning](https://github.com/AnanyaKumar/transfer_learning), `unlabeled_extrapolation/baseline_train.py` lines 335–355). The benefit comes from starting FT with a head that already aligns with the pretrained features, which reduces how much the trunk has to move to accommodate a random head.

### Learning rates

- **LP phase:** much higher rates are typical. The Kumar code sweeps `[3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]` for the head when using SGD.
- **FT phase:** the same code sweeps `[3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]` for the full network (SGD, cosine schedule, batch size 64).

So the published recipe is **not** a discriminative-learning-rate recipe. However, **layer-wise / discriminative fine-tuning** (Howard & Ruder, 2018) is explicitly mentioned by Kumar et al. as a related idea, and **Surgical Fine-Tuning** (Lee et al., ICLR 2023) shows that freezing or differentially updating the right layers can outperform full FT.

### Proposal for our arm

Use our existing FT recipe but initialize the head from the 5-epoch LP checkpoint, and keep the head **trainable**. Optionally add a small discriminative-LR safeguard:

- Trunk LR: our current `4e-5` (or lower, `1e-5` for the first epoch).
- Head LR: `4e-4` (10× trunk) for the first 1–2 epochs, then unified `4e-5`.

This is not in the original LP-FT paper, but it is a conservative way to avoid destroying the LP head immediately. The core experimental variable should be **head initialization vs random head**, not the LR schedule.

---

## 3. LP-FT with pooled / volume features

### No direct precedent

We did not find any published LP-FT study where the input is a set of slice features pooled per study (deep-set / multi-instance / MIL). All LP-FT results cited above are on single 2D images. The medical-imaging LP-FT literature (e.g., Davila et al., “Comparison of fine-tuning strategies for transfer learning in medical image classification”, 2024) treats the model as a standard 2D CNN and reports that LP-FT helps, but it does not address slice aggregation.

### Where the pooling layer belongs

Our current architecture extracts slice-level features from the trunk and then pools them (`cls_mean_focal_max`, etc.) into a study-level vector before the final classifier. Conceptually:

- If the pooling is **fixed and parameter-free** (mean / max / focal-weighted mean with frozen weights), the pooling belongs to the **trunk side** — it is part of feature extraction.
- If the pooling has **learnable parameters** (e.g., attention weights over slices), those parameters should be trained in the **LP phase** and treated as part of the head.

Since our pooling is likely parameter-free or has very few parameters, the cleanest mental split is:

- **Trunk:** DINOv2/DINOv3 encoder + fixed per-slice pooling → produces a study vector.
- **Head:** linear classifier on the study vector.

That makes LP-FT straightforward: the LP phase trains only the final linear layer on cached study vectors. If we ever move to learned slice attention, that attention module must be included in the LP phase.

**Bottom line:** this is an unmeasured choice, not a settled one. We should arm one variant where pooling is treated as trunk-side and one where any learnable pooling is trained during LP, but the simpler “LP = final linear layer only” variant is the right first arm.

---

## 4. Cheapest early discriminator

Claude wants to avoid spending 3.5 card-hours to learn that an arm lands at 0.85. The literature gives three cheap signals that can be combined:

1. **Frozen LP accuracy is high but one-epoch FT accuracy is lower than expected.** This is the direct signature of feature distortion: the features are good enough to separate the task linearly, but full FT has already started to destroy them. In Kumar et al., this is exactly the regime where LP-FT wins.
2. **Feature drift after one epoch of FT.** Kumar et al. measure `||g_B(x) − g_B0(x)||_2` and find FT changes features 10–100× more than LP-FT. We can compute this on a small validation set after the first FT epoch. Large drift → distortion → LP-FT candidate. Small drift → the trunk is either already fine or the features are too poor for LP to matter.
3. **Relative gradient norm per layer (Auto-RGN) after one epoch.** Lee et al. (Surgical Fine-Tuning, ICLR 2023) show that layers with large relative gradient norms (`||g||_2 / ||θ||_2`) are the ones driving the distribution shift. If the early trunk layers show large RGN while the head also does, the model is distorting the whole representation; LP-FT can mitigate this by starting from a better head. ([arXiv:2210.11466](https://arxiv.org/abs/2210.11466))

### Practical one-epoch screen

After epoch 1 of a candidate FT run, compute on the holdout set:

- `frozen_LP_AUC`
- `epoch1_FT_AUC`
- `mean_feature_drift = E[||trunk_out_epoch1 − trunk_out_pretrained||_2]`
- `early_layer_RGN` (average over the first 2–3 transformer blocks)

If `frozen_LP_AUC` is within ~0.01 of the best LP trunk **and** `epoch1_FT_AUC` is not clearly above the LP AUC **and** feature drift > some baseline, schedule an LP-FT run for that trunk. Otherwise, the trunk is either already fine (small drift) or not worth saving (low LP).

This screen costs one epoch (~1065 s) per candidate, not a full 12-epoch run.

---

## 5. Licence lines

| Work / code | Licence | Notes |
|-------------|---------|-------|
| **LP-FT** ([AnanyaKumar/transfer_learning](https://github.com/AnanyaKumar/transfer_learning)) | **No explicit licence** | No `LICENSE` file; treat as read-only reference only. |
| **Surgical Fine-Tuning** ([anniesch/surgical-finetuning](https://github.com/anniesch/surgical-finetuning)) | **No explicit licence** | No `LICENSE` file found via GitHub API. |
| **Kumar et al. paper** | Paper only | ArXiv / OpenReview; no code licence to report beyond the repo above. |

---

## 6. Concrete arm proposal

**Arm name:** `ortholarge_lpft_v1` (or `radnet_family_lpft_v1`).

**Phase 1 — LP (minutes, on CPU or idle card):**
- Input: cached study-level vectors for the trunk.
- Train a single linear layer (multi-label sigmoid).
- Epochs: up to 5, early-stop on validation loss with patience 2.
- Save head weights.

**Phase 2 — FT (normal card recipe):**
- Initialize the final linear layer from Phase 1.
- Fine-tune the entire trunk + head for 12 epochs.
- LR: trunk `4e-5`, head `4e-4` for epochs 1–2, then both `4e-5` (cosine anneal).
- Keep the exact same augmentations, mixup, and label smoothing as the baseline FT arm so the only difference is head initialization.

**Control arms:**
- Baseline FT from random head (already run for `ortholarge`).
- Frozen LP only (already run).

**Success criterion:** LP-FT beats the better of {frozen LP, baseline FT} on the holdout AUC. If it does, we have a lever; if it ties or loses, the trunk is either too brittle or the head initialization is not the bottleneck.
