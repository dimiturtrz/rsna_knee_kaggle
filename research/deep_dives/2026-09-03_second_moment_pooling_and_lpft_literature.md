# Q-CK — Second-moment pooling along the slice axis and the LP-FT inversion

2026-09-03. Answers two literature questions from the frozen-probe re-read
(`interpretations/kneemri/2026-09-03_the-frozen-probe-re-read.md`, beads `knee-2rmdc`, `knee-6hfi`).

---

## Q-CK-1 — Is slice-axis variance / higher-moment pooling a known thing, and does the frozen gap survive fine-tuning?

### Short answer

Variance pooling is a named technique in **multiple-instance learning (MIL)** for whole-slide images, but the published form is over **patch bags**, not over MRI/CT slice stacks. General-purpose **second-order / covariance pooling** (B-CNN, MPN-COV, DeepO2P) is well known in computer vision, but it is applied to spatial feature maps, not specifically to the depth/slice axis. I did not find a paper that explicitly pools standard deviation or variance **along the slice axis** of a 2.5D medical volume. Whether the frozen-feature `mean+std` advantage survives full fine-tuning is an **open empirical question**; the literature gives arguments both ways and no direct measurement.

### Named precedents

| Technique | Where | What it pools | Citation |
|---|---|---|---|
| **Variance pooling in MIL** | Computational pathology / WSI | Variance of patch embeddings in a tumor bag | Carmichael et al., *Incorporating Intratumoral Heterogeneity into Weakly-Supervised Deep Learning Models via Variance Pooling*, MICCAI 2022 / arXiv:2206.08885. Code: [mahmoodlab/VARPOOL](https://github.com/mahmoodlab/VARPOOL) |
| **Global covariance / second-order pooling** | General vision | Covariance of local descriptors across spatial positions | B-CNN (Lin et al. 2017), MPN-COV (Li et al. 2017), DeepO2P (Ionescu et al. 2015) |
| **"VarPool" layer** | EEG / time-series | Variance inside sliding windows along the temporal axis | A Spatial-Spectral and Temporal Dual Prototype Network for MI-BCI, arXiv:2407.03177 |

Carmichael et al. is the closest medical precedent: they add a variance-pooling branch to attention-based MIL and report improved survival prediction on 4,479 TCGA whole-slide images. The architecture explicitly models the **spread of instance embeddings** within a bag, which is conceptually the same operation your `SliceMoments` readout performs along the slice axis.

### What is *not* in the literature

I found no published 2.5D medical-imaging work that pools the **standard deviation or variance along the slice axis**. The standard 2.5D aggregation operators are:

- **max / mean pooling** over slices (e.g. RSNA 2023 1st-place solution: "simple maximum aggregation on sigmoid predictions"),
- **attention-based MIL** (Ilse et al., Shi et al. Loss-Based Attention),
- **RNN / GRU / LSTM / Transformer** over the slice sequence (Tomita et al., Salehinejad et al., AW-Net, many others),
- **triplet/stacked-slice inputs** to a 2D CNN (KneeXNet-2.5D, many segmentation papers).

A 2024 pathology paper (*Triage of 3D pathology data via 2.5D MIL*) mentions "variance pooling" in passing, but it is referring to the Carmichael patch-bag method, not to slice-axis statistics.

### Does the frozen `mean+std` gap survive fine-tuning?

**No direct evidence either way.** The relevant mechanisms are:

1. **Arguments that it could vanish.** A deep enough head with attention can, in principle, learn to estimate the spread of a channel from the slice sequence. If the trunk is also free to move, the network could learn the same discriminative shape end-to-end. Kumar et al.'s *feature distortion* story (below) says full fine-tuning can destroy good frozen features, but if the head is strong enough it may simply re-learn the needed statistic.

2. **Arguments that it could hold.** Your own data show that a **learned attention head over a max pool can be much worse than a linear ridge on the same frozen trunk** (`ortho_lpft_head` 0.7537 vs ridge 0.8249; see Q-CK-2 below). If a trainable head cannot even beat a simple ridge on the first moment, there is no reason to assume it will automatically recover the second moment. The `moments > mean > max` ordering on frozen features is also a statement about **linear separability** of a statistic the head currently discards.

3. **The spread is not attention in disguise.** Attention weights are a normalized set of scalar coefficients; they can do weighted selection but they do **not** directly compute second moments. A head could be *trained* to approximate variance (e.g. by self-attention over squared deviations), but that is a learned approximation, not an equivalent representation. The aggregation chapter already measured that learned first-moment pools (`attend`, `labelwise`) add nothing over `max`; `moments` tests a genuinely different axis.

### Practical reading

The cleanest way to settle the survival question is a **controlled fine-tuning arm**:

- Trunk: `ortholarge` (or any trunk where the frozen `moments` gap is large).
- Slice pooling: `moments` (the new module already exists in `kneemri.models.moment_pooling`).
- Head: the same `SlotHead` used for `max`/`attend` arms.
- Control: the same recipe with `slice_pooling=max` on the same trunk.

If `moments` closes to zero after fine-tuning, the frozen gap was a probe artefact. If it holds, the project has an unexploited axis. The cost is one 12-epoch run; the literature cannot answer it without this measurement.

---

## Q-CK-2 — Trunks that probe best and fine-tune worst

### Short answer

The named phenomenon is **feature distortion under fine-tuning**, studied by Kumar et al. (ICLR 2022) as *LP-FT* (linear-probing then fine-tuning). For a continued-pretraining checkpoint, the same dynamic is usually discussed as **catastrophic forgetting of the domain continuation** or, more precisely, as a **rank inversion between linear-probe and full-fine-tune protocols**. The 0.071 gap between a ridge on frozen features and a small trained head on the same frozen trunk is **not** explained by feature distortion; it is explained by a **pooling + head mismatch** plus likely underfitting/optimization failure in the stage-one LP head.

### Named literature

| Name | Paper | What it says |
|---|---|---|
| **Feature distortion / LP-FT** | Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*, ICLR 2022 / arXiv:2202.10054 | Full fine-tuning with a randomly initialized head can distort pretrained features and underperform linear probing, especially OOD. LP-FT (fit a linear head on frozen features, then fine-tune the whole network from that initialization) recovers both ID and OOD performance. The theory is proven for two-layer linear nets; nonlinear heads are listed as an open question. |
| **Protocol / rank correlation** | Knott et al. / Marks et al., *A Closer Look at Benchmarking Self-Supervised Pre-training*, IJCV / arXiv:2407.12210 | Linear probing has high average correlation with other evaluation protocols (Spearman r ≈ 0.90 on ImageNet), but **full fine-tuning is the most idiosyncratic protocol** (r ≈ 0.79). LP ranking does not perfectly predict FT ranking; inversions are expected. |
| **Transferability estimation** | LogME (You et al., ICML 2021) | Estimates how well frozen features transfer, but Bassignana et al. (*Evidence > Intuition*, EMNLP 2022) find LogME's predictive power drops after full fine-tuning because frozen and fine-tuned representations misalign. |
| **Surgical / discriminative fine-tuning** | Lee et al., *Surgical Fine-Tuning*, ICLR 2023 / arXiv:2210.11466 | Freezing or differentially updating the right layers can outperform full FT. Useful as a repair when a continued trunk collapses under full FT. |
| **Discriminative learning rates** | Howard & Ruder, *Universal Language Model Fine-tuning for Text Classification*, 2018 | Lower layers get smaller LR. Mentioned by Kumar et al. as a related idea to prevent feature distortion. |
| **Medical FM adaptation survey** | Phuntsho et al., *Adaptation of Foundation Models for Medical Image Analysis*, arXiv:2511.01284 (2025) | Reviews SFT, domain pretraining, PEFT, etc.; notes that full fine-tuning can underperform linear probing or partial adaptation when pretrained features are already strong. |

### What explains a continued trunk that probes best and serves worst

Your controlled pair is strong: `ortholarge` vs `v3large` is the same DINOv3-L architecture, same read size, differing only by OrthoFoundation's in-domain continuation on 1.25M knee images. The frozen probe reads `ortholarge` **+0.020** above `v3large`, four times. The served members read `ortholarge` **below** the DINOv2 seats.

This pattern has three plausible, non-mutually-exclusive explanations in the literature:

1. **Feature distortion (Kumar).** The continuation produced features that are highly linearly separable on the target domain. Full fine-tuning with a randomly initialized head then moves the trunk away from that good solution before the head has learned to use it. Kumar et al. identify this as the **alignment error of random head initialization (φ₂)** in their lower-bound analysis. LP-FT is the published repair.

2. **Catastrophic forgetting of the continuation.** Full fine-tuning on 4,407 studies may overwrite the 1.25M-knee statistics that made the trunk good, reverting it toward the base DINOv3-L behavior. This is a specific instance of the stability-plasticity tradeoff studied in continual learning.

3. **The continuation made features more brittle.** A narrower domain distribution can improve linear separability while making the features easier to distort with a small update. There is no published measurement of this tradeoff for knee MRI; it is a conjecture that matches your numbers.

### The 0.071 gap: ridge 0.8249 vs trained head 0.7537 on the same frozen trunk

This comparison is between two different readouts of the **same frozen `ortholarge` trunk**:

| Readout | Score | What it is |
|---|---|---|
| Ridge, `moments` readout | **0.8249** | Linear classifier on `[mean; std]` over slices, 2× width, per-trunk ridge penalty |
| Trained `SlotHead`, `max` pooling | **0.7537** | `LayerNorm → Linear → GELU` projection to 256-d, per-diagnosis slot attention, max over slices |

The gap is not mainly about LP-FT. It is about the **head and pooling choice on frozen features**. Decompose it:

- `moments` vs `max` on frozen features (both ridge): from the readout table, for `ortholarge@focal` the `moments` ridge is **0.8279** and the `max` ridge is **0.8146**. The second-moment gain is ~**+0.013**.
- Trained `SlotHead` with `max` vs ridge with `max`: 0.7537 vs ~0.8146. The trained head is **~0.061 below** a linear ridge on the same max-pooled frozen features.

So ~85% of the 0.071 gap is that the trained stage-one head is a worse readout than a ridge, and ~15% is the missing second moment.

### Why a trained head can underperform a ridge on identical frozen features

This is a known, under-discussed phenomenon. The closest published example:

> "Critically, the 513-parameter attention head degrades below the parameter-free baseline on ARID v1.5 (66.06% vs. 68.37%)."  
> — *Human activity recognition under low-light conditions using frozen CLIP with lightweight adaptation* (Springer 2026).

That paper finds that a lightweight attention adapter on frozen CLIP embeddings can do worse than no adapter at all. Mechanisms that can cause this:

1. **Too little capacity in the wrong places.** `SlotHead` projects 1024-d slice features down to 256-d before attention. If the target task needs directions in the original 1024-d space that are not preserved by a random 1024→256 projection, the head starts from a damaged representation. A ridge has direct access to all 1024 dimensions.

2. **The attention prior may fight the data.** `SlotHead` uses an anatomy-based prior over slots. When the trunk is frozen, the per-diagnosis attention is asked to discover which sequence carries evidence without any slice-level supervision. If the prior is slightly wrong for `ortholarge`'s feature geometry, it can suppress signal that a mean/max ridge would keep.

3. **Optimization mismatch.** The stage-one head was trained for 12 epochs with `head_lr=0.001`, `backbone_lr=8e-6`, no mixup, and the full `one_cycle` schedule. A ridge with a five-decade penalty search is a much better-optimized linear readout. The head may simply be undertrained relative to the ridge's hyperparameter search.

4. **The head was designed for fine-tuning, not frozen-feature reading.** Slot attention may only become useful once the trunk can move to produce slot-discriminable features. On a frozen trunk, the inductive bias is not yet earned.

### What would explain it away

The obvious confound is that the 0.8249 ridge uses `moments` while the 0.7537 head uses `max`. Run a ridge on `max` for `ortholarge` (the readout table already has it at 0.8146) and a trained head on `moments`. If the trained head on `moments` reaches ~0.82, the gap was mostly pooling. If it stays near 0.75, the head is the problem.

---

## Actionable recommendations

1. **To test the second-moment axis:** arm `ortholarge` (or `v3large`) with `slice_pooling=moments` and the existing `SlotHead`, against `slice_pooling=max` as control. This is the only way to know whether the frozen `moments` gain survives fine-tuning.

2. **To understand the 0.071 head gap:** run the four-cell ablation on frozen `ortholarge`:
   - ridge + max
   - ridge + moments
   - SlotHead + max (already have: 0.7537)
   - SlotHead + moments

   This separates pooling from head quality. If `SlotHead + moments` is still far below `ridge + moments`, the project should reconsider whether `SlotHead` is the right readout for frozen-feature screening.

3. **To rescue the continued trunk:** the LP-FT triple (`ortho_lpft_head`, `ortho_fresh_lowlr`, `ortho_lpft`) is the correct experiment. If even LP-FT does not lift `ortholarge` above the DINOv2 seats, the continuation is either being forgotten or the distribution match is weaker than the probe suggests. Consider surgical fine-tuning (freeze lower blocks, tune upper blocks + head) or a much lower trunk LR as follow-ups.

4. **Do not use frozen LP alone to rank trunks for full FT.** Knott et al. show full FT is the most idiosyncratic evaluation protocol; inversions are expected. Add a 10%-study or 1–2-epoch short FT probe for ranking.

---

## References

- Carmichael, Song, Chen, Williamson, Chen, Mahmood. *Incorporating Intratumoral Heterogeneity into Weakly-Supervised Deep Learning Models via Variance Pooling.* MICCAI 2022. arXiv:2206.08885. Code: https://github.com/mahmoodlab/VARPOOL
- Kumar, Raghunathan, Jones, Toner, Liang. *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution.* ICLR 2022. arXiv:2202.10054
- Knott / Marks et al. *A Closer Look at Benchmarking Self-Supervised Pre-training with Image Classification.* IJCV / arXiv:2407.12210
- You, Liu, Yao, Ma, et al. *LogME: Practical Assessment of Pre-trained Models for Transfer Learning.* ICML 2021.
- Bassignana, Plank, Søgaard. *Evidence > Intuition: Transferability Estimation for NLP.* EMNLP 2022.
- Lee, Malladi, Chen, Arora, Sanyal, Dimakis. *Surgical Fine-Tuning Improves Adaptation to Distribution Shifts.* ICLR 2023. arXiv:2210.11466
- Howard & Ruder. *Universal Language Model Fine-tuning for Text Classification.* ACL 2018.
- Lin, RoyChowdhury, Maji. *Bilinear CNN Models for Fine-Grained Visual Recognition.* ICCV 2015.
- Li, Xie, Yang, Lin. *Is Second-order Information Helpful for Large-scale Visual Recognition?* ICCV 2017 (MPN-COV).
- Ionescu, Vantzos, Sminchisescu. *Matrix Backpropagation for Deep Networks with Structured Layers.* ICCV 2015 (DeepO2P).
- *Human activity recognition under low-light conditions using frozen CLIP with lightweight adaptation.* Springer 2026.
- Yu et al. *A multimodal vision foundation model for generalizable knee pathology.* arXiv:2601.18250 (OrthoFoundation).
- Phuntsho et al. *Adaptation of Foundation Models for Medical Image Analysis.* arXiv:2511.01284 (2025).
