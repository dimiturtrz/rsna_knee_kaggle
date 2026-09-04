# Q-DT — what is the "proper adaptation" in AnyMC3D, and what is the measured spread of slice-fusion operators?

**Question from Claude (Q-DT, 2026-09-04):** AnyMC3D beats 3D FMs with a 2D backbone, so the win is in the slice-to-study adaptation/fusion. What exactly does it do, is there a trunk-held ablation of fusion operators, and does anyone report per-finding behavior for focal labels?

## Bottom line

- AnyMC3D's adaptation is **query-based attention pooling over per-slice CLS tokens**, with **no positional encoding** over slice index or spacing.
- The 2D FM is adapted with **LoRA** (rank 8, scaling 16) on patch embedding and self-attention projections.
- A trunk-held fusion ablation exists (AnyMC3D Appendix F, Table 5): on T5 (PDAC early detection, CT), query-based pooling reaches **0.962 AUC**, vs average 0.958, max 0.946, median 0.944, Transformer-over-slices 0.950, LSTM 0.903.
- **No paper** reports the same trunk-held mean/max/attention/Transformer spread on knee MRI, and **no paper** reports per-label/per-finding behavior by fusion operator.

---

## AnyMC3D mechanism

From Liu et al., *Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification* ([arXiv:2512.12887](https://arxiv.org/abs/2512.12887)):

> *"Let the 3D input be \(\mathbf{x}\in\mathbb{R}^{C\times H\times W\times S}\), with \(S\) slices along a chosen axis. We form 2D slices \(\mathbf{x}^{s}\in\mathbb{R}^{C\times H\times W}\), encode each slice with the adapted 2D FM, and extract the class-token from the last block as the slice embedding: \(\mathbf{h}_{s}=\tilde{f}_{\theta,\psi_{t}}(\mathbf{x}^{s})\in\mathbb{R}^{d},\ s=1,\ldots,S\)."*

> *"To fuse slice embeddings, we design a lightweight module that maintains strong performance while ensuring scalability. Prior work typically imposes ordered priors through sequence modeling with RNNs or Transformers. However, 3D medical images often exhibit anisotropic spacing and variable coverage. Strict sequence modeling can be overly prescriptive and sensitive to acquisition variability. Therefore, we propose to fuse slice embeddings in a permutation-invariant manner via query-based attention pooling."*

> *"Specifically, slice embeddings are stacked as \(\mathbf{H}=[\mathbf{h}_{1},\ldots,\mathbf{h}_{S}]^{\top}\in\mathbb{R}^{S\times d}\) and aggregated using a learnable task query \(\mathbf{q}_{t}\in\mathbb{R}^{d}\) that assigns higher weights to task-relevant slices. The volume embedding \(\mathbf{v}\) is computed as: \(\boldsymbol{a}=\operatorname{softmax}(\mathbf{H}\mathbf{q}_{t}/\sqrt{d})\in\mathbb{R}^{S},\ \mathbf{v}=\boldsymbol{a}^{\top}\mathbf{H}\in\mathbb{R}^{d}\)."*

Key points:
- **Fusion operator:** one learnable task query attends over all slice embeddings; softmax attention weights produce a weighted sum.
- **No positional encoding:** the aggregation is explicitly permutation-invariant. Slice order / spacing is not injected.
- **Pooled tokens:** per-slice **CLS tokens** from the last transformer block. Patch tokens are used only in an auxiliary segmentation branch, not in classification fusion.
- **FM adaptation:** LoRA adapters on patch embedding and self-attention projection layers; the task query and classification head are the only new learned parameters for the fusion. (LoRA rank 8, scaling 16; task query initialized from truncated normal std 0.02.)

---

## Trunk-held fusion ablation

AnyMC3D Appendix F, "Impact of Slice Fusion Strategy," compares aggregation strategies on task T5 (PDAC early detection) with the same backbone/adaptation:

| Fusion Method | Sequential | Params (M) | AUC |
|---|---:|---:|---:|
| Average pooling | No | 0 | 0.958 |
| Max pooling | No | 0 | 0.946 |
| Median pooling | No | 0 | 0.944 |
| LSTM | Yes | 5.5 | 0.903 |
| Transformer encoder | Yes | 7.0 | 0.950 |
| Query-based attention (Ours) | No | 0.001 | **0.962** |

The authors' commentary:
> *"Simple pooling operations (average, max, median) require no additional parameters but treat slices independently without modeling inter-slice relationships. Sequential methods like LSTM and Transformer encoder explicitly model slice order but introduce substantial parameters ... and show mixed results, with LSTM underperforming (0.903 AUC) despite high parameter cost. Our query-based attention pooling achieves the best performance (0.962 AUC) with minimal parameters (0.001M)."*

**Observed spread:** query-based vs average = +0.004 AUC; vs max = +0.016 AUC; vs Transformer-over-slices = +0.012 AUC. This is 0.4–1.6% on a single CT task.

---

## Other literature

| Paper | Relevance |
|---|---|
| **Medical Slice Transformer (MST)** ([arXiv:2411.15802](https://arxiv.org/abs/2411.15802)) | DINOv2 + Transformer over slice embeddings; evaluated on breast MRI, chest CT, and **knee MRI (MRNet meniscus)**. Ablation compares Transformer vs linear layer vs averaging, and tests additive positional embeddings. Transformer without positional encoding wins, but numeric AUC deltas are not extractable from the PDF tables. Direction matches AnyMC3D. |
| **DinoAtten3D** ([arXiv:2509.12512](https://arxiv.org/abs/2509.12512)) | DINOv2 + soft attention pooling over slice embeddings on brain MRI. Does not compare mean/max/Transformer with the same trunk. |

No paper was found that holds the 2D trunk fixed and reports a numeric comparison of mean vs max vs attention vs Transformer-over-slices specifically on a knee MRI classification task.

---

## Per-finding / focal-label behavior

No published study reports per-label AUCs stratified by fusion operator for meniscus, lateral meniscus, or any other focal knee-MRI finding. The AnyMC3D ablation reports overall AUC only, and MST reports overall AUC on MRNet meniscus (0.85 ± 0.04) without per-label or per-fusion breakdowns.

Therefore, the hypothesis that attention pooling disproportionately helps focal labels like Lateral Meniscus remains plausible but unmeasured in the literature.

---

## Implications

- The published operator spread is small (≤1.6% AUC) and measured on CT, not knee MRI.
- AnyMC3D's design choice suggests that **ordered sequence modeling is not the win** — the win is a lightweight, task-conditioned attention over an unordered bag of slice embeddings.
- If the project wants a knee-MRI-specific number for Lateral Meniscus, the ablation must be run locally; it is not available to read.

---

*Written 2026-09-04 in response to Claude Q-DT.*
