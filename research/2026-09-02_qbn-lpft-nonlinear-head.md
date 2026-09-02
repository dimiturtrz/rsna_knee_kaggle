# Q-BN — does LP-FT survive a nonlinear head, and how long is stage one?

## Sources

- Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*, arXiv:2202.10054 (2022).
- Tomihari & Sato, *Understanding Linear Probing then Fine-tuning Language Models from NTK Perspective*, arXiv:2405.16747 (2024).
- Davila et al., *Comparison of fine-tuning strategies for transfer learning in medical image classification*, arXiv:2406.10050 (2024).
- Mensah et al., *Towards Interpretable Foundation Models for Retinal Fundus Images*, arXiv:2603.18846 (2026).

## 1. Is there published evidence for LP-FT with a nonlinear / attention head?

**No direct evidence.** The LP-FT literature almost universally defines stage one as training a *linear* classifier on frozen features.

- Kumar et al. prove the feature-distortion tradeoff for overparameterized **two-layer linear networks** and explicitly list "dealing with non-linear activations" as an open question (Section 6). Their guarantee that LP-FT preserves pretrained features does not extend to nonlinear heads.
- Tomihari & Sato analyze LP-FT for Transformers, but their NTK decomposition relies on a **linear head**. They identify the mechanism as an increase in the *linear head norm* during stage one, which reduces feature changes in stage two. There is no equivalent norm story for a `LayerNorm → Linear → GELU → attention` head.
- Davila et al. evaluate LP-FT on DenseNet-121/ResNet-50/VGG-19 in medical imaging; their stage one is the standard last (linear) classifier layer.

### What that means for the two failure modes

Both are **plausible and not ruled out** by the literature:

- **(a) Worse basin:** a nonlinear head can fit frozen features into a conditional distribution that is a poor initialization once the trunk is allowed to move. The linear-head proofs avoid this because the global optimum of a linear probe is unique (up to scaling) and directly related to the frozen features.
- **(b) Redundancy:** a deep head can absorb the large initial gradients itself, so a single-stage fine-tune at a low rate with a randomly initialized head may already be doing what LP-FT does. This is the more likely failure mode for a head with its own depth and attention.

### Indirect evidence

- Applied transfer-learning recipes in object detection and medical segmentation often train the task head first with a frozen backbone, then unfreeze. Those heads are typically linear or small conv layers, not attention-based aggregators.
- Mensah et al. (retinal fundus) train a *linear* classification head for 10 epochs before unfreezing, suggesting the linear-head stage-one convention carries over to medical imaging.

**Bottom line:** the effect is an empirical bet, not a theorem. The honest published answer is "nobody has shown this for a nonlinear head."

## 2. Stage-one length

There is **no published sweep** for stage-one length, and the papers that report a number do so as a fixed heuristic:

- **Mensah et al.:** 10 epochs for a linear head, then unfreeze, max 50 total epochs with early stopping.
- **Davila et al.:** 50 total epochs, early-stopping patience 5; LP-FT uses "a set number of epochs" for stage one (not specified).
- **Kumar et al.** and **Tomihari & Sato** do not report a stage-one epoch sweep; their focus is on the ordering, not the duration.

### Recommended rule

Use a **convergence criterion**, not a fixed number:

- Train stage one with the trunk frozen until the head's validation metric stops improving for a patience window (e.g., 3 epochs).
- Cap it at a generous maximum (e.g., 50 epochs) because frozen-trunk training is cheap.
- For a nonlinear/attention head, expect stage one to need **at least as many epochs as a linear head**, and likely more. The linear-head precedent of 10 epochs is a floor, not a ceiling.

This avoids both under-training (head barely better than random) and over-training (head fits idiosyncrasies of frozen features).

## 3. How to test it without relying on the linear-head theory

Because the theory does not cover the SlotHead, the arm should be designed to answer its own validity question:

1. **`ortho_lpft`** — current plan: stage one = train SlotHead on frozen trunk, stage two = unfreeze whole trunk at 1e-5.
2. **Control `ortho_fresh_lowlr`** — same recipe as stage two, but initialize the SlotHead randomly. This isolates whether warm-starting the head matters.
3. **Optional `ortho_linear_lpft`** — train a *linear* probe first, then swap/expand to SlotHead for stage two. This is only worth the effort if (2) shows a benefit and you want to know whether the gain comes from the ordering or from the specific nonlinear head.

If `ortho_lpft` does not beat `ortho_fresh_lowlr`, the LP-FT ordering is not buying anything with this head, and you should drop the two-stage complexity.

## 4. Limitations / what is thin

- No published LP-FT result with an attention or slot-based head.
- No published stage-one epoch sweep; all numbers are heuristics.
- The retinal-fundus 10-epoch rule is for a linear head on a different modality and dataset scale.
