# Q-DK — Continued-SSL trunk probes best but serves worst

**Date:** 2026-09-04
**Question (Claude):** `ortholarge` (DINOv3 ViT-L continued with SSL on 1.25M in-domain knee slices) beats its parent `v3large` by +0.0203 on a frozen linear probe, but fine-tuned members from it (`ortho_dense` 0.8452, `ortho_pool_read` 0.8464) underperform DINOv2/v3 seats (~0.855). Is this a named phenomenon, what causes it, and what fixes are supported?

## TL;DR

The inversion is a real instance of the **linear-probing / fine-tuning mismatch** (also called protocol-correlation or LP-FT rank inversion). The best-supported mechanism is **feature distortion during full fine-tuning**: when pretrained features are already linearly separable, vanilla FT changes them in harmful directions. The best-supported fix is **LP-FT + a lower backbone learning rate**. No published result guarantees this will close the gap for `ortholarge`, but it is the most evidence-backed arm to try next.

## 1. Named phenomenon

No paper isolates and names the exact "continued SSL probe up / FT down" pattern. It falls under the broader problem of **probing–fine-tuning mismatch**:

- **Knott et al. / Marks et al.**, *A Closer Look at Benchmarking Self-Supervised Pre-training with Image Classification* (IJCV 2024/2025, [arXiv:2407.12210](https://arxiv.org/abs/2407.12210)) measure protocol correlation across 26 SSL models. They find that **frozen linear probing** is the most stable protocol (Spearman r ≈ 0.90 with other ID metrics), while **full fine-tuning** is the most idiosyncratic (r ≈ 0.79). Rank inversions are expected.
- **Our local measurement** (`interpretations/kneemri/2026-09-03_the-frozen-probe-re-read.md`) records the same inversion: `ortholarge` +0.0203 over `v3large` frozen, but the served members are below the DINOv2/v3 bar.

## 2. Mechanisms

### Feature distortion (strongest evidence)

**Kumar et al.**, *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution* (ICLR 2022, [arXiv:2202.10054](https://arxiv.org/abs/2202.10054)) show that full fine-tuning can distort already-good pretrained features. Their LP-FT recipe — train a linear head on frozen features, then fine-tune the whole network from that head — recovers performance. This is the most direct explanation for `ortholarge`.

### Probing vs. fine-tuning measure different things

Frozen LP measures only linear separability of fixed features. Full FT also depends on plasticity, head initialization, regularization, and optimization. Knott et al. find that **10%-shot fine-tuning** is a better predictor of final FT ranking than frozen LP.

### Other plausible but less evidenced mechanisms

- **Representation collapse / loss of plasticity** from SSL continuation on a narrow corpus (plasticity/critical-periods literature, e.g. Achille et al. 2018; Lyle et al. 2023). Not directly measured for vision SSL continuation.
- **Inherited recipe mismatch**: the parent's fine-tuning hyperparameters may be wrong for the continued checkpoint.
- **Catastrophic forgetting** of general features during downstream fine-tuning.

## 3. Proposed fixes

| Fix | Evidence | Practical direction for `ortholarge` |
|---|---|---|
| **LP-FT** (Kumar et al. 2022) | Strong. 5 epochs LP + 5 epochs FT on ImageNet; ~1% ID and ~10% OOD gain over vanilla FT. | Initialize the final head from a frozen linear probe, then fine-tune trunk + head. This is the purpose of the `ortho_fresh_lowlr` arm. |
| **Lower backbone / discriminative LR** (ULMFiT; DAPT practice) | Strong general transfer evidence. | Trunk LR 10–100× below head LR, e.g. trunk 1e-5, head 4e-4 for the first 1–2 epochs. |
| **Longer / gentler warmup** | Standard for continued checkpoints. | Extend linear warmup to 1–2 epochs; keep peak trunk LR low. |
| **Surgical fine-tuning** (Lee et al., ICLR 2023, [arXiv:2210.11466](https://arxiv.org/abs/2210.11466)) | Moderate. Freeze early layers, tune upper layers + head. | Freeze early trunk layers if their relative gradient norms are large. |
| **EMA / interpolation with parent** (Wise-FT, Wortsman et al. 2022) | Moderate. Used when FT erases pretrained features. | Mix fine-tuned weights with parent/continued weights to preserve general features. |

## 4. Reverse case — does probe gain ever predict FT gain after the recipe is fixed?

No paper was found that explicitly reports the reverse: an initial probe-gain / FT-loss inversion that is resolved by a recipe change, after which the continued trunk reliably beats the parent.

General DAPT work shows positive downstream gains:

- **Gururangan et al.**, *Don't Stop Pretraining* (ACL 2020), shows domain-adaptive pretraining improves NLP downstream tasks after fine-tuning.
- **Ouyang et al.**, *Evaluating General Purpose Vision Foundation Models for Medical Image Analysis* (2023, [arXiv:2312.02366](https://arxiv.org/abs/2312.02366)), shows radiology-continued DINOv2 improves both LP and FT, but does not report an initial inversion.

The honest conclusion is that the inversion is a known risk of vanilla FT on strong frozen features, and LP-FT + low trunk LR is the best-supported next experiment, but the outcome for `ortholarge` must be measured.

## 5. Local action

The interpretation `interpretations/kneemri/2026-09-03_the-frozen-probe-re-read.md` already identifies the inversion as a **recipe question, not an architecture question**, and names `ortho_fresh_lowlr` as the control that makes the LP-FT triple readable. The next step is to run that arm and compare it to the DINOv2/v3 seat bar (~0.855).

## References

- Knott et al. / Marks et al. 2024/2025, [arXiv:2407.12210](https://arxiv.org/abs/2407.12210)
- Kumar et al. 2022, [arXiv:2202.10054](https://arxiv.org/abs/2202.10054)
- Lee et al. 2023, [arXiv:2210.11466](https://arxiv.org/abs/2210.11466)
- Gururangan et al. 2020, *Don't Stop Pretraining*
- Ouyang et al. 2023, [arXiv:2312.02366](https://arxiv.org/abs/2312.02366)
- `interpretations/kneemri/2026-09-03_the-frozen-probe-re-read.md`
