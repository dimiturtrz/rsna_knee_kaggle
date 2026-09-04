# Q-EI — can Dehghan et al.'s patch-size gains be replicated at 224 px?

Claude asked three replication questions about Dehghan et al., *Effect of
Patch Size on Fine-Tuning Vision Transformers in Two-Dimensional and
Three-Dimensional Medical Image Classification* (arXiv:2602.18614v1).
This note answers them from the paper text.

Source: [arXiv:2602.18614v1](https://arxiv.org/html/2602.18614v1)

## 1. Epoch budget: fixed or per-patch-size?

**Fixed maximum budget, with validation-loss checkpoint selection.**
From the experimental setup (Section 2.3):

> "The models are trained for 80 epochs using the cross-entropy loss
> function, and the best-performing checkpoint is selected based on the
> lowest validation loss."

All patch sizes use the **same 80-epoch ceiling**. The learning-rate
schedule is also shared: AdamW at 10⁻⁴, halved every 25 epochs. There is
no evidence that smaller patches received a larger epoch budget or that
each patch size was trained to its own convergence point independently.

However, because the best checkpoint is chosen by validation loss, the
*effective* stopping epoch differs across patch sizes. In that limited
sense the schedule is not rigidly fixed to the last epoch. But the
maximum budget and the LR schedule are identical for all patch sizes.

## 2. Regularization: tuned per patch size?

**No.** The paper gives one optimization recipe for all patch sizes:

- Optimizer: AdamW (cites Loshchilov & Hutter 2017).
- Learning rate: 10⁻⁴ for all experiments.
- Scheduler: halved every 25 epochs for all experiments.
- Loss: cross-entropy for all experiments.
- Epochs: 80 for all experiments.
- Augmentation: random resized crop, horizontal flip, rotation, light
color jitter for 2D; random flip/rotation for 3D.

The paper does **not** report a weight-decay value, drop-path / stochastic
depth, layer-wise decay, or any other regularizer that was adjusted with
patch size. The two training pipelines (2D and 3D) differ only in input
dimensionality, not in regularization.

## 3. Image resolution of the twelve datasets?

**28 × 28 for 2D and 28 × 28 × 28 for 3D.** The authors explicitly chose
the smallest MedMNIST V2 resolution "to ensure that all experiments can be
run in a single GPU setup":

> "Although the MedMNIST V2 collection provides datasets in various
> resolutions (up to 224 for 2D datasets and up to 64 for 3D datasets), we
> use the smallest available size, namely 28×28 for 2D datasets and
> 28×28×28 for 3D datasets."

Patch sizes evaluated on the 2D data are {28, 14, 7, 4, 2, 1}. That gives
token grids of:

| patch | tokens on 28 px image |
|-------|-----------------------|
| 28    | 1 × 1  = 1   |
| 14    | 2 × 2  = 4   |
| 7     | 4 × 4  = 16  |
| 4     | 7 × 7  = 49  |
| 2     | 14 × 14 = 196 |
| 1     | 28 × 28 = 784 |

Claude's `radiodino_s16` uses 224 px slices with patch 16 → 14 × 14 =
196 tokens, matching Dehghan's **patch 2**. Claude's `radiodino_s8` uses
224 px with patch 8 → 28 × 28 = 784 tokens, matching Dehghan's
**patch 1**.

## What this means for the arm

**The resolution mismatch is real and it undermines the transfer
argument.** Dehghan's headline numbers (up to +12.78% balanced accuracy on
2D, +23.78% on 3D) are measured on 28 px images. At that resolution,
patch 1 is a 1 × 1 px token and patch 2 is a 2 × 2 px token. In Claude's
setup, the same token counts are produced by 8 × 8 px and 16 × 16 px
patches on 224 px slices. The physical granularity per token is
completely different.

The authors themselves flag this limitation in their conclusion:

> "...these datasets may not fully reflect the resolution and complexity
> of real clinical images. Future work should investigate whether the
> patch-size trends observed here also hold for real-world high-resolution
> clinical datasets."

So the arm's premise — "smaller patch size helped Dehghan on medical
images, therefore patch 8 should help on 224 px knee MRI slices" — is
not a direct replication. It is a cross-resolution transfer guess.

The schedule and regularization answers do not rescue the arm: Dehghan
used the same budget and the same regularization for all patch sizes, so
Claude's −0.0137 is not a schedule artefact in the narrow sense. The
more likely explanation is the resolution / token-granularity mismatch
combined with the well-known overfitting risk when token count grows on a
small corpus.
