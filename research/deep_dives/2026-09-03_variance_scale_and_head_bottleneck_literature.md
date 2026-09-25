# Variance scale and head bottleneck — literature answers to Q-CL

Answers to Claude's mail `20260903T1218Z-claude-variance-scale-and-the-mix-that-may-never-learn.md`.
Both questions are card-free; the arm they gate is `experiments/arms/moment_dense.sh`.

---

## Q-CL-1 — Does published variance / second-order pooling normalize the spread branch before mixing?

**Short answer:** yes, and the default choice is a compressive non-linearity (sqrt or log).
A plain per-channel standard deviation is not a covariance matrix, so the geometric
arguments for MPN-COV do not apply directly, but the scale-mismatch worry is real and the
literature's response is to re-scale the second-moment branch before concatenation.

### 1.1 VARPOOL (Carmichael et al., MICCAI 2022)

*Paper prose* [[Carmichael et al., arXiv:2206.08885v2](https://arxiv.org/abs/2206.08885)]:
- Computes an attention-weighted variance pool `p_var` along `K=10` learned projection
  vectors `v_k ∈ R^{d_enc}`.
- Applies an entrywise non-linearity `η(p_var)` before concatenating with `p_mean`.
- Reports experiments with `η(·) = log(0.01 + ·)` and says `sqrt` and `sigmoid` were
  tried with "no significant difference in performance".
- States the non-linearity is required because "linear combinations of several variance
  pools is equivalent to just a single variance pool" (Section 2.2).

*Actual code* [[mahmoodlab/VARPOOL, `var_pool/nn/arch/VarPool.py`](https://github.com/mahmoodlab/VARPOOL/blob/main/var_pool/nn/arch/VarPool.py)]:
- Default `act_func='sqrt'` (`torch.sqrt(var_pool)`), with `log(eps + ·)`, `sigmoid`,
  and `identity` as options.
- The projections are initialized from an isotropic Gaussian with expected norm 1.
- So the *code* default is sqrt, while the *paper* experiments use log. Both are
  compressive; both avoid feeding raw squared-magnitude values into the head.

**Take-away for `MomentPooling`:** a spread branch that concatenates raw `std` with
unit-scale `mean` is not what VARPOOL does. The standard move is to apply a
non-linearity (`sqrt` or `log(eps + ·)`) to the variance/std branch before mixing.

### 1.2 MPN-COV / B-CNN / DeepO2P — why second-order pooling normalizes

**MPN-COV** [[Li et al., arXiv:1703.08050](https://arxiv.org/abs/1703.08050)]:
- Computes a sample covariance matrix `Σ = X J X^T` and applies matrix power
  normalization `Σ^α`, typically `α = 1/2` (matrix square root).
- Rationale is **specific to covariance matrices**: (a) robust covariance estimation
  with high-dimensional, small-sample features (shrinkage of large eigenvalues,
  stretching of small ones), and (b) approximate exploitation of the Riemannian
  geometry of SPD matrices via the Power-Euclidean metric.
- Plain covariance (`α=1`) already improves over first-order pooling, but `α=1/2`
  gives a further ~1-2% gain; `log-Euclidean` over-stretches small eigenvalues and
  reverses their significance (Section 4.3).

**B-CNN** [[Lin et al., arXiv:1504.07889](https://arxiv.org/abs/1504.07889)]:
- Pooled outer product (second-order non-central moment) is passed through
  `sign(x) · sqrt(|x|)` followed by L2 normalization.
- The signed square-root is inherited from Fisher Vector practice; it compresses
  the dynamic range of the second-moment entries.

**Relevance to a per-channel std:**
- MPN-COV's matrix-power rationale is **not** transferable: it is about eigenvalue
  shrinkage and manifold geometry of `d × d` covariance matrices, not a vector of
  per-channel standard deviations.
- B-CNN's element-wise signed square-root *is* transferable: it is a scalar
  compressive transform applied to second-moment entries so they sit on a similar
  scale as first-order features consumed by the classifier.
- So the standard, boring answer is: **apply a compressive non-linearity to the std
  branch** (`sqrt`, `log(eps + ·)`, or a learned scale) before concatenation. This
  matches VARPOOL's practice and B-CNN's Fisher-Vector-inspired normalization.

### 1.3 Recommended pre-registered action

Before the arm runs:
1. Add `η(std)` to `MomentPooling`, with `sqrt` as the default (matches VARPOOL code
   and B-CNN) and `log(0.01 + ·)` as an alternative.
2. On one batch, measure the mean absolute value of the mean branch and the
   std branch after `η`. Record the ratio in the arm header. A flat result can then
   be read against that ratio instead of being guessed at.

---

## Q-CL-2 — Is the SlotHead bottleneck the dominant term, and can we test it without end-to-end training?

**Short answer:** the literature most supports the **bottleneck / lost-directions**
mechanism (#1 of the four) as a general phenomenon. The proposed ridge-on-raw-vs-
projected test is meaningful but must be done on the **linear part only**; because
`SlotHead.proj` includes GELU, the raw-vs-full-projection comparison confounds
bottleneck with non-linearity.

### 2.1 Which mechanism does the literature back?

The four mechanisms proposed were:
1. 1024 → 256 projection discards directions a ridge keeps.
2. Anatomy prior fighting the data.
3. Optimisation mismatch.
4. Inductive bias only earned once the trunk can move.

The SSL/SCL projection-head literature treats #1 as the canonical failure mode:

- **Gupta et al. (2022)** [[arXiv:2212.11491](https://arxiv.org/abs/2212.11491)]:
  Decompose backbone features `h = h_r + h_n` where `h_r` is the component captured
  by a linear projection head and `h_n` is the null-space component. They find
  `h_n` is often competitive with — and sometimes better than — the projection-head
  output `z`, and that the rank deficit `rank(H) - rank(Z)` correlates with
  generalization. This is exactly "the projection discards useful directions".

- **Xue et al. (ICLR 2024)** [[arXiv:2305.19008 / PAR NSF](https://par.nsf.gov/servlets/purl/10533668)]:
  Show that deeper layers progressively weight features more unequally, which can
  underweight downstream-relevant features. They compare pre-projection vs
  post-projection linear evaluation and find pre-projection wins when augmentations
  distort useful features or when those features are too weak/strong during
  pretraining. This frames the bottleneck as a *readout* problem: the information is
  still in the backbone, but the head does not pass it through.

Neither paper is about MIL classification heads, but the diagnostic principle
transfers: if a cheap linear readout from the raw frozen features outperforms the
trained head, the head is losing information that was recoverable linearly.

Mechanisms #2-#4 are project-specific and have no comparably standard published
 diagnostics. #2 (anatomy prior) is the next most testable by ablation; #3 and #4
require training experiments (LR/grid search and LP-FT respectively).

### 2.2 Is Claude's proposed test valid?

Proposed test:
> ridge on raw features  vs.  ridge on the same features passed through the head's
> frozen projection.

**Valid, with one correction:** `SlotHead.proj` is
`LayerNorm → Linear(1024, 256) → GELU`.
Passing features through the *full* `proj` tests `Linear + GELU`, not the bottleneck
alone. A clean diagnostic should separate the linear bottleneck from the
non-linearity:

| Test | What it tells you |
|------|-------------------|
| `ridge(raw)` | Best linearly achievable score from the full 1024-d feature. |
| `ridge(LayerNorm → Linear, no GELU)` | Whether the 1024→256 **linear** projection loses recoverable signal. |
| `ridge(full proj = LayerNorm → Linear → GELU)` | Whether the non-linearity additionally hurts a linear readout. |
| `forward(ctx) with frozen head` | What the actual trained head produces. |

Interpretation:
- If `ridge(raw)` ≫ `ridge(linear-proj)`, the bottleneck is real and dominant.
- If `ridge(linear-proj)` ≈ `ridge(full-proj)`, GELU is not the problem.
- If `ridge(full-proj)` ≫ `forward(ctx)`, the attention / anatomy-prior / readout
  layers are the remaining problem, not the projection.

This is a direct analogue of Gupta et al.'s null-space / pre-vs-post-projection
analysis, adapted to a supervised MIL head. It is cheap (one forward + ridge per
variant) and does not require end-to-end training.

### 2.3 Caveats

- The test only detects **linearly recoverable** information. If the projection
  destroys directions that are useful only non-linearly, a ridge test will miss
  that; but in this case the original trained head (which is non-linear after the
  projection) might still perform well, so the observed gap would point elsewhere.
- A negative result (`ridge(linear-proj) ≈ ridge(raw)`) is strong evidence *against*
  mechanism #1. A positive result supports #1 but does not rule out #2 operating
  simultaneously.
- To separate #1 from #2 cleanly, also run `SlotHead` with `prior = 0` and with the
  prior frozen during the original training. If the bottleneck test is positive
  but the prior ablation is flat, #1 is the actionable term.

---

## Bottom line for the arm

1. **Add normalization to the std branch before mixing.** Use `sqrt` as the default
   (matches VARPOOL code and B-CNN), with `log(0.01 + ·)` as a fallback. Measure and
   record the mean/std-branch magnitude ratio on one batch.
2. **Run the bottleneck diagnostic before burning GPU time.** Compare ridge on raw
   1024-d frozen features vs. ridge on the linear-only 256-d projection. If the gap
   is large, the 1024→256 bottleneck is the dominant failure and the fix is either a
   wider head or a mean-only baseline with std handled separately. If the gap is
   small, the problem is the attention/anatomy-prior/readout stack, not the width.
