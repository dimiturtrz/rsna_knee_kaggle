# 87% of what is left is the label, not the model — and 79 members disagree less than 12 do

knee-qrtx, read 2026-08-31. `python -m kneemri.kaggle.champion_uncertainty`, commit `78a54aef`. No GPU:
numpy over the cached held-out probabilities that were already on disk.

## What was asked

knee-nj4l has said since it was filed that the plateau is a monoculture and *"is closed by a member whose
errors are somewhere else"*. That reading came from the entry side — seven differently-seated entries in a
0.004 band against a 0.0053 paired floor. It is an inference from a spread, and a spread that narrow is
also what a noisy ruler looks like.

The sibling project `cardiac-seg` hit the same wall on cross-vendor segmentation and did not settle it by
argument either. It decomposed its ensemble's predictive uncertainty and found only 15-18% of it was
epistemic, which is what licensed *"stop adding members, change model class"*. This is that measurement,
here.

## The decomposition

For K separately trained members with per-study, per-finding Bernoulli probabilities `p_k`:

```
total     = H[mean_k p_k]        what the ensemble does not know
aleatoric = mean_k H[p_k]        what no member knows — the label's own ambiguity
epistemic = total - aleatoric    what a DIFFERENT member could still fix
```

The epistemic term is exactly the K-way Jensen-Shannon divergence, which is why this reuses
`member_divergence.py`'s arithmetic rather than adding a second entropy helper — `H(p) = 1 bit −
KL(p ‖ 0.5)`, so the clamp keeps one home.

Bits are per study per finding, averaged over studies and then over the twelve findings — the same
per-label-then-mean shape `member_divergence` uses, because one number over the flattened array is
dominated by prevalence. Pooled is a ratio of means, never a mean of ratios.

## Measured

**`honest12`** (0.937, 12 members), 909 studies every member held out:

| | total | aleatoric | epistemic | reducible |
|---|---|---|---|---|
| ACL | 0.5818 | 0.4851 | 0.0967 | 16.6% |
| MCL | 0.5511 | 0.4762 | 0.0749 | 13.6% |
| Medial Meniscus | 0.6425 | 0.5402 | 0.1022 | 15.9% |
| Lateral Meniscus | 0.6197 | 0.5267 | 0.0930 | 15.0% |
| Medial OA | 0.6594 | 0.5786 | 0.0808 | 12.2% |
| Lateral OA | 0.6581 | 0.5836 | 0.0744 | 11.3% |
| PF OA | 0.7157 | 0.6185 | 0.0971 | 13.6% |
| Effusion | 0.7396 | 0.6437 | 0.0959 | 13.0% |
| Synovitis | 0.7717 | 0.7131 | 0.0586 | 7.6% |
| Baker's | 0.6422 | 0.5613 | 0.0809 | 12.6% |
| Contusion | 0.6315 | 0.5477 | 0.0838 | 13.3% |
| Fracture | 0.5753 | 0.5270 | 0.0483 | 8.4% |
| **pooled** | **0.6490** | **0.5668** | **0.0822** | **12.7%** |

**The whole roster** — 79 members (17 `_ff`/`_cf` fold-−1 members dropped and named), same 909 studies:
pooled total 0.6807, aleatoric 0.6242, epistemic 0.0565, **reducible 8.3%**.

Every member is one separately trained checkpoint. No TTA windows, no view variants — weight sharing
would make the epistemic term a lower bound for a reason that has nothing to do with the pool.

## The reading

**Adding members to this pool goes the WRONG WAY.** Twelve members disagree over 12.7% of their
uncertainty; seventy-nine disagree over 8.3%. Sixty-seven additional members did not widen the pool's
belief, they concentrated it. That is knee-nj4l's monoculture claim in bits rather than in entry spread,
and it is a stronger statement than the entry side could make: the 0.004 band could have been a noisy
ruler, and this cannot.

**It also closes the other branch, the one I opened this morning.** The hope was that a low ceiling would
at least implicate the COMBINER — that members disagree and a rank average throws it away. They do not
disagree. There is 0.0565 bits of disagreement in a 0.6807-bit pool for any combination rule to exploit.
A better vote is bounded by the same wall as a better seat.

**A ceiling, stated honestly.** 12.7% epistemic does not mean a new member can win 12.7% of anything on
the board. It means that is the whole budget a new member is competing for, before any of it converts to
AUC. Every member arm this project has queued — including the one on the card right now — is drawing on
that budget.

## The caveat that is actually a lead

`aleatoric` is what no member knows GIVEN THE LABEL. Our labels are `report_labels_v2.csv`: LLM-read
findings from radiology report text. So the 87.3% is not cleanly "the imaging is ambiguous" — it is
imaging ambiguity **plus label noise**, and nothing here separates them.

That distinction changes what to do. If the mass is genuine imaging ambiguity, the task is near its
ceiling and the remaining game is calibration and vote mechanics. If a material share is annotation
error, then **label quality is the untouched lever**, and it is one no architecture search can reach —
a better trunk fitted to a wrong label learns the wrong label better.

We have the instrument to bound it: 58 gold studies with radiologist annotations. Fifty-eight × twelve
is too thin for a per-finding split, but it is not too thin for a pooled one, and the direction of the
gap between gold-aleatoric and report-aleatoric is the whole answer. That is the next read, and it costs
no card.

## A caveat the field supplies, added the same day

`arXiv 2508.04457` benchmarks thirteen uncertainty-quantification methods for MULTI-LABEL chest X-ray
classification and reports that **"almost all methods show significant correlations between EU and AU"**,
that many methods fail to disentangle the two at all, and that the information-theoretic framework used
here should be applied "with caution". That is the same decomposition this page runs.

Two things keep the reading standing. Ours is computed over a **deep ensemble** of separately trained
members, which is the method class that won that benchmark on every axis it measured — the degeneracy it
reports is worst for single-model evidential heads. And the number is used as an **upper bound on what
another member can buy**, not as a per-cell signal, which is the use least sensitive to the two terms
bleeding into each other.

What the caveat does change is the wording. "87.3% aleatoric" should be read as **"at most 12.7% is
addressable by adding members"**, not as a clean partition of the world into imaging ambiguity and model
ignorance. The bound is the claim; the partition is not.

## One thing this could not measure

The pinned champion `raptorweight_v1` **carries no split at all**. One of its six members has a holdout
record; `calib_span_ff`, `final_ws_cf`, `grade_ladder_cf`, `lr4e-5_ws_cf` and `through_dense_cf` were
fitted at `--fold -1` and have seen every holdout study. The tool refuses rather than reporting a
contaminated number, and names `honest12` instead — the same refusal `foreign_overlap` makes, for the
same reason, after the same defect voided `raptor_solo`'s 0.8938.

The champion is not wrong because of this. But it is unmeasurable by every offline instrument this
project owns, and that is worth writing down separately from anything above.
