# Embedding statistics are a null, and the two controls say why

**2026-09-08.** Measured by the researcher (D212) against a question this file's owner asked directly:
*"embeddings metrics — mahalanobis distances, knns from predictions, pca results and stuff like that.
im going to expect we know nothing and we never bothered trying."* We had not. Now we have, and the
answer is a null with a mechanism, which is a stronger thing to write down than "we never tried".

## The frame

Every number below is `@holdout909`, cut **midpoint** (`kneemri/evaluation/scores.py:21 MIDPOINT = 0.5`
— a number in this repo is midpoint unless it says otherwise). Loss is the per-study mean BCE. All 118
seated members are joined; the ensemble is their mean. The statistic under test is per-study, and it is
scored by Spearman against the ensemble's per-study loss: a statistic that could route, triage or
reweight would have to order studies by how badly we do on them.

**Pre-registered bar: |rho| >= 0.15 to be worth building on.** Stated before the run, not after.

## The result: five statistics, five nulls

Median over the 9 complete caches:

| statistic | rho vs ensemble loss | verdict |
|---|---|---|
| `maha` — Mahalanobis distance to the training mean | **-0.0100** | null |
| `pca_resid` — residual off the training PCA basis | **+0.0151** | null |
| `nn_dist` — distance to the nearest training study | **+0.0504** | null |
| `knn_truth` — label mean of the 32 nearest training neighbours | **+0.0491** | null |
| `knn_pred` — deviation from the neighbourhood's predicted mean | **-0.1126** | null, and backwards |

The best single cache anywhere in the sweep is `maha` on `vjepa2l` at **+0.1236** — still under the bar,
and it is the max over 45 cells, which is where a spurious best comes from.

## The controls are the finding

A table of nulls is not an explanation. Two controls run alongside it are:

**1. The features carry the signal, and the members have already taken it.** `knn_truth` against a
study's positive MASS reads **+0.5202**. So the 32-nearest-train-neighbour label mean does predict what
is wrong with a study — the frozen trunk features are not blind to the finding. They are **redundant**
with what the members already extract. That is the difference between "no signal in the embedding space"
and "no signal LEFT in the embedding space", and it is the whole reason the axis closes: a statistic
cannot route on information the thing being routed has already used.

**2. Half the per-study loss ordering is just prevalence.** `rho(positive mass, ensemble loss)` =
**+0.5022**. A study with more findings is a study we lose more on, and any per-study statistic that
correlates with prevalence will inherit a correlation with loss for free. **Every future per-study
statistic must be scored against this control**, or it will report prevalence and be believed.

**3. `knn_pred`'s sign is backwards, and the sign is the mechanism.** Deviation from the neighbourhood's
predicted mean anti-correlates with loss (-0.1126): studies where the ensemble departs from its
neighbours are the ones it gets RIGHT. It is a confidence proxy on easy studies, not an error detector.
Filed here because the owner's standing correction applies — an anticorrelation is not a result to take;
if it were exploitable we would train a classifier to be wrong. It is a description of what the
statistic measures, and what it measures is not error.

## What this does and does not close

**Closed: the proposal.** Any routing, triage, reweighting or OOD-gating rule built on these statistics
would have to be FITTED, and the only labelled frame we own is this one. There is no signal here to fit.
The queue dies; this file is where it is written down.

**Not closed: the claim "OOD scores cannot work on the real test set."** This holdout is drawn from our
own corpus, so every study in it is in-distribution, and the owner's standing point stands — *the real
kaggle test dataset is NOT our dataset*. This measurement cannot see a shift it does not contain. What it
rules out is fitting a rule here and shipping it, which is the only thing we could have done.

## One identification result worth keeping

Raw Mahalanobis is **not identified across trunk width**: Spearman(median d², width) = **+0.8216**, a
6.61x spread (201.1 to 1329.7), which is exactly `E[d²] = p` under a Gaussian. Comparing raw d² between
two trunks compares their widths.

Under **common-rank(64)** it is identified in scale (**+0.0822**, 1.28x spread) and in ordering:

- matched control (one cache split into two random 512-coordinate halves): **+0.9429** — the ceiling
- cross-trunk, equal width: **+0.6673** (n=12)
- cross-trunk, unequal width: **+0.6986** (n=24)

Unequal width agrees slightly MORE than equal, so width is not what separates the orderings. Keep this:
it means a future distance statistic taken across trunks is comparable if and only if it is common-ranked,
and that is a free fix rather than a caveat.

## Two cache defects found on the way

Both are repo defects the sweep surfaced, fixed the same day (`knee-` beads filed 2026-09-08):

1. **Truncated caches read as healthy.** `orthodiff_336px_12sl_r256_gap256_volume` holds **284 of 4407**
   studies, `large_336px_12sl_r252_cls_mean_focal_max` holds **504** — contiguous prefixes from study 0,
   0 interior breaks, and `exists()` said True. `build` allocates the whole features file up front with
   `open_memmap(mode="w+")` and writes mask and index last, so a REBUILD that dies leaves the previous
   build's mask and index beside a part-written array. Fixed with a zero-cost oracle: **index mtime older
   than features mtime ⇒ incomplete**. Ledger check: both truncations are 2026-09-04 rebuilds and the only
   probe on record naming these stems ran 2026-09-02 22:22Z, with no probe artefact written since — **no
   recorded result is contaminated.** The exposure was prospective.
2. **The mask docstring was backwards.** It claimed an unacquired slot's features are zeros. The mask
   reports 20130 slot-instances present while the features are non-zero at 26442 of 26442: an absent slot
   holds the encoder's ordinary response to a BLANK IMAGE. The mask is more load-bearing than the wording
   implied — an unmasked mean ingests blank-image features for 24% of slot-instances, invisibly.

## What it cost

One researcher sweep, no card time, no submission slot. That is the right price for an axis that closes.
