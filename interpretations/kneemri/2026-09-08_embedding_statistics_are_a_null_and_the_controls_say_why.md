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

## The follow-up that made this worth the sweep (D214, same day)

The redundancy control above says the members have already taken what the features carry. The obvious
next question is where they have NOT — a study the frozen features get right and the fitted members get
wrong would be the cleanest instance of failing to generalise this project can produce. Pre-registered
with three bars; **all three cleared, and two unregistered controls dismantled the clear.**

- The intersection of the nine per-cache subsets is 166 of 909 against a pre-registered bar of 18. But
  nine RANDOM neighbour subsets intersect at **106.6 ± 3.7**, so the feature-attributable part is
  **+59.4 studies, not 166** — 64% of the headline was a fact about the ensemble.
- On that subset the kNN is **flat**: 0.6027 there against 0.6039 on the complement, corpus 0.6037. It is
  equally mediocre everywhere. The subset is not where the features are good; it is where the ensemble
  drops below a constant predictor's line. The reading the bar was built to produce is false.
- The sharpness-immune sign test has a null of **0.5, not 0** — a random 32-neighbourhood's expectation is
  the base rate. Measured null 0.487. Pooled real **0.375**, z **-4.66**, ten of twelve labels below the
  null, label 7 at -6.39 on 1611 cells. The apparent best cell (label 10, 0.518) is z +0.53, a coin flip.

**So the redundancy is worse than total.** On the positives the ensemble misses, the frozen-feature
neighbourhood is on the correct side LESS often than chance. The members and the features share a blind
spot and the features are *anti-informative* inside it — the fitted heads are in fact slightly better than
their own substrate there, which is what a trained head is for. Not a prevalence artefact: intersection
positive mass 4.030 vs complement 4.310.

**This is the load-bearing consequence for the plan.** The failure to generalise is not a head-fitting
problem and cannot be fixed downstream of the trunk: a better head, a better pooling, a better seat
selection, or any reweighting over these nine trunks **cannot reach these studies, because the
information is not in the nine trunks.** What could reach them is a representation that separates them —
different pretraining, a different input (resolution, slice span, series), or supervision that forces the
axis these encoders collapse. That is a ranking of the open levers, produced for no card time.

Also worth carrying: the standing owner correction applies to the sign result. An anticorrelation is not
a result to take — the take-home is the blind spot, not a rule that bets against the neighbourhood.

## And the blind spot is not on any input axis we control (D217, same day)

D214 says the blind spot is in the representation. The cheap hope was that it is concentrated on
something a FLAG could fix — too few slices, the wrong band, an unoccupied slot, a protocol we
under-sample, a resolution we resample badly. Twelve axes were pre-registered with their bars, their
permutation floor and their null control BEFORE the sweep ran (D216), and the answer is **flat**.

The statistic is `E(s) = W(s) − W_rand(s)`, the count of `wrong` cells per study (12 labels, averaged
over the 9 complete caches) minus the same count under a random 32-neighbourhood. E and not W, because
**the random neighbourhood accounts for 81% of W** — W is bounded almost entirely by how many positives
the ensemble missed. `E mean 0.1001, sd 0.4077`; 310 of 909 studies have E > 0. Everything below is
`@holdout909/base-rate` — the ensemble-miss cut is the base rate, not this repo's default midpoint.

Bars: continuous `|rho| ≥ 0.10`, categorical `η² ≥ 0.01`, above a 2000-shuffle Bonferroni floor
(α = 0.05/12); CONCENTRATION Cohen's d ≥ 0.30 on the top decile of E; and a SPECIFICITY veto —
if `|rho(axis, W_rand)| ≥ |rho(axis, E)|` the axis is measuring the difficulty channel, not the features.

**The largest |rho| anywhere on the four real axes is 0.0645, against a bar of 0.10 set deliberately
low.** Slice count total, deepest-series depth, series count, slot fill, weighting combination,
fat-suppressed fraction and count, pixel spacing, minimum pixel spacing, slice spacing — nothing clears,
and the nearest thing to a signal (`slice_spacing_mm`, d_top −0.263) points backwards from the coverage
story and is vetoed because W_rand tracks it harder.

**What this costs the plan: the next move is in the weeks bucket, not the flag bucket.** No card arm goes
to `--slices`, the band, or slot routing on the strength of a coverage argument. That is a real saving —
it is the arm that was about to be written.

Three things worth keeping from the sweep independent of its null:

1. **E is decontaminated of prevalence and NOT of difficulty.** `rho(positive mass, E) = −0.0635`
   (against +0.1034 on W and +0.2027 on W_rand — the subtraction worked), but
   **`rho(missed positives, E) = +0.3388`.** No future reading may call E a feature-only quantity.
2. **The dominant flag setting means materially different things across the corpus.** With `--slices 20`
   and band 0.1–0.9 (196 and 198 of ~300 arms respectively), the implied stride runs
   **min 0.60 · median 1.40 · max 12.80 — a 21.3x spread**, and **128 of 909 studies have stride < 1.0**,
   meaning the sampler fills its 20 slots with duplicate slices. That is a description of our sampling, not
   a lever: `rho(stride, E) = −0.0193`, vetoed. It is also not an independent test — `stride = deepest ×
   0.04` is an affine rescale of depth, and rank correlation is invariant under it, so registering it added
   nothing to k. Rescaling a variable cannot disambiguate it from a confound; the prevalence control and
   the W_rand veto are what do that.
3. **`Fat_Suppression` is a duplicate column.** `population_split.py:16-18` finds it equal to
   `Fluid_Sensitive` on **24,371 of 24,371** rows, so every reading anyone took off the shipped column was
   reading the other one. The header-derived flag is genuinely different and genuinely varying —
   fat-suppressed series present in **902 of 909** studies, `frac_fatsat` median 0.600, and **no study is
   entirely fat-suppressed** — and it is flat against E. Two more columns the headers fill that the
   shipped table does not: `pixel_spacing_mm` 909/909 against 64/909, `slice_spacing_mm` 909/909 against
   159/909.

**The null control fired and it was not the retraction condition.** D216 pre-registered retraction as
*everything fires including axis 5*. What happened is the opposite: nothing fired except one column
inside axis 5. That column was `source` — how laterality was DETERMINED (`tag` / `geometry` /
`unresolved`) — which is header completeness, and header completeness is a vendor fingerprint (Siemens is
77% of `tag`; `tag` holds zero Canon studies). Conditioning on scanner kills it: pooled within-scanner
η² **0.0097** against 0.0164 unconditioned, below the 0.01 bar. `side`, the actual laterality and the
actual null, is flat (η² 0.0024, vetoed). **A mis-registered column is a correction, not a retraction.**

**What it does NOT say:** that the hard set has no structure. It says it has no structure on these twelve
axes at these bars. A DIFFICULTY axis is what this design deliberately could not see, since E retains
+0.34 coupling to missed-positive count.

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

## Where that leaves the levers, after the audit these two results forced

D214 named three ways to reach the blind spot. Two of them are already spent on this ledger, which is
only visible if you read it rather than re-deriving the list:

- **Different pretraining is spent at the trunk-swap level.** RadImageNet DenseNet, `radnet_pinned`
  (0.7505), RadioDino b16/s16/s8 — `radiodino_corpus` at 0.8214 moved the pretraining CORPUS with the
  architecture and objective held fixed, and priced its seat at **−0.0008** on honest12 despite carrying
  the strongest prior any non-DINOv2 candidate has had. MedSAM, MRI-CORE, Swin, ViT-in21k, ConvNeXt,
  ResNet-50d, CoatNet are all on the ledger too.
- **Different input is what D217 measured flat.**
- **Supervision is spent at one attach point, and explicitly reserved at another.** `contrast_dense`
  (SupCon at batch 12) read −0.0087; `contrast_pool` (SupCon + a remembered pool, 96 keys per anchor)
  read 0.8515, −0.0119 against the no-contrast control, **seat value 0.0000 to four decimals**, with the
  training loss moving exactly as widening a log-sum-exp denominator eightfold must. That closes the lane
  on MECHANISM rather than on price. Its own closing paragraph reserves one cell: *a representation term
  that shapes a different space, or attaches at a different stage.*

**Both contrastive arms attach at the head's per-finding contexts — the readout. D214 measured the blind
spot in the frozen TRUNK feature space.** So the one cell these results jointly leave open is a metric
term attached at the pooled trunk embedding, in the same space the 32-neighbour statistic was computed
in — a single-variable change against `contrast_pool` whose variable is the attach point. Gated, before
any card is spent, on a card-free reachability probe: if a fitted metric over the trunk features cannot
lift the 32-neighbour sign rate off its 0.487 null on held-out `wrong` cells, no term attached there has
anything to descend.

## The gate fired, and the last reserved cell is closed too (D219–D221, same day)

The probe above was run. Everything here is `@holdout909/base-rate` — the `wrong` cut is the
ensemble-miss cut, not this repo's default midpoint. 64-dim trunk PCA, 9 caches, 12 labels, logistic
`max_iter 2000` fitted on 3,498 cells; wrong-cell n 426–508 per cache (pooled 4,335), right-cell 263–345
(pooled 2,604).

**`LIFT = R_fit − R_null = 0.1190 − 0.1390 = −0.0201`, against a pre-registered bar of +0.05.** The
fitted metric recovers *less* of the blind spot than the same probe fitted to permuted labels.

**The null is the whole lesson, and without it this would have been written up as a 12% recovery.**
`R_null = 0.1390` is not noise, it is selection: `wrong = miss & (knn <= base)` picks cells *on* the
neighbourhood pointing the wrong way, so the sign rate there is 0.000 by construction and **any**
perturbation flips some back by regression to the mean. A probe fitted to shuffled labels is such a
perturbation. Every future probe on a selected subset must carry a permuted-label null or it is
measuring its own selection.

**The null was then matched on churn rather than assumed comparable**, because a shuffled fit and a real
fit need not perturb equally. Interpolating `w(a) = (1−a)·1 + a·w_null` over nine points, `a = 0.995`
lands at churn 0.3478 against the fit's 0.3479 — matched `R_null = 0.1283`, **matched LIFT −0.0093**.
Clearing the bar would need matched `R_null ≤ 0.0690`: missed by 1.9x in the null and ~6x in the lift.
Three readings of the same quantity, worst to best for the idea: −0.0196 unmatched, −0.0047
linear-rescaled, **−0.0093 measured**. Recovery is **sub-linear in churn** — the curve bows above its own
chord by +0.0150, and half the final recovery (0.0744 of 0.1386) is bought by the first 38% of churn — so
**a linear rescale of a null is the charitable assumption**, under-charging it for churn it never spent.
Measure the crossing; do not rescale.

**And the ceiling says redundancy, not silence.** With `wrong` as the positive class the probe's AUC is
**0.2928** pooled, all nine caches 0.219–0.327. That is 0.21 from chance **on the far side**: a high probe
score predicts a cell the ensemble gets *right*. AUC 0.50 would have meant the trunk features are silent
about these cells, which would have left room for a different objective to find something there. 0.29
means they speak clearly and say exactly what the 32-NN already says. There is nothing left for a term
attached in that space to descend.

**The reserved cell was not open in the first place.** Read off the run logs rather than the arm scripts
(the arms delegate to `$RECIPE`): `contrast_pool_s2026.full.log` and `contrast_dense_s2026.full.log` both
carry `unfreeze_last 6`, `backbone_lr 4e-05`, `layer_decay 1.0`, `lora_rank 0`. **`contrast_pool`'s trunk
was trainable**, and the contrastive gradient already flowed through the same last block the pooled
embedding is taken from. Seat value 0.0000. The "different attach point" was the attach point already
used. Caveat kept and not converted into a reservation: the bottom half of the trunk moved in neither arm.

One consistency check worth keeping: `radnet121` is the lone cache that inverts on both churn diagnostics
(PR 16.9 fit vs 28.9 null; churn ratio 1.125), which is what a real per-cache effect looks like against a
pooled statistic.

**Closed on four independent reasons** (bead `knee-osnmj`, closed, no card spent): the lift fails by ~6x
under the most favourable churn matching; the probe is redundant with the neighbourhood at AUC 0.29; the
most favourable case the attach point can ever see is the case it was measured on; and the arm that was
supposed to be its control had already made the change. **Killed:** any term whose effect is a metric over
trunk features, frozen or trainable. **Not claimed:** that supervision is exhausted at every attach point —
nothing here touches the bottom half of the trunk, or a term that acts before pooling.

Two method rules earned here, both general:

1. **A baseline must not be a deterministic function of the target.** A drafted follow-up fitted
   `error ~ member_score` where `error = (prediction < base)` inside a truth class — a threshold of the
   baseline itself, giving `AUC = 1.000` and `ΔAUC ≡ 0` before any data spoke. Print the baseline-alone
   statistic first and refuse a degenerate instrument.
2. **Do not pool across truth classes when the effect is signed.** A probe that scores positives low and
   negatives high cancels under pooling; a pooled `|AUC−0.5| < 0.05` bar would have closed a live axis on
   an averaging artefact.

## So: is the embeddings queue worth keeping?

No, and it is now closed with a mechanism rather than abandoned. Five statistics (Mahalanobis, PCA
residual, NN distance, kNN truth, kNN prediction) are nulls against a pre-registered bar; the control says
why (redundancy, not blindness); the blind spot that redundancy implies is flat on twelve input axes; and
a fitted metric over the same space cannot reach it either, by a factor of six, with the ceiling
confirming redundancy rather than silence. The one thing that survives is an identification result worth
reusing — a cross-trunk distance statistic is comparable if and only if it is common-ranked at rank 64.

What this does NOT license is "embeddings are useless". Every measurement here is fitted and scored on our
own corpus, so it cannot see a distribution shift it does not contain, and the owner's standing point
holds: *the real Kaggle test set is not our dataset*. What is ruled out is fitting a rule on this frame
and shipping it — which was the only thing that could have been done with one.

## What it cost

One researcher sweep, no card time, no submission slot. That is the right price for an axis that closes.
