# Q-EK — how do you price a seat you cannot measure?

Claude's champion entry `raptorweight_v1` has six members, five of which
were trained with `fold -1` (no holdout).  The project's `seat_value`
tool refuses to price them because a marginal contribution needs at least
one holdout prediction.  This note surveys what the literature and Kaggle
practice say about pricing full-fit/refit members.

## 1. Is there a standard way to price a full-fit member's marginal contribution?

**The textbook answer is out-of-fold (OOF) prediction.**  In stacking and
ensemble selection, each base learner is trained in a cross-validation
scheme and its predictions on the held-out folds are used to both (a)
score the learner and (b) feed the meta-learner or set ensemble weights.
The member's marginal value is then computed from those OOF predictions
(e.g., leave-one-model-out improvement on a holdout set).

When a model is later refit on the **full** training data for submission,
the OOF predictions are no longer available for the refit version.  The
standard Kaggle workaround is:

- Price the member using its **CV-fold version** (the fold-0 sibling, or
the ensemble of all CV folds).
- Refit the selected members on all data, keeping the **weights/order**
from the OOF pricing.
- Assume the relative ordering of members is stable under refitting.

This is a pragmatic convention, not a theorem.  I did not find a paper
that formally measures how often the OOF ordering survives a full-data
refit.  The closest foundational result is Caruana et al.'s *Ensemble
Selection from Libraries of Models* (ICML 2004), which selects and
weights models using a validation/holdout set and refits the ensemble.
However, Caruana's library consists of independently trained models, not
CV-fold siblings of a single refit model, and the paper does not isolate
the "ordering transfer" question.

## 2. How large is the full-fit advantage itself, published?

**I did not find a published curve that gives the gain from refitting on
the held-out fold as a function of fold fraction.**  The relevant
literature is fragmented:

- **Bates et al., *Cross-validation: what does it estimate and how well
does it do it?*** (JASA 2023, arXiv:2104.00673) proves that CV estimates
the average error of models fit on other training sets, not the error of
the model fit on the data at hand.  They propose a nested cross-validation
variance estimator, but they do not quantify the *gain* from refitting on
the full data.

- The gain can be bounded by a data-scaling argument.  In k-fold CV, the
refit model sees a factor of `k/(k-1)` more data than each fold model.  If
performance follows a power-law scaling `err(N) ∝ N^{-β}` (the usual deep
learning scaling-law form), the expected improvement from refitting is
roughly proportional to `β · log(k/(k-1))`.  With `k=5`, the data increase
is 25%; with typical vision-model β values, the expected AUC improvement
is often in the **low single-digit thousandths**, not the +0.007 Claude
observes between `honest12` (0.937) and `raptorweight_v1` (0.944).  That
suggests the observed +0.007 is not mostly a refit effect; it is seating,
diversity, or both.

So there is **no direct published measurement** of the refit advantage as
a function of fold fraction.  The scaling-law estimate says it should be
small at our corpus size.

## 3. Does a nested/repeated scheme buy the measurement back cheaply?

**No cheap unbiased scheme exists, and the banned option (re-running folds)
is the only one that is theoretically clean.**  Some possibilities and
why they fall short:

- **Influence functions / training-data attribution** (Koh & Liang 2017,
Hammoudeh et al. 2024 survey) estimate the effect of removing individual
**training samples**, not the effect of removing or adding an ensemble
member.  They do not price member-level marginal contributions.

- **Bagged surrogates.** One can train a small bag of cheap models on
subsets and use their disagreement as a proxy for the full-fit member's
value.  This gives a diversity estimate, not a seat value, and it
confounds model quality with model correlation.

- **Fold-0 sibling pricing.** If a full-fit member has a sibling trained
with fold 0 held out, the sibling's OOF predictions on fold 0 can be used
as a surrogate.  This is the standard Kaggle hack, but it prices the
sibling, not the refit member.  Its validity rests on the same
unmeasured "ordering transfers" assumption.

- **Nested cross-validation** would give an unbiased estimate, but it
requires training the full roster multiple times — exactly the compute
cost Claude wants to avoid and the project bans for other reasons.

## Bottom line

1. **Standard practice:** price full-fit members through their fold-0 / OOF
   siblings and assume the ordering transfers.  This assumption is
   widely used but, as far as I can tell, **not formally measured** in the
   literature.
2. **Refit advantage:** no published curve.  Scaling-law reasoning
   suggests it is small (low thousandths of AUC) at this corpus size, so
   the +0.007 champion gap is unlikely to be mostly refit gain.
3. **Cheap unbiased measurement:** none exists.  Influence functions are
   for samples, not members; bagged surrogates measure diversity, not
   seat value; nested CV costs a full retrain.

The honest ledger entry is therefore: the champion's seats are priced by
convention, not by measurement, and any swap among full-fit members is a
guess whose expected value cannot be bounded with the existing evidence.
