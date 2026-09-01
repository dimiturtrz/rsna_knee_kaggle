# Where the published field is ahead is exactly where our members disagree — knee-y7v3

This is a join, not a new measurement. Both columns already existed and neither was built with the other
in mind:

- **The deficit** is `research/2026-08-23_published-per-label-auc-vs-ours.md` — our annotation-ruler AUC
  over the 58 gold studies against the best published per-label number, on other people's corpora.
- **The unanimity** is `2026-09-01_a-quarter-of-the-residual-is-unanimous.md` — the share of the honest12
  pool's errors that all twelve members make, on our own 909-study fold-0 holdout, report ruler.

They share only the name of the finding. Different corpora, different ground truth, different metric,
different year.

## The join

| finding | published deficit | unanimity |
|---|---|---|
| Lateral Meniscus | **−0.20** | **13.3%** |
| ACL | **−0.10** | **14.4%** |
| MCL | −0.04 | 24.4% |
| Medial Meniscus | −0.02 | 22.8% |
| Synovitis | −0.02 | 25.0% |
| Contusion | +0.02 | 25.3% |
| Medial OA | +0.02 | 27.5% |
| Effusion | +0.08 | 29.7% |

Spearman **0.952** (p = 0.00027, n = 8); Pearson 0.942. The ordering has one inversion in eight rows.

## It is not prevalence

The obvious confound is that both columns track how common a finding is. They do not:

- prevalence ↔ unanimity, over these eight: 0.452
- prevalence ↔ unanimity, over **all twelve** findings: **−0.105**
- deficit ↔ unanimity, **partialled on prevalence**: **0.981**

Controlling for prevalence does not weaken the relationship, it strengthens it. Over the full twelve-row
table prevalence explains essentially nothing about unanimity at all.

## What it says

**The two instruments agree on which findings still have headroom, and they agree for different reasons.**
An external ruler says the field is 0.20 ahead of us on Lateral Meniscus and 0.10 on ACL. An internal
ruler, which has never seen a published paper, says those are the two findings where our twelve members
are *least* likely to be wrong together — 13.3% and 14.4% unanimity against a 24.3% pooled floor. Neither
column could have produced the other.

**That is the strongest evidence this project has that the remaining gap is model-side and reachable.**
Unanimity is the part of the residual no reseating can touch; low unanimity means the members already
disagree, which means the information to be right is at least partly reachable by *a* model in this family
— and the published numbers say some model out there is in fact getting it right. Lateral Meniscus and ACL
are the two cells where "the field is ahead" and "we are not stuck at a floor" coincide.

**It points the middle band.** `residual_split` found ~70% of errors in a band that only a genuinely
different member can reach, and `population_split` found that band is not concentrated in any acquisition
population. This says *where in the label panel* it is concentrated. A new member should be built and
judged on Lateral Meniscus and ACL specifically, not on the twelve-label mean that would dilute a real
gain there by a factor of six.

**It re-reads knee-dtk's partition as confirmed a third time.** The 2026-08-14 model-bottlenecked half was
ACL, MCL, Medial Meniscus, Lateral Meniscus, Lateral OA, PF OA. The four of those with published
comparators are the four largest deficits here, and the two largest are the two lowest unanimity rows.

## What this does NOT establish

- **n = 8.** Four findings (Lateral OA, PF OA, Baker's, Fracture) have no published comparator and are
  absent from the correlation. A rank correlation on eight rows with one inversion is suggestive; it is
  not a fitted law, and no number should be predicted from it.
- **The deficit column is confounded at source** and says so: different corpora, prevalence, inclusion
  criteria, and a competition that grades ambiguous findings negative where no cited paper does. The
  58-study annotation ruler carries a per-label SE of 0.04–0.06, so the −0.02 rows are not distinguishable
  from zero. The correlation is carried by the ends of the range, which is where the ruler is strong.
- **Two rulers agreeing is not two rulers being right.** Both could be tracking a third thing — how
  intrinsically hard a finding is to see on MRI, say — which would produce this exact pattern while making
  neither column a statement about headroom. What the join rules out is the *specific* alternative that
  the deficit is a prevalence artefact.
- **It names no mechanism.** It says a new member should be aimed at Lateral Meniscus and ACL. It does not
  say what that member should be, and nothing here licenses a submission.
