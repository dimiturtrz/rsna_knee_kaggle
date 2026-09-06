# Their five folds are healthy offline — so the vote rule is the only suspect left for the 0.929

2026-09-06, ~20:03Z. Subject: the -0.013 unexplained drop of `foreignseat_v1` (0.929) against
`raptorweight_v1` (0.944, identical seven members). Zero card, zero slots — every number here is read
off arrays already on disk.

`kneemri-foreignlogit`'s refutation table pre-registered one offline explanation for a sub-0.938 read:
*"their five folds disagree in a way our own five-fold members do not, and that is measurable offline
without a slot."* This is that measurement, and it comes back **negative**.

## Their family, measured

| | band 0.16-0.84 | band 0.08-0.92 |
|---|---|---|
| per-fold mean AUC (all-fold, contaminated) | 0.8668 | 0.8646 |
| five-fold rank-mean | 0.8932 | 0.8924 |
| five-fold prob-mean | 0.8951 | 0.8945 |
| **gain from averaging the folds** | **+0.0264** | **+0.0278** |
| pairwise rank agreement, mean of the ten pairs | 0.7939 | 0.7879 |

Low pairwise agreement (0.79) **with** a large averaging gain (+0.027) is the signature of a healthy
ensemble, not a noisy one: the folds disagree and the disagreement is useful. A family whose folds
disagreed pathologically would show the spread without the gain. The pre-registered explanation is
refuted.

## Two false leads killed on the way, both worth writing down

**"The served geometry is the worst band."** The all-fold rank-means above (0.892-0.895) sit +0.012
over the shipped `foreign_dinov3_matched` record (0.8812), which reads like the served band 0.12-0.88
being a bad choice. It is not. Those are **all-fold** reductions and every study is in-sample for one
of the five folds; the fold-matched pooled numbers the sweep actually records are **0.8582** (0.08-0.92)
and **0.8601** (0.16-0.84) against **0.8812** for the shipped band. The shipped band is the best band
measured, by +0.021. The contamination gap between the two reductions is **+0.035**, which is the same
order as `raptor_solo`'s 98.6%-in-sample void and is the reason `foreign_price` exists.

**"The seat was priced on a different geometry than it served."** It was not: `logs/seat_weight_served.log`
prices `foreign_dinov3_matched.holdout.npz` — the served band, fold-matched — at every weight from 1x
(+0.0046) to 10x (+0.0194), 6x reading +0.0162.

## What is left

Against `raptorweight_v1` three dials moved: the foreign family entered at 6/13, the vote went
LogitAverage -> RankAverage, and the read groups collapsed 3 -> 1. The read collapse is already priced
at -0.002 by `onegroup_v1`. The family is now cleared offline on its own terms, the geometry is cleared,
the pricing file is cleared, and `run_check` cleared the mount (6 of 6, five folds `m_f0..m_f4`).

**The vote rule is the surviving suspect**, and `kneemri-foreignlogit` — byte-identical except
`LogitAverage`, already pushed — splits it for one slot and no card. Its branches stand as registered:
0.942-0.945 the vote carried the drop; 0.929-0.933 the seat carried it and +0.0178 is void as a planning
number; 0.934-0.941 they share it.

Not closed: the possibility that no offline ruler we own sees this axis at all, which is what a
0.934-0.941 read would mean and which no further offline work can settle.
