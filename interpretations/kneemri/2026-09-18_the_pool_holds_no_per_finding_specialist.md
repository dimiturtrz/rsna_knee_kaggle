# The pool holds no per-finding specialist — 2026-09-18T01:34Z

## The question

`member_candidates.md` asks whether the 374 members that answer our holdout's question already contain
the "different member" — one that is mid-table overall while genuinely leading a finding. Until now that
document answered it with a flat per-finding rank: twelve columns, twelve leaders, no noise ruler. A
leader is the maximum over the pool on the same studies that price it, so every one of those margins is
selection-optimistic by construction and none of them was decidable.

`seat_profile --holdouts <dir> --against headlr_ship_s2026` now prints each leader's margin over the
shipping solo with the 95% interval of that margin (paired bootstrap over studies, `paired_interval`).

## The table

Pool: 431 of 450 records after `members.declines` void; **374 of 431** share `headlr_ship_s2026`'s
population (same studies, same table that graded them — the digest knee-g2vm9 put into `population`).

```
finding                leader                          best  spread    margin       95% interval  clears
Effusion               headlr_floor_s2026            0.9432  0.7631    0.0016 [ -0.0025,  0.0057]  no
Synovitis              fieldwide_s2026               0.9291  0.7346    0.0113 [ -0.0019,  0.0246]  no
ACL                    resnet_224_long_s2026         0.9518  0.6210    0.0097 [ -0.0042,  0.0252]  no
Lateral Meniscus       labelwise_drop_drop40_long_s  0.9039  0.5264    0.0057 [ -0.0066,  0.0189]  no
MCL                    warm_calib_lowlr_drop40_long  0.9121  0.5163    0.0302 [  0.0030,  0.0590]  YES
PF OA                  labelwise_depth24_drop40_lon  0.9021  0.5162    0.0002 [ -0.0083,  0.0091]  no
Lateral OA             labelwise_ext_drop40_long_s2  0.8820  0.5036    0.0125 [ -0.0013,  0.0273]  no
Fracture               merged_blend_cf_r392          0.8769  0.4877    0.0120 [ -0.0202,  0.0463]  no
Baker's                foreign_labels_dense_s2026    0.9317  0.4665    0.0124 [ -0.0032,  0.0293]  no
Medial OA              foreign_labels_dense_s2026    0.9298  0.4436    0.0096 [ -0.0004,  0.0203]  no
Contusion              labelwise_drop_drop40_long_s  0.9093  0.3297    0.0033 [ -0.0057,  0.0135]  no
Medial Meniscus        small_batch6_s2026            0.9135  0.2690    0.0062 [ -0.0044,  0.0166]  no
```

## The reading, and why the one YES does not survive either

Eleven columns close one-sided and need no further argument: the best member in the pool is not
demonstrably better than the member we ship, on the studies that priced both.

The twelfth is MCL, `warm_calib_lowlr_drop40_long_s2026` at +0.0302 with a lower bound of +0.0030. It
does not survive, for two reasons that are arithmetic rather than judgement.

1. **Twelve columns at 95% expect 0.6 false clears.** One clear out of twelve is what chance delivers
   before a single mechanism is invoked.
2. **The leader is a maximum over 374 members**, so the interval is the interval of a pair NAMED IN
   ADVANCE applied to the winner of a 374-way search — exactly the trap `paired_mean_interval`'s own
   docstring names. A 95% bound priced for one comparison has no standing after 374.

And its size is what the column's own scarcity predicts: MCL's interval is 0.056 wide, the widest but
one in the table, because it has the fewest positives. A wide interval is how a column produces a big
margin by accident, not evidence that a big margin is real.

**So the honest answer is zero of twelve.** The pool does not contain a per-finding specialist over the
shipping solo.

## What that closes, and what it points at

- **The pool-diversity search is CLOSED as a source of the next member.** Hunting through 374 existing
  records for a column edge has now been priced with a ruler and returns nothing. This is the
  one-sided close the screen was bought for: a failure to clear is valid even though a clear would not
  have been.
- It does **not** say the members are interchangeable — `spread` is 0.27 to 0.76 per column, so they
  disagree enormously. It says the disagreement is smaller than the studies it is read on, which is the
  ruler-ceiling result (offline ranking is exhausted above offline75 ~0.904) arriving on the per-finding
  axis instead of the macro one.
- The next member has to be **BUILT** on a mechanism argued in advance, not selected out of what exists.
  A selection over 374 cannot produce evidence; a designed arm with a pre-registered branch can.

## What was wrong on the way here

The first run of this screen printed twelve margins that sat OUTSIDE their own intervals — MCL read
+0.0302 against [-0.0590, -0.0030]. `paired_interval` reports SECOND minus FIRST and the pair was passed
the other way round. The point estimate falling outside its own interval is what caught it; the unit
test now asserts the interval brackets the margin, so the order cannot silently flip again.

That is the same family as the ruler defects this project keeps paying for: the number was computable,
plausible, and about a different quantity than the column it was printed in.
