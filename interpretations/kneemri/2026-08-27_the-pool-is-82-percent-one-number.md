# 82% of everything that separates our 105 members is one number, and the shared axis misses the deficit

2026-08-27, evening. Source: `python -m kneemri.evaluation.competence_axis --records <every holdout
record> --largest-pool`, a module written this session. 105 members over 909 shared studies. No card and
no slot — this is an SVD of arrays that already existed.

## The question, and the hypothesis it was written to test

`2026-08-27_every-no-card-axis-is-priced-and-flat.md` prices every seat permutation on this board between
−0.004 and +0.001, and nine seat candidates against the champion spread 0.0012 — a fifth of the 0.0062
arm-vs-arm floor. `seat_value` says which permutation is best; nothing said why they all land together.

`knee-54yp` pre-registered an explanation, from the through-plane reading earlier the same day: the pool
spans **one centre-versus-periphery trade-off**. `band_wide` against `band_dense` (`slices 16→20`, the
only diff) buys the centre and pays with the periphery for a mean of +0.0025 — a trade, not a gain. If
every member sits somewhere on that one line, no permutation of them can move the mean.

## The instrument

A members × findings matrix of holdout AUCs, **double-centred**, then its singular values.

By finding, to remove how hard each finding is. By member, to remove how good each member is — and that
second centring is the whole difficulty. A pool with one weak member and one strong one has a huge first
component loading every finding the same sign, and calling that a trade-off axis is reading member
quality back as pool structure. So quality is measured and reported before it is removed.

A first component always exists and always takes the largest share, so the share is read against a
**null that permutes each finding's column independently across members**: every finding keeps exactly
the spread it was measured with, and only the pairing between findings is destroyed. That is the null
the claim needs — not "members are identical", which nothing satisfies, but "members differ and the
differences do not line up". Per-cell AUC noise inflates the null bar exactly as it inflates the
observation, which is the right way round.

## What it reads

```
105 members x 12 findings, over 909 shared studies
member quality took 81.7% of the spread and was removed before the axis was read
component 1: 45.9% of what is left, against a null bar of 27.7%

loading (spread), most negative first:
  Synovitis -0.451 (0.0209)   Effusion -0.388 (0.0189)   Medial OA -0.198 (0.0119)
  Lateral OA -0.181 (0.0104)  Contusion -0.054 (0.0097)  PF OA -0.003 (0.0105)
  ACL +0.014 (0.0141)         Lateral Meniscus +0.028 (0.0183)   MCL +0.053 (0.0165)
  Medial Meniscus +0.215 (0.0167)   Fracture +0.316 (0.0209)   Baker's +0.649 (0.0300)
```

## Reading 1 — the flat seat search is explained, and the explanation is boring

**Four fifths of everything that separates these members is a single scalar: how good the member is.**
Only 18.3% of the spread is profile shape at all, and the shared axis inside that is 45.9% of it — so
the one trade-off direction in the pool carries about **8% of the total member-to-member variation**.

That is a complete answer to why nine candidates spread 0.0012. A pool whose members differ almost
entirely by uniform quality has nothing to trade: rank-averaging six of them picks six points on one
scale, and the sixth seat can only add what the first five already say slightly louder. No selector,
however clever, extracts diversity from a pool that does not contain any.

## Reading 2 — the shared axis is real, and it is NOT the one I predicted

The axis resolves against its null, so the pool does have one. It is a **fluid-and-inflammation axis**:
Synovitis and Effusion at one end with Medial and Lateral OA behind them, Baker's cyst, Fracture and
Medial Meniscus at the other. The `by structure scale` split reads focal −0.0103 against regional
+0.0144, which is a weak echo of the scale split rather than the split itself.

The centre-versus-periphery reading in `knee-54yp` is **refuted on its content**. That hypothesis was
built from one controlled pair (`band_wide` vs `band_dense`); the pair is still what it was, but it is
not the direction 105 members actually vary along. One pair generalised to a pool, and the pool
disagreed — which is why the instrument was worth building rather than assuming.

## Reading 3 — the actionable half: the deficit findings vary, and vary OFF the axis

The three findings the supervision gap resolved as our deficit —
`2026-08-27_the-supervision-gap-is-four-findings-not-an-average.md`, ACL, MCL and Lateral Meniscus —
carry loadings of **+0.014, +0.053, +0.028**. Essentially zero.

A zero loading has two causes and they say opposite things, which is why the report prints each
finding's own spread beside its loading. These three read **0.0141, 0.0165, 0.0183** — ordinary, right
alongside Fracture (0.0209) and Medial Meniscus (0.0167), well above Contusion (0.0097). So it is not
that members agree on them. **Members differ on ACL, MCL and Lateral Meniscus about as much as on
anything else, and that difference is orthogonal to the pool's one shared direction.**

Which means the variation there is idiosyncratic — member-specific rather than a property of any design
axis we have swept. That is exactly the variation a flat rank-average destroys, and exactly the
variation a selector aimed at those three findings could keep — the first thing this project has found
that a targeted seat choice could select ON. The ranking it licenses was run the same hour and is below,
along with the ledger's ceiling on what such a selection can ever be worth. Read both: the ranking is
real and its payoff is capped.

## What this does not say

It does not say the pool cannot be improved by permutation. A one-axis pool with members spread widely
along that axis still has a best point; this says the axis is thin, not that it is absent.

It does not say member quality is unimportant — it says quality is what our members differ BY, which is
a fact about the search that produced them, not about the design space. 105 members built by sweeping
recipes around one architecture will differ mostly in quality whatever the design space contains. The
honest form of the conclusion is: **we have searched one direction very hard**, and that is consistent
with the owner's standing suspicion that improvements lie in different architectures.

It does not price component 2. Only the first component was compared to a null; a second real axis may
exist and nothing here excludes it.

## The denominator, and what it excludes

103 of the 105 members are `_s2026` fold-0 runs. Only `merged_blend_cf` and `merged_dense_cf` carry a
holdout record at the champion's fit regime, because a member trained `--fold -1` holds nothing out by
construction — **the champion cannot be scored offline at all**, and five of its six seats are not in
this matrix.

That cuts both ways and both are worth stating. It means the 81.7% quality share is NOT inflated by the
`+0.004` the fit regime is worth — this is one regime, and the spread is real within it. And it means
nothing here can be turned into a seat swap against the champion offline: the ranking below names
members in the wrong regime, and crossing regimes costs more than the whole axis could pay.

## The ranking the reading licensed, and the mechanism in it

`--on ACL MCL "Lateral Meniscus"` — the three findings named in advance by the supervision gap, ranked by
residual competence (how much better a member is there than its own overall level predicts):

```
mae_matched            +0.0242
encoder_adapt          +0.0189
encoder_adapt_gentle   +0.0178
encoder_layer_9        +0.0171
band_density           +0.0166
lr2e-4_ws              +0.0158
```

**Four of the top five change the ENCODER rather than the geometry, the labels or the head.** Three are
explicit encoder-adaptation arms and the fourth is an MAE-pretrained matched member. `band_density` is
the first geometry arm and it is fifth. Nothing in the ranking's construction favours encoder arms —
they are a minority of the 105 — so the clustering is the finding.

This is the first evidence in this project that ties the resolved deficit findings to a mechanism we can
actually build. It says: the thin structures are reached by ADAPTING THE REPRESENTATION, not by
sampling more pixels (already refuted today) and not by more slices (a zero-sum trade, measured today).
It converges with the owner's standing suspicion that improvements lie in different architectures, and
with the forum's own claim that encoder SIZE does not help — size and adaptation are different axes.

It is a ranking and not a price. What a deficit-strong member is worth in the champion's roster is
`seat_value`'s question, and it cannot be asked offline against a `--fold -1` champion.

## The ceiling the ledger already put on this, which I nearly wrote past

`knee-q1pf` (2026-08-23, `weights_gap`) measured our members against the twenty published CC0 members on
these same 909 studies, and it prices the ENTIRE per-label weak-column programme at **+0.0043 on the
mean** — a quarter of the bar, and a ceiling rather than an estimate. Worse for this page: we beat the
published members on **ACL by +0.0292 and on Lateral Meniscus by +0.0712, the largest margin on the whole
table**. Two of the three findings this document calls our deficit are the two we lead the field on
hardest. Only **MCL is a genuine competitive deficit (−0.0346), and buying it outright is +0.0029.**

Both readings are true and they answer different questions. The supervision gap is ABSOLUTE — our model
against our own labels — and it is right that ACL/MCL/Lateral Meniscus is where the pixels lag the
reports. `weights_gap` is RELATIVE, and it says nobody else extracts those pixels either. **Absolute
headroom is not a competitive lever when the field shares it.** So the selection value of the ranking
above is capped by +0.0043 and realistically by MCL's +0.0029 — below the 0.0062 arm-vs-arm floor, let
alone the 1.5% bar. `focus_mcl` was already cancelled on precisely this arithmetic.

What survives the ceiling is not the ranking, it is the MECHANISM inside it. `knee-q1pf`'s own
does-not-rule-out list names ensembling and *"a genuinely different encoder"*, and the top four here are
encoder arms. That converges with `knee-wbwq` — `fitted_encoder` exists, 0 of 131 arms use it, and
`merge_adapters` doubles adaptation capacity at no serve cost — which is a member-QUALITY lever and is
not capped by a per-label ceiling. `warm_calib_dense_s2026`, the one warm-started member on disk, sits
10th here at +0.0116.

**So the deliverable of this page is not a seat swap. It is a second, independent route to the same
conclusion `knee-wbwq` reached from arm-script archaeology: adaptation of the representation is the
under-searched axis, and it is under-searched by 131 arms to zero.**

## What it does not say about the ranking

The three findings were named by a DIFFERENT measurement made before this matrix was built, which is the
only thing that makes the ranking honest — a subset chosen after seeing the matrix would be chosen and
scored on the same data. The module's `--on` help says so, because the next caller will be tempted.

It does not say the encoder arms are good members. `mae_matched` and `encoder_adapt` also sit at the
negative end of the shared axis, so they are strong on the deficit while being unremarkable elsewhere —
which is exactly the profile a diversity seat should have and exactly the profile the holdout mean
throws away.

## The board says the same thing in one line, and it was already on file

This page argues from an SVD that our pool differs by quality and little else. `comp/board.jsonl` argues
it from three submissions, and the arithmetic is blunter:

```
solo_v1     0.868   2026-08-12
solo_v4     0.914   2026-08-17
soloseat_v1 0.925   2026-08-26   ONE member, final_ws_e12
calibsix_v1 0.941   2026-08-27   six seats
```

Six seats buy **+0.016** over one member, so ensembling is a real engine and not a rounding error — and
it is also saturated for us, seven seatings inside 0.937–0.941. Against that, rivals self-report SINGLE
models at **0.926–0.942**. **Their singles are worth our whole ensemble.** Our best board-read member sits
at or below the bottom of their single range while our six-seat entry sits inside it.

That is the gap, in board numbers rather than inferred from a holdout: not how we combine members, but
what a member is. It converges with everything above — 81.7% of the pool's spread is quality, the axis
that remains is thin and points away from our deficit, and the members that lead on the deficit are the
encoder arms. Four independent routes, one conclusion, and it is the one `knee-wbwq` is already the
bead for.

Related: `knee-d500` (the slot-5 reasoning and its correction), `knee-54yp` (the refuted hypothesis),
`knee-q1pf` and
`2026-08-23_the_per_label_axis_is_worth_0.004.md` (the ceiling), `knee-wbwq` (the mechanism this
converges with), `2026-08-27_every-no-card-axis-is-priced-and-flat.md`,
`2026-08-27_the-supervision-gap-is-four-findings-not-an-average.md`, `member_distance` (the pairwise,
label-free companion this generalises).
