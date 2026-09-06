# One more member cannot close the gap — the inversion, and what it costs to be wrong about it

*2026-09-06 03:22Z, while `chainfour_ff` held the card. Zero card, zero slots — CPU only.*

`kneemri/kaggle/seat_requirement.py` measures every priceable member that is not already seated
against the **optimal** seven-seat base
(`interpretations/kneemri/2026-09-06_the-ensemble-ceiling-is-not-a-selection-artefact.md`), on three
axes: what it scores alone, how far its rankings sit from the seated blend, and what it adds.

## The raw number, which needs no fit

```
181 candidates against a 7-seat base
  crop160_span_s2026      solo 0.8564  agreement 0.9114  gain +0.0002
  disagreed_ramp_s2026    solo 0.8574  agreement 0.9082  gain +0.0001
  through_dense_s2026     solo 0.8558  agreement 0.9102  gain +0.0001
  calib_span_hold_s2026   solo 0.8565  agreement 0.9142  gain +0.0001
  calib_span_s2026        solo 0.8545  agreement 0.9034  gain +0.0001
```

**The best eighth member this project owns adds +0.0002.** First place needs +0.0117 —
**74x** that. This is the whole finding, and it is a raw measurement over 181 members with no
modelling in it at all.

## The inversion, and why its own numbers should be distrusted

```
fit: gain = -0.0531 +0.0963*solo -0.0324*agreement  (rms residual 0.0003, best observed gain +0.0002)
TO ADD +0.0117 — 74.2x the largest gain ever observed here
  at the pool's median agreement 0.9073: solo must reach 0.9781
  at the pool's best solo 0.8635: agreement must fall to 0.5668
```

Both coefficients have the sign the mechanism predicts: gain rises with quality, falls with agreement.
Both requirements are far outside anything reachable — 0.9781 solo is +0.115 over the best single
model this project has ever trained, and 0.5668 agreement is a model that ranks studies almost
unrelatedly to the seated seven while still scoring 0.8635.

**Say the weakness out loud: the rms residual (0.0003) is LARGER than the largest gain being fitted
(+0.0002).** The plane is fitted through what is mostly noise, so the two coefficients are not
resolved and the exact requirement numbers are not quotable. They are an argument about direction and
order of magnitude, which is all a 74x extrapolation could ever be.

That does not weaken the conclusion, because the conclusion does not rest on the fit. It rests on the
+0.0002.

## The frontier, which is the part that was not guessable

The pool is NOT distance-free. Thirty of the 181 members sit below 0.90 agreement with the seated
blend, sixteen below 0.85, nine below 0.70. What they cost is accuracy, and what they add is negative:

```
the furthest member owned: focus_mcl_s2026 at agreement 0.0395, solo 0.5332, gain -0.0017
best solo owned inside each agreement band:
  [0.90, 1.01)  111 members, best solo 0.8635 (stack_increments_base_s2026, gain -0.0000)
  [0.85, 0.90)   40 members, best solo 0.8509 (base224_s2026,              gain -0.0001)
  [0.80, 0.85)   14 members, best solo 0.8337 (vjepa_warm_s2026,           gain -0.0007)
  [0.70, 0.80)    7 members, best solo 0.8255 (swin_distill_fit_s2026,     gain -0.0004)
  [0.00, 0.70)    9 members, best solo 0.7505 (radnet_pinned_s2026,        gain -0.0016)
```

Two things fall out, and neither needed the fit:

1. **Best solo falls monotonically with distance** — about 0.04 of score per 0.2 of agreement. Distance
   in this project has always been bought by giving up accuracy.
2. **Every band below 0.90 has a NEGATIVE best gain.** The far members are not underpriced diversity
   waiting to be seated; they subtract when seated.

That kills the naive reading of the inversion. "Get a member at 0.57 agreement" is not a plan, because
the frontier says a member at 0.57 agreement scores about 0.72 here, and the requirement was for one at
0.8635. The requirement is not merely extreme — it is **off the observed tradeoff curve**, roughly
+0.09 of solo above where 0.57 agreement has ever landed.

## What this closes

**The single-member route is arithmetically closed.** No member — trained, bought, or downloaded —
plausibly closes +0.0117 from a seat in this pool. Nor does a handful: members that each add +0.0002
also saturate against each other, which is the ceiling the previous interpretation measured three
independent ways.

This retires a class of plan that has been consuming card time all week: *train one more member, seat
it, ship it*. Seven board flats on seating operations already said this; now the pre-registration says
it before the card is spent, and says by how much (74x, not "a bit short").

## What it leaves

Two routes, unchanged from last night but now priced rather than asserted:

1. **Full-fit dose** — about +0.007 board, applying to whatever entry ships. `chainfour_ff` is on the
   card for exactly this. Production, not discovery, and it does not reach +0.0117 alone.
2. **A member that breaks the distance/quality frontier**, which is a narrower and harder thing than
   "a more diverse member". Distance alone is already owned and already priced negative. What is
   needed is a model that stays at the top of the score range while erring on DIFFERENT studies —
   +0.09 of solo above where this project's far members sit. Foreign trunks (~25) and medical SSL
   (four) were the two ways tried to buy that, and both bought distance by losing accuracy, which the
   frontier now explains rather than merely records: `vjepa_warm` at 0.8337/0.82 and
   `swin_distill_fit` at 0.8255/0.77 are exactly on the curve, not below it.

   The one mechanism that has never been tried and is not on the closed list is **external DATA** —
   a trunk continued on outside images rather than swapped for an outside trunk
   (`interpretations/kneemri/2026-09-02_the-external-corpus-seat-axis-closes.md` closes the SEAT axis
   and explicitly leaves this open). Whether it moves a model OFF the frontier rather than along it is
   unknown, and it is the only remaining question with a mechanism behind it.

## What would refute this

A member measuring **below** ~0.75 agreement against the seated seven while scoring above 0.85 solo —
a point above the frontier table, not on it. The best this project owns in that band is 0.8255. If
such a member ever appears, this
interpretation's arithmetic is the thing to re-run first, because a single point that far outside the
cloud would re-fit the plane it was taken from.
