# Warm-starting was refuted on its SCHEDULE, not on its mechanism, and the corrected arm has never run

2026-09-17. Bead knee-2co4r. No card spent: a read of `kneemri.experiments.recipe_pairs` over the run
records already on disk, plus a grep of `experiments/arms/`.

This answers a question the owner asked directly and which has until now only ever been answered in
conversation: *"you are warm starting right? ... my reasoning is that the model is somewhat good -
maybe in global min/max, maybe not. the reason it didnt progress might be the training parameters, not
that the model as a whole is saturated."*

**The answer, stated first: yes, once, and it lost — but it lost to a rising learning rate on an
annealed trunk, which is a schedule defect its own arm script predicted in writing before it ran. The
arm that follows from that refutation, a warm start at a LOWER `--backbone-lr`, has never been armed.**
The rest of the corpus's warm-start evidence is flat and was taken far below the recipe we ship.

## The three facts

**Twenty-six arms carry `--encoder-weights`. Zero of them sit at the shipping corner.**

```bash
grep -lE "encoder.weights" experiments/arms/*.sh | wc -l                      # 26
grep -lE "encoder.weights" experiments/arms/*.sh \
  | xargs grep -lE "slice-pooling labelwise" | xargs -r grep -lE "epochs 24"  # empty
```

**Where it WAS tried, it is flat — and the sign leans against it.** `recipe_pairs --key
encoder_weights` resolves six single-key isolating pairs, **none clearing the 0.0062 paired floor**.
Four of the six compare a warm start against a cold one (`None`), which is the comparison the question
is about:

| warm start | ledger75 | against | ledger75 | delta to COLD | runs |
|---|---|---|---|---|---|
| `swin_distill_pre.enc.pt` | 0.8507 | cold | 0.8537 | **cold +0.0030** | `swin_distill_fit_s2026` / `swin_supervised_s2026` |
| `adapt_latent_v3small_m60_e8` | 0.8609 | cold | 0.8614 | cold +0.0004 | `latent_register_deep_s2026` / `v3small_control_s2026` |
| `swin_distill_pre.enc.pt` | 0.8507 | cold | 0.8509 | cold +0.0002 | `swin_distill_fit_s2026` / `swin_wholetop_s2026` |
| `adapt_latent_v3small_m30_e8` | 0.8619 | cold | 0.8614 | warm +0.0006 | `latent_register_dense_s2026` / `v3small_control_s2026` |

The remaining two pairs swap one warm start for a different warm start (+0.0059 and +0.0010) and say
nothing about warm-vs-cold.

Three of four favour COLD, the largest margin is 0.0030, and the floor is 0.0062. **Warm-starting, as
this project has done it, buys nothing measurable and may cost a little.**

## Why that is not yet an answer to the question

Read the ledger75 column. The four pairs sit at **0.8507–0.8619**. The shipping corner is **0.9053**.
Every warm-start measurement this project owns was taken **0.04 to 0.05 below the recipe we ship**, on
`swin` and `v3small` trunks that are not the one we ship either.

**But the corpus is NOT silent, and `recipe_pairs` cannot see the run that matters.** `warm_calib_dense`
(bead knee-wbwq, run 2026-08-27, read 2026-08-30) warm-started `calib_dense_hold_s2026` — **the best solo
this project owned at the time, 0.8634** — at the parent's own 4e-5, and scored **0.8562, a loss of
0.0072**. It is absent from the six pairs above because it isolates no single key against a matched
sibling, which is exactly the hole a pair table has: the one warm start aimed at a good model is the one
the instrument drops.

Its two trajectories say opposite things and both are load-bearing:

```
cold 4e-5  .7515 .8007 .8280 .8438 .8544 .8554 [.8634] .8620 .8607 .8596 .8603 .8598
warm 4e-5  .8503 .8469 .8555 .8510 .8549 .8526 [.8562] .8529 .8540 .8518 .8515 .8513
```

The warm run's FIRST epoch reads 0.8503, which the cold run needs four epochs to reach — one warm epoch
is worth about five cold ones. Then epoch 1 DROPS and the run oscillates 0.847–0.856 for eleven epochs
without ever reaching its parent's peak. **The arm pre-registered that failure and named its cause:
`one_cycle` does not resume the optimiser, it restarts and ramps to peak near epoch 2, so an annealed
trunk is hit with a RISING rate on epoch 1.** Prediction and observation are the same epoch.

So the honest statement is not "warm-starting a good model is untested" — it is **tested once, refuted
once, and the refutation is of the SCHEDULE rather than of warm-starting.** The bead's own conclusion
stands: that outcome sends the next arm to a lower `--backbone-lr`, not to the graveyard. That arm has
never run.

That is the same shape as the labels finding written the same day
(`2026-09-17_labels_is_the_most_priced_leaf_and_the_corner_has_tried_three_schemes.md`): an axis swept
thoroughly on a family we have since left, and its deltas inherited by a corner that never took them.
The difference matters here because the owner's mechanism is specifically about the corner — a model
that is *already good* sitting in a local optimum, where a fresh schedule over good weights could move
what a cold restart cannot. **The corpus contains no test of that.** The 26 arms warm-start models that
were not good yet.

## What it would cost to answer, and why it is not armed

The arm is not "warm-start the corner". It is **warm-start the corner at a rate the annealed trunk can
survive** — `labelwise_depth24_drop40_long` with `--encoder-weights` from the 0.9053 checkpoint and a
`--backbone-lr` well under the parent's 4e-5, which is the arm knee-wbwq's refutation asks for and
which has sat unarmed for three weeks. At the corner's 24 epochs that is ~1.6 h of card.

Note what this is NOT in conflict with. The corner's rate axis is closed UPWARD — `lr8e5_ship_s2026`
doubled the shipping rate and read −0.0022, refuted by its own train loss. That says nothing about a
LOWER rate on a WARM trunk, which is a different starting state; the two are only confusable if rate is
read as a knob rather than as a property of the trajectory it enters.

Price it against the inversion in
`2026-09-17_the_solo_transfer_fit_predicts_the_board_to_the_thousandth.md`: a solo needs **+0.0089
offline75** to reach the 0.955 goal, and **+0.0021** to beat the champion once refit. The warm run's
own trajectory is what prices this: it held 0.847–0.856 for eleven epochs, i.e. it never escaped the
basin it started in, so the mechanism on offer is "reach the parent's basin in one epoch and spend the
remaining twenty-three annealing inside it" rather than "find a better basin". That is a plausible
+0.002-class lever and an implausible +0.009 one. **So it is priced as a champion-beat attempt, not as
a goal-reaching one, and the honest odds are near the coin flip the fit already describes at that
margin.**

## What this does NOT license

- It does not say warm-starting is dead. It says the four measurements are flat, were taken on other
  trunks well below our corner, and that the corner is untested. Per the standing rule, a refuting
  experiment kills that USE of an asset, not the asset.
- It does not say the model is saturated. `warm_calib_dense` tested the local-optimum hypothesis ONCE
  and could not separate it from its own rate ramp, so the hypothesis is **confounded**, not refuted.
- It does not say `recipe_pairs` is wrong. It says a pair table answers only about runs with matched
  single-key siblings, and the most informative warm start we own has none — so **an axis can read
  "6 pairs, 0 clearing" while its decisive run sits outside the table.** Check the beads for that axis
  before pricing it off the pair count.
- The deltas are ledger75 on population `66c6f231` over 284 priced recipes. They are not comparable to
  a number read on another population or at the midpoint cut.
