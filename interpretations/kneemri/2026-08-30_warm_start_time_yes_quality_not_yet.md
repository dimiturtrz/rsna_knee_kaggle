# One warm epoch is worth five cold ones, and the ramp costs 0.0072

**2026-08-30 · knee-wbwq · the run happened 2026-08-27; this is the first reading of it**

## What was on disk that nobody had read

`warm_calib_dense` ran to completion on 2026-08-27 and wrote `ARM DONE`. Its result then sat unread for
three days while the bead that commissioned it stayed open and its follow-up stayed unwritten. That is the
cheapest kind of loss in this project — a card hour already spent, and no reading taken off it.

It is also a recurring one. `roster_check` exists because 76 scoreable members were in no roster;
`convergence_check` exists because three killed processes entered the ledger as facts about
architectures. This is the same shape: a finished arm is not a read arm, and nothing was checking.

## The measurement

`calib_dense_hold_s2026` (0.8634) is the best solo member this project owns. `warm_calib_dense` restarted
from its own encoder — `--encoder-weights` and nothing else moved — and scored **0.8562**, losing 0.0072.

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | **6** | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cold, 4e-5 | .7515 | .8007 | .8280 | .8438 | .8544 | .8554 | **.8634** | .8620 | .8607 | .8596 | .8603 | .8598 |
| warm, 4e-5 | .8503 | .8469 | .8555 | .8510 | .8549 | .8526 | **.8562** | .8529 | .8540 | .8518 | .8515 | .8513 |

The peaks say the warm start lost. The **shapes** say two different things, and only one of them is a loss.

## The time claim is validated, by more than it asked for

The warm run's **first** epoch reads 0.8503 — a score the cold run does not reach until **epoch 4**.
`fitted_encoder.py` predicted the saving as "the first eight epochs, the ones that only re-learn what a
member on disk already knows." Measured on this recipe, one warm epoch is worth about five cold ones.

That half of knee-wbwq is settled and needs no further arm. Against 98 members carrying 36 recipe
skeletons — where the modal skeleton accounts for 31 of them, one decode path re-run with a different
loss or label table — this is the mechanism that stops each of those paying a full cold fit.

## The quality claim is refuted at this rate, and the suspect was named in advance

Epoch 1 **drops**, 0.8503 to 0.8469, and the run then oscillates in a 0.847–0.856 band for eleven epochs
without ever reaching the parent's peak. A converged trunk does not wander like that under a rate it has
already annealed through. It wanders when something pushes it off its optimum first.

`warm_calib_dense.sh` wrote the mechanism down before any of this existed:

> one_cycle ramps the backbone rate above 4e-5 before annealing, and a converged trunk can be walked off
> its optimum by the ramp. That outcome sends the next arm to a lower `--backbone-lr`, not to the
> graveyard.

A warm restart does **not** resume the optimiser. The schedule starts over and ramps to peak near epoch 2,
so the parent's annealed weights meet a rising rate on epoch 1. **The prediction and the observation are
the same epoch.** This is a pre-registered branch firing, not a disappointing number being explained after
the fact — which is the only version of this reasoning that is worth anything.

## Two corrections to knee-wbwq's own framing

**The capacity argument does not apply to every parent.** The bead prices warm starts partly on
`merge_adapters` folding a LoRA parent's correction into the base weights so `--lora-rank` can build a
*fresh* adaptation on top — doubled capacity at no serve cost. That holds for a LoRA parent. It does not
hold for a full fine-tune (`lora_rank 0`): there is nothing to merge and nothing is doubled, and the
mechanism reduces to a second anneal from a better starting point. The bead was stating one claim where
there are two experiments.

**A checkpoint is the peak, not the last epoch — which is why warm starting is worth more than time.**
`cnx_own_recipe_s2026.pt` stores epoch 6 / 0.8186, while that run's last epoch read 0.8130 after
declining five consecutive epochs. A warm start resumes from a state the parent *run* threw away. That is
a property of every member on disk, not a ConvNeXt quirk.

## What was armed off this, and what was not

**Armed.** `warm_calib_lowlr` — the pre-registered follow-up, one flag against the warm control
(4e-5 → 1e-5). And `cnx_warm_anneal` — the same ramp question on a **convolutional** trunk, because the
answer is not assumed to transfer. `FittedEncoder` was written for LoRA ViT members and had never been run
on a CNN; converting `cnx_own_recipe_s2026` wrote 378 tensors at variant `cnxbase`, `PretrainedEncoder.into`
neither refused the variant nor tripped the key check, and the smoke ran. The warm-start path is not
ViT-specific, which widens this bead from one family to every trunk with a peak checkpoint.

**Not armed, and explicitly so.** No warm start buys **seat diversity** — a warm member is correlated with
its parent by construction, and decode-path diversity is the axis measured to transfer at ~2.4× the board
slope (knee-nas5). Warm starts should replace *repeat* arms and never replace *new-mechanism* arms. Any
seat claim off either arm needs `record_mix` against the roster afterwards.

## A correction to something I said today

Reading ConvNeXt's three runs across-run, I claimed 4e-4 was the best rate ever tried and therefore that
the rate axis was open above it — an underfit trunk. The **within-run** trajectory refutes that: at 4e-4
the run peaks at epoch 6 and declines for five epochs, while one_cycle is already annealing. It is
overfit past its own halfway point, not underfit. A larger rate arrives there sooner, not higher. The
rate axis on that trunk is bracketed, and the arm I had drafted against it was deleted before it ran.
