# `stackfit` — the additivity test could not fail, and that is the finding

*2026-08-27 · bead `knee-orjq` (does a directional gain ADD?) · arm `experiments/arms/stackfit.sh` · gate on `knee-nu1m`*

## The arithmetic, before the number

The arm pre-registered its verdict as a *computed* prediction, not as absolute bands:

```
predicted = calib_dense_hold + (ema_dense − band_dense)
```

Filling it in from the ledger:

| term | run | holdout | source |
|---|---|---|---|
| control (term one alone) | `calib_dense_hold` | **0.8634** @ e6 | ledger row 4273 |
| term two, measured alone | `ema_dense` | 0.8588 @ e7 | ledger row 4578 |
| term two's own control | `band_dense` | 0.8578 | ledger row 309 |
| ⇒ term two is worth | | **+0.0010** | |
| ⇒ **predicted** | | **0.8644** | |

`stackfit` observed **0.8631** @ e7. Twelve of twelve epochs, `convergence_check` clean (it is in
neither the truncated nor the undertrained list), curve 0.7467 / 0.7994 / 0.8330 / 0.8502 / 0.8579 /
0.8618 / 0.8624 / **0.8631** / 0.8618 / 0.8621 / 0.8614 / 0.8609 — flat over the last five, so the peak
is not a truncation artefact. Both flags verified in the run's own recorded recipe:
`"calibrated_targets": "True"`, `"ema": "True"`, with `slices 20`, band `0.1–0.9`, `slice_pooling max`,
`whole_stack true`, `backbone_lr 4e-05`, `hold_annotations true` identical to the control.

**Observed − predicted = −0.0013.** That is inside the arm's ±0.0062 band, which is the row the header
labelled ADDITIVE — "the result the page needs. It licenses stacking as a METHOD."

**It does not license it, and the reason is in the table above.**

## Why the ADDITIVE verdict is unearned

The whole separation this arm was built to resolve is **+0.0010** — the distance between the additive
prediction (0.8644) and the null in which the second term contributes nothing and the control's 0.8634
stands. The decision band is **±0.0062**, the three-seed paired noise floor. The band is six times the
effect.

So the ADDITIVE region and the "second term did nothing" region are the *same region*. Any outcome that
reads ADDITIVE here also reads FLAT, by construction, and no run of this arm at any score could have
told them apart. The observed −0.0013 against predicted is also −0.0003 against the control: the number
is equally consistent with "the two levers added perfectly" and with "`--ema` is a no-op at this corner
and `--calibrated-targets` carried all of it".

This is a **design** fault, not a result. The arm chose its second term for portability (`ema_dense`
already existed, zero extra card hours) rather than for size, and a term smaller than the floor cannot
be the subject of an additivity test. The rule this buys, and it is the deliverable:

> **Never test additivity with a term whose solo effect is below the noise floor.** State
> `predicted − control` before choosing the terms and check it clears the band; if it does not, the
> experiment has no power and the ADDITIVE branch is unfalsifiable.

Stating the arithmetic before the number — which the header itself demanded — would have caught this
before the card was spent, because `ema_dense`'s +0.0010 was already on the ledger when the arm was
written.

## What the arm DID deliver, and it is the clause worth the card

The header's own justification for running it was the cheap kill: *"That KILLS the stack route, and it
kills it cheaply and early, which is worth the card on its own."* The negative branches are the ones
this design has real power against, because they are far from the null:

- **`≤ predicted − 0.0062` = 0.8582, SUBSTITUTES-NOT-COMPLEMENTS** — the `depth_regions_span` shape,
  where two levers answering the same question cost more together than either alone. Observed 0.8631 is
  **+0.0049 above** that line. **Refuted.**
- **`≤ band_dense_hold` = 0.8573, SUSPECT THE HARNESS** — both terms positive alone and negative
  together, which is a plumbing shape before it is a mechanism. Observed is **+0.0058 above** it, and
  the flags are confirmed present in the recipe stamp. **Did not fire.**

So: **stacking is not killed.** Two levers that each measured positive alone do not cancel when applied
together at this corner. That is a genuine, if weak, negative result, and it is the only claim this arm
is entitled to make.

## Consequence

- `knee-orjq` (do directional gains ADD?) is **not** answered by this arm. It is answered, at the level
  the project can currently afford, by the board evidence written into that bead on 2026-08-27: the
  climb is +0.022 over eleven days in steps none of which exceeds 0.004, and an independent offline
  split moved +0.0087 while the board moved +0.009. **Only the SUM is a fact; a single step is a ~2:1
  coin.** `stackfit` adds one datum consistent with that and resolves nothing.
- `knee-nu1m`'s gate on `stack3` **opens, with a condition**: run it only if every term it stacks has a
  solo effect above 0.0062. Stacking three sub-floor terms reproduces this arm's failure three times
  over and costs three card hours to learn nothing.
- `stackfit_s2026.pt` is a valid checkpoint at 0.8631, the second-best offline single member on the
  board behind `calib_dense_hold`'s 0.8634. `seat_value` could not price it against `calibsix`
  (five of six seated members are fold −1 and have seen every holdout study, so only one carries a
  holdout record and a marginal contribution needs at least two) — that is a limitation of the roster,
  not of the checkpoint, and it is why the seat question for this member is open rather than negative.
