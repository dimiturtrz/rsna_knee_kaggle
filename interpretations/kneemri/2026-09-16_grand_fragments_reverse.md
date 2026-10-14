# The grand recipe's fragment block is net harmful, and the one fragment that seemed to carry it reverses sign

2026-09-16. Beads knee-dcl6e, knee-3tk4b, knee-feb62.4, knee-feb62.11.

## What was measured

`experiments/recipes/trunks/probe224.toml` pins horizon (12 epochs) and geometry (12 slices, 224 px, band
0.2–0.8), so a pair stacked as

```
control  --config base.toml --config trunks/probe224.toml
grand    --config base.toml --config grand.toml --config trunks/probe224.toml
```

differs in `grand.toml`'s **fragment block alone** — `ema`, `warmup`, `patience_from_progress`,
`stochastic_depth`, `ghosting`, `intensity_aug`, `mixup`, `scale_jitter`, `padding_mode`, the `aug_*`
ramp and the `hard_*` ramp. Nine arms, all fold 0, all 909 holdout rows, priced with
`python -m kneemri.evaluation.ledger75 --holdouts D:/data/volumetric/mri/models/rsna_knee_abnormality`:

| arm | ledger75 | midpoint | vs control |
|---|---|---|---|
| `probe224_control` | **0.8914** | 0.8556 | — |
| `probe224_control_aug` (control **+** the aug block) | 0.8876 | 0.8528 | −0.0038 |
| `probe224_grand_norest` | 0.8825 | 0.8497 | −0.0089 |
| `probe224_grand_trim` | 0.8811 | 0.8494 | −0.0103 |
| `probe224_grand_noema` | 0.8808 | 0.8478 | −0.0106 |
| `probe224_grand_nomixup` | 0.8806 | 0.8492 | −0.0108 |
| `probe224_grand_nohard` | 0.8789 | 0.8462 | −0.0125 |
| `probe224_grand` (the full stack) | 0.8776 | 0.8445 | −0.0138 |
| `probe224_grand_noaug` | 0.8726 | 0.8386 | −0.0188 |

## The sign flip

Removing the aug block **from grand** costs −0.0050 (0.8776 → 0.8726), which is why the leave-one-out
priced it as the one fragment that carries. Adding the same block **to the control** costs −0.0038
(0.8914 → 0.8876).

A fragment cannot both be worth +0.0050 and cost −0.0038. It is not producing value; it is
**compensating for damage the rest of the stack does**. Read on its own denominator it is another
negative fragment, and every leave-one-out number in this family is an interaction term wearing a main
effect's clothes.

This is the `STATISTIC / POPULATION / CUT` rule one level up: the *baseline* a delta is taken against
is part of the reading, and `aug@grand` and `aug@control` are two different quantities.

## The fragments do not compose

The leave-one-out positives were `mixup` +0.0030, `hard` +0.0013, `ema` +0.0032 — +0.0075 predicted if
additive. `trim` drops the first two (+0.0043 predicted) and delivers **+0.0035**. `norest` drops four
more on top and delivers **+0.0049 in total**. Sub-additive throughout. `trim` and `norest` were run
precisely to answer additivity before a long seat was spent assuming it, and the answer is no.

## What the residual is

`norest` still carries the aug block (−0.0038 on its own denominator) and the three fragments nothing
ever varied — `ghosting` 0.04, `intensity_aug` true, `scale_jitter` 0.25, against the control defaults
0.0 / False / 0.08 read off `probe224_control_s2026.full.log`'s own recipe stamp. Those two together
account for the −0.0089 that `norest` still trails by. A confirming arm (`norest` plus those three at
control values, predicted to land on 0.8914) is an instrument at ~11 min of card and 0 expected score;
it is not needed for the decision, which −0.0138 already settles.

## What this does NOT prove

**Nothing here says the fragments are harmful at 36 epochs and 336 px.** `trunks/probe224.toml` pins horizon
and geometry, so every arm above varied fragments alone, at short horizon and small geometry — and
regularisation earning nothing at 12 epochs while earning at 36 is exactly what regularisation is for.

What it does prove is that the leave-one-out those fragments were justified by was reading interactions.
The justification is gone; the burden has moved. The cheap way to settle it is the same probe pair at
the long horizon with geometry still pinned — ~33 min an arm, ~1.1 h for the pair, to de-risk a 5.3 h
run. See knee-feb62.4.

## Related

- The champion family's own recipe at this scale IS `probe224_control` — `labelwise_dense_drop40_long`
  with the fragments off. It is the best arm in the table, by 0.0038 over the nearest thing to it.
- `base224_drop40_long` 0.8865 ledger75 / `_run1` 0.8855 ledger75 are one recipe at 24 epochs, two seeds: a 0.0010
  seed spread, which is the scatter every delta above should be read against. The −0.0138 is over ten
  times it; the +0.0049 `norest` recovers is about five.
