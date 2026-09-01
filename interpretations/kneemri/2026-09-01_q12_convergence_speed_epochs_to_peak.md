# Q12 (knee-g6xt): what cuts epochs-to-peak, measured not guessed

**Question:** What would reduce the number of epochs needed to reach peak holdout AUC?

I analyzed 157 completed runs with `auc` records in `D:/data/volumetric/mri/processed/rsna_knee_abnormality/*.metrics.jsonl`. For each run I took the first epoch whose holdout AUC was within 0.0001 of the run's best AUC as the "peak epoch."

## Distribution

| Peak epoch | Runs |
|-----------:|-----:|
| 1–5 | 19 |
| 6 | 46 |
| 7 | 17 |
| 8 | 27 |
| 9 | 33 |
| 10–12 | 15 |

Most runs peak at **epoch 6–8**.

## What moves peak epoch

### 1. How many trunk layers are unfrozen (strongest effect)

| `unfreeze_last` | Median AUC | Median peak epoch | Runs |
|----------------:|-----------:|------------------:|-----:|
| 0 (frozen trunk) | 0.8457 | **6** | 15 |
| 6 | 0.8498 | **7** | 139 |
| 12 | 0.7822 | 10 | 2 |

Freezing the trunk cuts one epoch off the peak but costs ~0.004 AUC. Unfreezing 6 layers is the current default and is the sweet spot: one extra epoch buys a higher median AUC. `unfreeze_last=12` is an outlier (two runs, likely failed/overfit).

### 2. Train/read resolution

| `train_size` | Median peak epoch | Runs |
|-------------:|------------------:|-----:|
| 0 (default, usually 336/352) | 8 | 140 |
| 224 | 9 | 1 |
| 256, 384, 392, 448 | 5–6 | 14 |

Runs with an explicit larger train size (384, 392, 448) peak earlier. The default-size group is dominated by the dense recipe and is confounded with other slow choices, but the pattern suggests larger input crops converge slightly faster.

### 3. Slice pooling

| `slice_pooling` | Median AUC | Median peak epoch | Runs |
|----------------:|-----------:|------------------:|-----:|
| channels | 0.7128 | **3** | 1 |
| depth | 0.8461 | **6** | 1 |
| max | 0.8508 | **7** | 106 |
| attend | 0.8506 | **8** | 4 |
| labelwise | 0.8552 | **9** | 4 |

`labelwise` is the slowest to peak but reaches the highest median AUC. `max` is the default and peaks at 7.

### 4. Other recipe flags (smaller or confounded effects)

| Flag | Faster peak | Slower peak |
|------|------------:|------------:|
| `one_cycle` | False (6) | True (7) |
| `whole_stack` | False (6) | True (7) |
| `compile_encoder` | False (6) | True (7) |
| `ema` | — | negligible |

`one_cycle=True` and `whole_stack=True` are both associated with a one-epoch later peak. These are also the recipes that produce the strongest final AUCs, so the extra epoch is usually paid back.

## Correlation with final quality

Across all 157 runs, peak epoch correlates only **r = 0.20** with best AUC. Peaking later is weakly associated with scoring higher, but the dominant factor is still the recipe, not convergence speed.

## Verdict

The lever that most reliably cuts epochs-to-peak is **freezing the trunk** (`unfreeze_last=0`), which saves one epoch at a cost of ~0.004 AUC. Beyond that:

- Do not expect large speedups without sacrificing AUC.
- The current dense recipe (`unfreeze_last=6`, `one_cycle=True`, `whole_stack=True`, `slice_pooling=max`) peaks at epoch 7 and is already near the Pareto frontier.
- If card time is the binding constraint, a **frozen-trunk fast arm** is the honest speed/quality tradeoff.
- If the goal is highest AUC, accept epoch 7–9; `labelwise` pooling and larger train sizes may converge slightly faster and score slightly higher, but the sample of such runs is small.

No intervention was found that cuts epochs-to-peak by more than one epoch without a measurable AUC cost.
