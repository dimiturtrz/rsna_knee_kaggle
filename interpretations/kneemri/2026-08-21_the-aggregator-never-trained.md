# The slice aggregator never trained, and a closed cell reopens

Found 2026-08-21 while checking that a NEW arm's mechanism had fired, not while investigating any of the
arms it invalidates. That ordering is the point: the defect is invisible to a score.

## What was wrong

`StudyModel.parameter_groups` built the optimiser's head group as `list(self.head.parameters())`.
`self.slices` — the slice aggregator — is a **sibling** attribute of the model, not a child of the head.
Its parameters were therefore in no param group, and the optimiser never updated them.

`SliceAttention`'s score layer is zero-initialised **on purpose**, so that a fresh gate starts as the
arithmetic mean over slices and training moves it away from there. A gate that never moves is a
permanent mean. So the failure mode was not a crash and not a bad score — it was a model that computed
something reasonable under the wrong name.

Proven mechanically rather than inferred: `parameter_groups` contained 0 of the aggregator's 6 tensors
for `attend` and 0 of 8 for `depth`, and a checkpoint dumped after a full epoch had
`slices.pool.gate.3.weight` at exactly `0.000e+00` — two independently-initialised tensors both exactly
unmoved is not a small gradient.

## What it invalidates

| row | recorded as | actually measured |
| --- | --- | --- |
| `slice_attention` 0.8473 (+0.0016) | attention over slices vs max | **mean vs max** |
| `attend_dense` 0.8569 (−0.0009) | a learned 64-wide gate vs max | **mean vs max** |
| `head_axes.jsonl` `slice_pooling` pair | the proxy's ordering vs the pixel model's | **a trained gate vs an untrained one** |

The last row is the subtle one. `head_trial` optimises `TrialHead.parameters()`, and its aggregator IS a
child of that object — so the frozen-feature proxy has always trained the gate the pixel model did not.
The two harnesses were not running the same function on the axis where they were being compared.

The checkpoints are real and the numbers are real; `attend_dense` stays a valid mounted member and the
champion package still reproduces, because `parameter_groups` is train-time only and a reloaded zero gate
computes the same mean it always did. What does not survive is the **mechanism sentence** attached to
`attend_dense`: "a strictly more expressive allocator does not move the score" was never tested, because
the allocator that ran was a fixed mean. The pooling-OPERATOR cell reopens, and with it the half of the
fixed-budget refutation that rested on it. `focal_dense`'s width axis is untouched and still reads −0.0044.

## The fix

The head group is now every parameter **not** in `self.backbone`, read by exclusion rather than by naming
the modules — naming them is what failed, and exclusion means the next sibling module cannot go missing
the same way. A regression test asserts, for every entry in `SLICE_POOLINGS`, that no trainable tensor
falls outside every group and that none is handed to two. Re-smoked: the gate, the mixer's neighbour taps
and its bias all move (2e-3 after one epoch on 48 studies), and holdout on the smoke went 0.5426 → 0.5578.

## What the cheap tier then said

The frozen-feature proxy, which trains the gate, ran all three poolings on the cached 12-slice features:

| pooling | proxy holdout mean AUC |
| --- | --- |
| max | 0.7605 |
| attend | 0.7632 |
| depth (neighbour mixing, then attend) | 0.7625 |

A genuinely trained gate is worth **+0.0027** over max there, and through-plane neighbour mixing on top
of it is worth **−0.0007**. Both are small on a ruler `head_axes.jsonl` itself records as unvalidated
(one resolved pair). That is not enough to spend 1.1 h of card on `depth_dense`, so the pixel arm queued
instead is `attend_true` — the 12-slice geometry where the pair's control already exists, run as an
INSTRUMENT: it either gives the proxy its second resolved pair or catches the proxy disagreeing with the
pixel model on the one axis where their optimisers differed, and either way it decides whether the
reopened cell deserves the 20-slice arm.

## The transferable lesson

The plumbing check that caught this — dump the checkpoint and assert the new mechanism's tensors actually
moved — was run because a NEW arm was about to take the card. Every arm that was affected passed all its
gates: the flag was in the recipe, the flag was in the checkpoint config, the module was constructed, the
forward pass used it. **Only the optimiser knew, and nothing asked it.** `arm_check` proves a flag was
named; `landed` proves a flag reached the config; neither proves a parameter receives gradient.
