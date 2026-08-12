# What the leaderboard taught us that the holdout could not

2026-08-12. Six entries were spent to isolate single variables, and three of the findings changed how
the project is run rather than adding a number to it.

## 1. Ensembling is the largest effect here, and DIVERSITY is what makes it work

| entry | members | holdout / member | public LB |
|---|---|---|---|
| our strongest single model | 1 | 0.8518 | 0.868 |
| whole-stack | 2 | 0.8481 | 0.882 |
| whole-stack | 3 | 0.8493 | **0.882** |
| ours, older windowed recipe | 5 | 0.8269 | 0.888 |
| the published CC0 members | 20 | 0.8398 | 0.888 |
| ours 3 + the published 20 | 23 | mixed | **0.894** |

The best single model this project has produced is **0.020 worse** than an ensemble of models that are
each far weaker. But the two rows that matter sit next to each other: a **third member of our own
recipe bought nothing** (0.882 → 0.882), while those same three members added to a **different recipe**
bought +0.006 over either component alone.

So it is not count. Members that fail the same way do not help each other. Our three whole-stack
members are three draws of one function; the published twenty are a different reading of the same
pixels, and that difference is the entire gain.

Measured offline on our own members alone, with no grafted weights, the same shape appears: best
subset 0.8475 at one member, 0.8552 at two, 0.8562 at three, 0.8567 at four, then flat. **+0.0092 for
four differently-failing members**, saturating exactly where the leaderboard saturated.

## 2. Whole-stack and overlap TTA are one mechanism bought twice

Each member scored twice — once by training at its best epoch, once at inference with overlapping
test-time windows:

| member | trained | with overlap TTA | gain |
|---|---|---|---|
| `lr5x_d6` — windowed | 0.8323 | **0.8475** | +0.0152 |
| `max12` — windowed | 0.8310 | 0.8451 | +0.0141 |
| `wsfold0_e6` — whole-stack | 0.8457 | 0.8457 | **0.0000** |
| `slices16` — whole-stack | 0.8452 | 0.8452 | **0.0000** |

Averaging ten overlapping three-slice windows shows the model every slice — in PREDICTION space, where
whole-stack does it in FEATURE space. Same mechanism, two implementations, and they do not compose:
the windowed lift (+0.0146) is almost exactly whole-stack's training-time advantage (+0.0147).

This re-prices the project's headline result. Whole-stack's +0.0147 was measured against a control
scored WITHOUT the trick that control gets at inference; given both their best readings, the windowed
member edges it (0.8475 against 0.8457). It also explains part of a leaderboard gap that had been
attributed entirely to ensemble size.

**What survives untouched is resolution.** `res392` scores 0.8523 with no TTA available to it at all,
still ahead of every windowed member with TTA. Resolution is a genuinely separate axis, and it is the
one training-side change that has not been explained away.

## 3. Seven bugs of one shape: a value travels in the config, a consumer reads a default

| what was recorded | what the consumer used | cost |
|---|---|---|
| `slices` | a hardcoded 12 | −0.018 when inference ran at 32 |
| `variant` | small, always | a silently wrong network |
| `slice_pooling` | channel stacking | every max-pooling member unsubmittable |
| `unfreeze_last` | a module constant | a fingerprint that lied |
| `whole_stack` | three-slice windows | our best members served as a different function |
| `img` at inference | `SlotReader`'s default 336 | latent; would have hit the 392 recipe |
| the slot scheme | the module's canonical six | 54 GB of zeros, twice over |

None of these fails loudly. The fingerprint cannot catch them because it pins image size and group and
nothing else — a member with the wrong depth, pooling, reading mode or scheme verifies clean and
predicts from an image it never saw.

Two structural answers landed. `Member.window()` is now the single definition of how a member is read,
used by inference and analysis alike, after three places decided it independently and each got it wrong
at least once. And every run now records a **witness** — what the training path predicted for eight
named studies — which the submission gate re-derives through the inference path, turning a structural
claim into an end-to-end one.

## 4. The noise floor is the holdout's sampling error, not the model's variance

Hanley-McNeil standard errors from the positive counts on an 881-study holdout predict a mean-AUC
floor of **0.0053**. The seed spread measured across three runs is **0.0062**. They agree.

The floor is therefore a property of the corpus, not of training: more seeds cannot lower it, because
the holdout is still the same 881 studies with the same 62 Fracture positives. Per-label precision is
far worse — Fracture ±0.061 at 95%, MCL ±0.040 — which refuted a P0 plan built on lifting our "two
worst labels", since those labels are weak on ONE fold and unremarkable on the others.

## 5. What is ours

**0.894 is not our number**: twenty of its twenty-three models are CC0 weights someone else trained.
Our own best is **0.888**. The finding in section 1 is ours, is reproducible without anyone else's
weights, and is what the remaining work is built on.
