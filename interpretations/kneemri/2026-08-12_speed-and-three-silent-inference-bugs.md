# What a day of measuring our own tooling cost, and what it bought

2026-08-12, second half. No method changed and the holdout did not move. Two things happened anyway:
an arm got 2.4x cheaper, and three bugs were found that had already spent a submission between them.

## 1. Every speed number we believed was inherited, and two of the three were wrong

| lever | what we believed | what it measures here |
|---|---|---|
| `torch.compile` | "unavailable on Windows" | **+48% img/s**, once `dynamic=True` was dropped |
| gradient checkpointing | "about 30% more compute" | **19%** — and it buys 3.5x the memory |
| bf16 | "same speed, insurance only" | +6%, holdout unmoved |
| 12-epoch schedule | the recipe | 8 epochs scores the same, at two-thirds the cost |

**Compile had never worked, for a reason nobody had checked.** Triton is not shipped for Windows, so
`--compile-encoder` failed at the first backward on every run since the flag was added. Installing
`triton-windows` was necessary and not sufficient: inductor still raised `AssertionError` inside its own
size reasoning, with checkpointing and without it, and the cause was `dynamic=True` — which the
docstring had justified by arguing that a static graph would recompile every epoch, because the last
batch of an epoch is partial.

That argument was wrong in a way that could have been checked in a minute: there are TWO shapes in a
run, not one per epoch, and each compiles once per process. Better still, the second shape did not have
to exist — training now drops the partial batch, so the encoder compiles once and stays compiled. The
cost of dropping it is bounded and knowable: the order reshuffles every epoch, so a study is seen 11.98
times over twelve epochs instead of 12.

**The checkpointing tax was a literature number.** Measured at the only batch that fits both ways: 798
img/s against 946, so 19% rather than "about 30%". Its other half was never measured either — 8.78 GB
against 2.50 GB at batch 2, which is 3.5x and the entire reason whole-stack reading is possible at
batch 12 instead of 3.

**Utilisation is not the metric.** The card reports 97-99% whether or not checkpointing is on, because
util means "a kernel was resident", not "the work was useful". Checkpointing does MORE work at the same
utilisation. Power (96% of budget) and img/s are the honest signals; util is the one that looks like an
answer and is not.

## 2. Three inference bugs, and the one that scored 0.500

**fusion v2 scored exactly chance, and the model had nothing to do with it.** Three failures composed:

1. `predict_member` sent the whole decoded chunk through the encoder in one call. A chunk is 48 studies,
   but what reaches the encoder is studies x slots x window width — 864 images for a windowed member and
   **3,456** for a whole-stack one. Four times the peak activation, decided by a member field the kernel
   could not see. Invisible on a 32 GB card under a VRAM cap; an OOM on the kernel's 16 GB T4.
2. The run therefore wrote no submission of its own.
3. `Submission.fallback()` had already copied `sample_submission.csv` into place — "so a crash later
   still leaves a scoreable file". That constant table was scored. One of five daily slots, spent to
   learn nothing, reported as a number that reads like a model result.

Compute was never the problem, and that is what rules out a timeout: reading whole is CHEAPER than ten
overlapping windows, 12 images against 30. The entry was faster than the 0.894 one right up to the
single forward that asked for four times the memory.

The fallback is deleted. A kernel that fails now produces no `submission.csv`, Kaggle fails the
submission itself, and that costs **no slot** — a fallback converts a loud failure into a quiet wrong
answer, which on a five-a-day budget is the more expensive of the two.

**The fix then had the next OOM already loaded.** Bounding the forward by images used one constant for
every member, which is the same mistake one constant for every depth was:

| encoder | activation/image | at 864 images | T4 (~15 GB) |
|---|---|---|---|
| small (384) | 0.0099 GB | 8.6 GB | fits |
| base (768) | 0.0187 GB | 16.5 GB | no |
| large (1024) | 0.0235 GB | 21 GB | no |

864 is validated by every scored entry — all of whose members are `small`. `large_deep_rank24` sits in
our pool at 0.8450 and has never been submitted. The budget now divides by encoder width.

**A third, caught before it shipped.** Adding DINOv3's ViT-S+/16 broke mount matching silently: `vits16`
is a prefix of `vits16plus` and both are 384 wide, so each encoder claimed the other's directory and a
member built on the wrong weights would have loaded with no shape error at all. Matching is now by whole
path segment.

## 3. The shape of all of it

Every one of these is the same defect this project keeps producing: **a value that travels per run, read
by a consumer that assumes a constant.** The image budget assumed one encoder width. The window assumed
three slices. The mount assumed a substring. `dynamic=True` assumed a recompile per epoch. The
checkpointing docstring assumed a number from someone else's workload.

The fix that generalises is not "check more carefully" — it is that the value has to be READ from the
thing that owns it: the member's own width, the encoder's own patch size, the backbone's own config,
this card's own measurement. Where that is impossible, the constant needs the measurement written beside
it, so the next person to trust it can see what it was measured on.

## 4. What it bought

An arm that cost 66 minutes this morning costs ~31 now, and a DINOv3 question at 224 px costs 16. That
is the point of the day: none of it moved the score, and all of it moves how many chances we get to.
