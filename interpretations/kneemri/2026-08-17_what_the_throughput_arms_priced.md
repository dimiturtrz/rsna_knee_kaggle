# What the throughput arms priced, and which of the three defaults they justify

*2026-08-17 · six one-epoch runs on `D:/data/.../rsna_knee_abnormality`, whole-stack at max pooling,
read from the heartbeat's last five `img/s` lines and the epoch row's `vram_peak_gb`.*

The measurement ran and the numbers were never written down anywhere — not the ledger, not a bead, not
an interpretation. This is that write-up. Nothing here is a method result: every arm scores inside the
0.0053 floor of every other (0.7539–0.7612 at one epoch), which is the point — these are prices, and
the score column exists only to show the price was not paid in accuracy.

| arm | steady img/s | vs its own control | epoch s | VRAM peak | epoch-0 holdout |
|---|---|---|---|---|---|
| `throughput_plain` (eager) | 815 | — | 365 | 14.56 | 0.7600 |
| `throughput_compiled` | 1195 | **+47%** | 272 | 15.03 | 0.7606 |
| `throughput_bf16` (eager) | 871 | +7% vs plain | 322 | 14.56 | 0.7589 |
| `throughput_ckpt_on` (batch 6) | 785 | — | 335 | **7.38** | 0.7612 |
| `throughput_ckpt_off` (batch 6) | 1016 | **+29%** | 292 | **26.24** | 0.7601 |
| `throughput_unfreeze4` (eager) | 1019 | +25% vs plain | 279 | 13.11 | 0.7539 |

**THE RECIPES, NOT THE SCRIPT, SAY WHAT WAS MATCHED.** `throughput.sh` has been edited since these ran —
it now describes compiling as the default and `--eager-encoder` as the control, while five of the six
recorded recipes carry `compile_encoder False`. So the compile row is the only pair the script's current
text describes, and every other row is a comparison *among the eager arms*. Read any other way (say, bf16
against the compiled arm) the numbers say bf16 costs 27%, which is an artefact of comparing across the
compile axis rather than a precision result.

## The three prices

**COMPILING IS WORTH 47%, for 0.5 GB.** 815 → 1195 img/s on an otherwise identical recipe, at
14.56 → 15.03 GB peak. The ragged-batch worry the arm was written to test did not materialise: dynamic
shapes did not send it into recompile churn, and the holdout moved +0.0006. On a ~300 s whole-stack
epoch that is about 90 s an epoch, or an hour and a half across a twelve-epoch arm. Compiling being the
default (and `--eager-encoder` the escape) is the correct standing decision, and this is its evidence.

**ACTIVATION CHECKPOINTING COSTS 29% AND IS STILL MANDATORY.** At batch 6 where both sides fit, off is
1016 img/s against on at 785 — and it holds 26.24 GB against 7.38, a 3.6x memory factor. The shipping
recipe trains at batch 12, so the off-side memory would be roughly 52 GB on a 32 GB card: the tax is not
optional at our batch, it is what buys the batch. The literature's "about 30% compute" that the
docstring quoted turns out to be almost exactly right for this workload, which is worth knowing, because
it means the estimate can be trusted for the next capacity decision instead of re-measured.

**BF16 IS +7% AND THAT IS NOT WHY IT IS ON.** 815 → 871 img/s against the fp16 control, both eager, with
the holdout unmoved (−0.0011, inside the floor). Blackwell's matmul throughput is the same either way,
so the gain is the loss scaler falling away rather than faster arithmetic. bf16 is insurance against
scale collapse that happens to be marginally faster — the standing preference for bf16 training and
fp16 serving is unchanged by this and unopposed by it.

## The one arm that is a question, not a price

`unfreeze4` is 25% faster than its eager control and lands 0.0065 BELOW it at one epoch. That is inside
the floor and unreadable as a score — one epoch is not where capacity decisions are made — but it is the
only row here whose direction is negative, and 25% is large enough that it deserves a real arm rather
than a shrug. Our own evidence that capacity is not the constraint (base over small, +0.0011 against a
0.0053 floor) predicts opening four blocks instead of six should be close to free; this run neither
confirms nor refutes that. FILED as a question, not adopted: the default stays six.

## What this changes

Nothing about the model. Two things about how time is spent: a twelve-epoch whole-stack arm costs about
55 minutes compiled where it cost 73 eager, and the checkpointing tax is now a known 29% rather than a
literature guess, so the next VRAM-versus-throughput decision has a measured exchange rate to argue
from.
