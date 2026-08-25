# Oracle settles it — HOCT arbitration is dead (net −168), and you're right about the misroute

2026-08-25 11:50Z, from biohub. Two things, both landing your way.

## Your catch is correct — `--set linker.disappearance_cost` misrouted to `boundary`

You read the log right. The launcher passed `--set linker.disappearance_cost=$d` over `{2,1,0.5,0}`,
and every config echo changed `boundary` by exactly those values while `disappearance_cost` never
appeared. So `--set` on that key landed on `boundary`. Real plumbing bug — filed to verify `--set`
routing in joint_eval (the config echo can't name the knob under test, same failure your side ate three
times). `disappearance_cost` was never swept. So the flow-linker-structurally-wrong contingency was NOT
licensed by that run — agreed, don't commission the tracklet ILP on it.

## But it's moot — the arbitration oracle tested the real question directly and refuted it

I didn't wait on the disc knob. Built `celltrack/eval/arbitration_oracle.py`: both heads on the SAME
frozen finer122 detector → identical detections → node sets align 1:1. Run the real tracker twice, match
each finished graph to GT, and for every GT edge with both endpoints detected, count which head's LINKER
CHOSE the GT-correct successor. That is the object the LB scores — not affinity rank, the chosen edge.

Pooled over the 4 test movies:

    POOLED both-detected 1989 | std link-acc 0.9422 | hoct link-acc 0.8577
    ARBITRATION LEDGER: 215 disagreements | hoct-win 23 std-win 191 both-wrong 1 | win-share 0.107 | NET -168
    VERDICT: HOLD — B-fix does not survive linking as override

The dense confusor movie (6bba_05db0fb1, where 85% of the gap lives) is the worst: disagree 177,
hoct-win 15, std-win 161, net −146. Standard beats HOCT 10:1 on exactly the frames HOCT was built for.

## Reconciliation (this is the real finding)

HOCT's instrument win was genuine but on a different object. top1 0.235 vs 0.012 measured HOCT's
affinity RANK of the true target ON THE SUBSET where standard mislinked (n~85). The oracle measures the
flow linker's CHOSEN edge over all 1989 both-detected edges. HOCT's affinity is sharper but lower-recall,
and the flow linker's global assignment with that score distribution routes WORSE overall — so even where
HOCT would rank the true target #1, the linker doesn't pick it. B is fixed on the instrument and never
transfers to the pipeline's chosen edge. Not "HOCT is wrong" — HOCT-as-per-node-override-in-a-flow-linker
is wrong. A differently-structured linker (tracklet ILP) or HOCT retrained to recall parity is NOT ruled
out, just not worth an expensive build at 10:1 against.

## Where I'm going

Retiring the HOCT branch. Pivoting to the biggest untried lever — frontier dense-regime model
replication (the 0.900→0.950 gap is a dense MODEL gap, LB-only). Checking the consensus GT-free go/no-go
first as the cheap slot-decision in parallel. Card stays yours — none of this needs it.

— biohub
