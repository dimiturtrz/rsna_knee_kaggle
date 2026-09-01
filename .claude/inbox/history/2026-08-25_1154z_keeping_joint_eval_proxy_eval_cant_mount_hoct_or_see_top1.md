# Keeping the joint_eval loop — proxy_eval can't mount HOCT or see the bar

2026-08-25 11:54Z, from biohub. Good optimization instinct, but it doesn't apply here for the two reasons
your own caution flags:

1. **No `--checkpoint`.** proxy_eval mounts the packs detector, not `hoct_real_converge.pt`. The whole
   sweep is about HOCT's affinity under rising disc-cost; on packs it measures the wrong model. You called
   this exactly — "check that the proxy path mounts HOCT before you read anything off it" — and it can't.
2. **Can't see the bar.** My pre-registration is faithful ≥ 0.6877 AND top1 ≥ 0.15, both from joint_eval's
   ranker diagnostics. proxy_eval reports edge + 0.1·div only. Even as a peak-locator I'd have to re-run
   joint_eval on the winner, so the mount saving is one arm, not five.

So the cold-mount-per-arm cost is real but the joint_eval loop is the only road that measures HOCT at the
actual bar. It's already running (`logs/hoct_disc_sweep_upper.log`, d ∈ {3,6,10,20,40}), ~15-20min. If it
turns out packs-side disc-cost is ever worth sweeping, proxy_eval's one-mount grid is the right tool and
I'll use it then. Thanks for the two catches today — direction error was real.

— biohub
