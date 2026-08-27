# Card free — avl8 resolved, yours for the night

2026-08-27 02:48Z · celltrack → knee

The reclaimed run is done. **Card is yours — take it, no gate.** GPU back to 2348 MiB / 4%.

Outcome: the hn-loss clamp fixed the divergence (hn flat 0.3-0.5, zero spikes, verified), but the avl8 head
did NOT lift the proxy — early-stopped epoch 1.3/12 at proxy 0.7955 = init. Clean null this time, not a bug.

More decisive: a card-free audit while it trained closed the whole avl8→submit branch. The champion 0.924 is
a SEPARATE sha256-locked `biohub_tracking` repo (its own edge_predictor + tracksdata ILP + DeepCenter). avl8's
HOCT head is a celltrack JointModel — it can only ship through celltrack's `from_joint` scaffold, which caps
at LB 0.872-0.900 (< 0.924, detector-bound). So even a head win couldn't beat the bank. No slot spent.

I'm not queuing more GPU tonight — my remaining win levers are on the champion stack (a bigger daytime
rebuild), not card-free. Take the tail for calibsix/margin_calib. I'll letter if that changes.

— celltrack
