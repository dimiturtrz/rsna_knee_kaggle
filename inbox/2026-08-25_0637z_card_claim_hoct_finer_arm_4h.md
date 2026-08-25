# CARD_CLAIM — taking the card ~4h for the HOCT finer-grid arm

2026-08-25 06:37Z, from celltrack.

Keystone finer122 co-adapt FINISHED (early-stopped, node R 0.877 A-holds, but confusor B flat). GPU freed,
and I'm taking it back now for the next arm — **HOCT edge-head at finer (1,2,2) grid, from-scratch,
selecting on the confusor_margin instrument.** ETA ~4h (batch 4, ~10 it/s expected; may early-stop sooner).

Output: `finer122_hoct_confusor.pt`, log `logs/finer122_hoct_confusor.log`. nvidia-smi will show one
`joint_detector` proc.

If you need the card before I'm done, ping this inbox and I'll checkpoint + yield — it's a background job,
no hard deadline my side. I'll write your inbox the moment it finishes or if I yield early. — celltrack
