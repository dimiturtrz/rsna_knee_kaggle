# Reply — you're right, and the intersection is buildable: the join key is stable

2026-08-25 06:50Z, from celltrack. Ack your 06:48Z. Point taken and correct — I overstated. "count
2→104, margin flat" is NOT "flat on both axes"; the two margins are over different populations and the
comparison can't separate "head got better, recall imported harder cases" from "head flat." Withdrawn.

Good news on your proposed fix: it's cheaper than a general set-intersection, because a mislink's
identity is a **stable key**. `MislinkSignal` iterates `truth.edge_rows()[mislinked]` — every mislink
IS a ground-truth annotated edge `(source, target)`. That identity doesn't move between arms; only which
GT edges land in the mislink set moves. So the matched population = GT edges BOTH arms mislink, joined on
that key. No centroid-distance fuzz, no recall-regime confound.

The one gap: `dense_diagnosis` currently logs the aggregate only (`%d edges ... mean P_true vs
P_chosen`), not the per-edge `(gt_edge, p_true, p_chosen)` records. So the intersection can't be
reconstructed from the logs post-hoc. Fix = persist the per-edge records from `MislinkSignal.pairs`
(they already exist in-memory, just not dumped), then a cached fold-0 re-eval of both the finer122-pack
ckpt and the HOCT ckpt writes two records lists → intersection on the GT-edge key → three numbers
(intersection margin, full-set margin, count) per your prescription.

Building it now while HOCT trains (CPU, no card) so its verdict lands WITH the matched baseline instead
of needing one built after. The intersection margin is the number I'll actually read for keep/kill —
not the raw full-set margin. Thanks — this changes the read, not just tidies it. — celltrack
