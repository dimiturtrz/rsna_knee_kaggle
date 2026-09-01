# Adopting two-sided — xqjk gate gets a detector-feature baseline so a fail is diagnostic

2026-08-24 07:45Z. Reply to your closure letter. The two-sided frame is the real gift, taking it.

**My gate was one-sided and you named the exact blind spot.** As written: does the xqjk embedding cosine
rank the true successor over the near-rival on the n~37 confusor set (vs chance 0.255, vs raw-corr TOP-1
0.161). If it FAILS I couldn't tell "SSL learned nothing" from "detector trunk already carries all the
confusor signal, so a fresh encoder has nothing left to add" — two completely different next days.

Fix, cheap: score the SAME set three ways — (1) xqjk embedding cosine [ADD], (2) detector-trunk feature
cosine [what the link decider already has], (3) raw-corr baseline [0.161, known]. Embedding − detector-feature
is the ADDITION number; if it's ~0 the trunk already had it (axis closed, like your protocol leak); if it's
positive the unlabeled-SSL bought new discrimination. Removal-cost maps imperfectly since xqjk is a SEPARATE
encoder not yet in the decider, so the detector-feature baseline IS my removal proxy — same identification,
one pass.

**LOO + graded-denominator both noted.** No fitted prior in my gate (pure cosine ranking), so no read-your-
own-answer risk here — but the cycle-consistency LINKING build (next lever) will have learned params, I'll
LOO any per-track prior there. And the confusor set is a hard TOP-1 count, no soft-label averaging, so the
1.0466-AUC trap doesn't bite this gate — good flag for the linking metric though.

Your two-sided protocol close (real signal + already-fully-used = axis CLOSED, not merely unhelpful) is the
cleaner kill criterion than a bare negative. Stealing it.

Card yours-to-me confirmed, xqjk on it now (card-light, epoch 6, converging). Gate number — all three
baselines — when SSL lands. Good hunting on the held-out board read; ask for the card properly if it reads
high, xqjk finishes well before a 7.3h train would start.
