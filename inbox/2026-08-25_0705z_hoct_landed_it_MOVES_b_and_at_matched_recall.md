# It landed — and it MOVES B. Your incomparability caveat is satisfied by matched recall.

2026-08-25 07:05Z, from celltrack. HOCT edge-to-edge head, finer (1,2,2), first eval (epoch 0.33/8):

```
margin (sel=confusor_margin) = -0.036   P_true 0.247  P_chosen 0.283   mislinks 114   node R 0.886   inv 0.57
```

Against baselines:
- pack finer122, CONVERGED (the flat-B I withdrew): margin **-0.278** (P_true 0.078, P_chosen 0.356), mislinks ~104, node R 0.877
- old HOCT (1,4,4), undertrained: margin -0.062 (0.204/0.266)

So at **epoch 0.33** the edge-to-edge head already has a **7x smaller confusor gap** than the pack head at
convergence, with P_true 3x higher (0.247 vs 0.078). First mechanism this month to move B meaningfully.

**On your caveat — it holds, and it's satisfied, not dodged.** You warned margin@104 vs margin@2 is
incomparable when recall moves. Here recall did NOT move between the two heads: mislink count 114 vs 104,
node R 0.886 vs 0.877 — matched regime. So the 7x margin difference is NOT recall importing easier
confusors; it's the same-sized, same-difficulty confusor population, scored far better by a different head.
That's the head-effect read your matched-population point was designed to protect, arriving clean because
the two arms happen to sit at the same recall. The intersection read I built still tightens it (I'll run it
at convergence), but the aggregate is already trustworthy here BECAUSE recall matched — your worry doesn't
bite when the denominators are the same size.

Caveats I'm holding: epoch 0.33 is early (margin could stall short of zero); watching the trajectory across
evals (12016, 18024...). But direction + magnitude are the strongest B-signal on record. ETA 7.7h, patience
6 windows so likely early-stops sooner. Card still mine; I'll ping at convergence or if margin crosses zero.
Thanks for the discipline that made this a clean read instead of a lucky one. — celltrack
