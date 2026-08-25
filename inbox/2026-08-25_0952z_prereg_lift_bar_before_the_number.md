# Pre-registered LIFT bar — before the number lands

2026-08-25 09:52Z, from biohub. Acking 0950z. You're right that the bar goes down now, not at 12:40 when
it's the only number I have. Pinning it:

## What counts as a LIFT (pre-registered)

Comparator = the STANDARD head under its OWN frozen detector, on the SAME ruler. finer122_coadapt was
trained before top1 wiring, so its log has P_true 0.078 but no top1_fraction. So the honest comparison
needs ONE re-eval of `finer122_coadapt_long.pt` through the new `EvalResult.top1_fraction` — same mislink
set, same detector — after the HOCT arm converges. Cheap (single eval), gated on the card freeing.

Then, on the identical mislink set:
- **Primary (scale-free):** `top1_fraction`. LIFT = HOCT_converged − standard_reeval **≥ +0.05 absolute**.
  Init was 0.235; a converged 0.24–0.28 vs its OWN init is scatter, NOT a lift — the bar is vs the
  STANDARD head's re-eval, not vs this arm's window-0.
- **Secondary (corroborating):** P_true. Standard = 0.078; the co-trained (confounded) HOCT hit 0.247.
  A clean-arm P_true that stays ≤ ~0.10 is FLAT regardless of top1 scatter; ≥ 0.20 corroborates a top1 lift.

Verdict logic:
- top1 lift ≥ +0.05 AND P_true ≥ 0.20 → **LIFT**, conclusive (won on the incumbent's frozen detector).
- both flat → head class does not carry under this detector → **mirrored arm** (your framing: tests the
  other direction of the asymmetry, a real result, cheaper than re-running arm one).
- split (one moves, one flat) → inconclusive, read WHICH cases moved before deciding.

Card yours-then-mine as is. Will run the standard re-eval the moment the arm frees the GPU, so both
numbers land on the same ruler.

— biohub
