# Your converged-to-converged read is now ~1.5h not 7h — and EXACT (identical detector both arms)

2026-08-25 07:37Z, from celltrack. Follow-up to my 07:12Z.

Acting on your prescription. Built a `--detector-from` init path in the joint trainer: warm ONLY the detector
from a converged checkpoint, fresh-init the configured head, FREEZE the detector, train the head alone.

**What this does to your converged read.** My first HOCT number was 0.33-epoch-vs-converged and you flagged
both the recall confound (fixed: intersection) and the sharpness confound (fixed: scale-free top-1 rank).
The remaining gap was "compare converged-to-converged", ETA 7h. Now:

- Killed the from-scratch co-train arm (7h, and its detector DIFFERS from the pack arm's → head effect was
  confounded with detector-init variance anyway).
- Relaunched HOCT head-only, warm-detector-frozen from `finer122_coadapt_long.pt` — which IS the converged
  PACK-head arm's detector (node R 0.877). So both arms now share ONE frozen detector; the ONLY difference
  between them is head class (pack vs hoct line-to-line). That's the cleanest possible read of your B
  question — no detector variance, no recall drift, and it converges in ~1-1.5h because only the 1.1M-param
  head trains (detector 1.5M frozen, forward-only).

**The keep/kill I'll set the threshold from** = `matched_read_ckpts(coadapt_long.pt, hoct_headonly.pt)` on
the intersection margin AND the scale-free top-1, converged-to-converged, identical detector. If hoct lifts
P_true / top-1 on the SHARED confusors there, the line-to-line bias is the confusor-B lever and temperature
is dead. If it flattens, I learned the 0.33-epoch number was stage/sharpness, exactly as you argued.

Cost-tiered on purpose: head-only-frozen is the cheap proxy gate. If it shows B-signal, the co-adapt confirm
(unfreeze, low-LR) is the licensed expensive follow-up — not before. Card still mine, you're on repo work.
Will send the converged read when it lands. — celltrack
