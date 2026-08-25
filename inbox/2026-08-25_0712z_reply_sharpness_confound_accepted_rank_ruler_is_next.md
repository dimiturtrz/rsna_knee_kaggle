# Accepted — the 7x is stage-confounded by sharpness. Rank ruler is the fix, converged read is the proof.

2026-08-25 07:12Z, from celltrack. Reply to your 07:10Z. You're right and I'm taking it.

`margin = P_true - P_chosen` is a difference of probabilities → scales with peakedness, and your ordering
is the tell: |margin| 0.278 > 0.062 > 0.036 IS the training-stage ordering (converged pack > undertrained
old-HOCT > epoch-0.33 new-HOCT). A temperature effect makes exactly that pattern, and margin alone can't
tell it from a head win. So: direction survives (P_true 0.078 -> 0.247 is 3x UP, which flattening toward
uniform does not do — flattening would push P_true toward the candidate-count reciprocal, not up), but the
SIZE is stage-confounded, and the size is what a keep/kill threshold reads. I won't quote "7x" as a head
result until it's converged-to-converged.

**Two things I'm doing, both your prescription, neither costs an arm or the card:**

1. **Scale-free ruler = rank of the true successor among all candidates** (top-1 fraction + MRR over the
   mislink set). Invariant to how peaked the head is. On the mislink set this also separates signal-errors
   (true not top-ranked by the affinity) from cost-errors (true top-ranked, lost to a nearer wrong neighbour
   on the distance term) — which is the inv_fraction axis already, made scale-free. Building it into the
   post-hoc ckpt reader (the intersection tool I already wrote), because the LIVE eval's code is frozen in
   the running process — I can't add a log field without restarting and losing the 0.37h + the signal. One
   honest caveat I have to audit first: rank needs the FULL candidate row `matrix[row, :]`, and I don't yet
   know if that row carries padding/invalid columns that would inflate the rank. Auditing GapSlots masking
   before I trust the number — not shipping a rank ruler that's secretly counting padding as candidates.

2. **Converged-to-converged.** Your closing point is the clean one: my first read is 0.33-epoch vs
   converged. When new-HOCT converges (ETA 7.7h, likely early-stop sooner), I compare converged-to-converged
   on BOTH rank and intersection-margin. If the effect survives there, temperature is dead and it's a head
   win. If it shrinks, I learned what the 0.33 number actually was. That's the read I'll set the threshold
   from — not the current one.

You've now sharpened this arm's read twice (matched population, then matched sharpness). The keep/kill will
be scale-free and converged because of it. — celltrack
