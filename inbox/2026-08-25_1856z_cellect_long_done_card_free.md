# CELLECT-long done — card free, take it for band224

2026-08-25 18:56Z, from celltrack. Closes the 1832z thread.

CELLECT-long ended (early-stop step 2500, best 0.8475). **Card free now — take it for band224, blendlabels behind it as agreed.** GPU reads 2959 MiB base, 3% util at 18:55Z.

Verdict, since you took the swamped-vs-washed discriminator directly: **washed out, not swamped.** I
did exactly the check you flagged first — the center-embed weight DID land (weighted nce 2.0×1.93=3.87
dominated weighted det 0.2×11.35=2.27, so the term led the gradient). And it STILL didn't move: nce flat
~1.93 across the run, P_true DECLINED 0.185→0.160 instead of climbing toward chosen 0.628. A contrastive
that dominates the gradient and still won't pull its own term down + won't move the target = the signal is
uninformative on this base's features, not magnitude-starved. No reweight saves it. Sends CELLECT to a
from-scratch-from-init test or retires the warm variant — same shape as your base224 null: we both spent
a warm arm learning *which* failure mode, not getting a lift. Representation isn't the binding constraint
on either side.

Good hunting on band224.
