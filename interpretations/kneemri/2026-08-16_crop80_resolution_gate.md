# The resolution gate: half the prediction landed, and it is the wrong half

`crop80_tissue` reads 80 mm about the tissue centre at 336 px — 0.238 mm/px against the canonical
130 mm's 0.387. It exists to decide, for one arm's cost, whether the multi-scale build (knee-1nn) is
worth a 71.6 GB cache and a training slot. Its prediction was deliberately two-sided: **small
structures up, diffuse findings down**. A uniform fall refutes the resolution mechanism outright; a
uniform lift is a confound; only the split earns the build.

## The comparison, and why it is at matched epoch

The obvious read — 0.8345 against the control's 0.8457 — compares an 8-epoch arm's best against a
12-epoch arm's best, which is a schedule difference on top of the crop difference. It is also the
read the arm's own header forbids, because a two-sided prediction averages its halves against each
other in the mean.

Both arms logged per-label AUC every epoch on the same fold-0 holdout, so the honest comparison is
**epoch 7 against epoch 7** — same data, same epoch, differing only in crop and centre. Worth noting
while we are here: the control peaked at **epoch 6** and declined for its remaining five epochs
(0.8457 → 0.8410 → 0.8394 → …), so the extra four epochs bought it nothing, and the −0.0028 that
`schedule_length` measures for 8-vs-12 elsewhere should not be lent to this pair.

| finding | crop80 e7 | control e7 | Δ | the prediction |
|---|---|---|---|---|
| Fracture | 0.802 | 0.783 | **+0.019** | small ↑ ✓ |
| Lateral Meniscus | 0.824 | 0.819 | +0.005 | small ↑ (at the floor) |
| PF OA | 0.821 | 0.820 | +0.001 | — |
| MCL | 0.740 | 0.743 | −0.003 | small ↑ ✗ |
| ACL | 0.860 | 0.864 | −0.004 | small ↑ ✗ |
| Contusion | 0.847 | 0.851 | −0.004 | — |
| Lateral OA | 0.809 | 0.817 | −0.008 | — |
| Medial OA | 0.873 | 0.883 | −0.010 | — |
| Medial Meniscus | 0.877 | 0.889 | **−0.012** | small ↑ ✗✗ |
| Effusion | 0.836 | 0.851 | **−0.015** | diffuse ↓ ✓ |
| Synovitis | 0.856 | 0.875 | **−0.019** | diffuse ↓ ✓ |
| Baker's | 0.869 | 0.897 | **−0.028** | diffuse ↓ ✓ |

Against a 0.0053 noise floor, five moves are real: Baker's, Synovitis, Effusion and Medial Meniscus
down, Fracture up.

## What this says

**The diffuse half is confirmed, and it is the mechanically obvious half.** Baker's cyst sits in the
posterior popliteal fossa; a 40 mm half-extent about the tissue centre cuts it out of the picture
entirely, and it is the single largest loss on the board. Effusion and synovitis are fluid
distributions read over the whole joint. Losing 0.028 on a finding whose anatomy you cropped away is
not a discovery about resolution — it is confirmation that field of view matters per finding, which
is what knee-dtk already argued.

**The half the build depended on did not appear.** The multi-scale case rests on tight slots buying
back the small structures. Three of the four went nowhere (ACL, MCL at the floor) or went *down*
(Medial Meniscus, −0.012 — the largest small-structure move on the board, and the wrong sign). Only
Fracture rose, and fracture is the finding whose label agreement is worst, so it is also the one whose
single-arm move is least trustworthy.

**So the resolution mechanism is not refuted — it is unrewarded.** Doubling in-plane sampling did not
make meniscal tears more visible. That is evidence against the reading of knee-jk3 that says "336 px,
for findings like meniscal tears that are a few voxels across" is what stands between us and the
frontier. Depth of view kept paying (whole_stack, +0.0147); finer pixels over a narrower view did not.

## The decision on knee-1nn

Multi-scale ADDS the tight slots to the wide ones rather than replacing them, so none of the diffuse
losses above would recur — the context slots stay. Its value is therefore exactly what the tight slots
buy, and this arm measured that: Fracture, and nothing else that clears the floor. Spread over twelve
labels that is roughly **+0.0016 on the mean**, against a 1.5% bar, for a 71.6 GB cache that must be
built alone and a training slot.

**It does not earn the build.** That is the gate working as designed and for one arm's cost, which was
the entire point of running it first.

What this does NOT rule out, so a later use is re-tested rather than assumed dead:

- **Resolution at constant field of view.** `res392` (448 cache read at 392) is still our best single
  member at 0.8523. This arm changed pixel size by *cropping*, which pays for sampling with context.
  The two are not the same intervention and only one of them has failed here.
- **A tight slot for fractures specifically.** The one real lift was +0.019 on the finding most
  starved by label noise. That is a per-finding reader, not a global scheme, and it is not what
  knee-1nn proposed.
- **The centre change.** This arm moved crop and centre together (the control is image-centred). Tissue
  centring cannot explain a *loss* on Baker's — it should have helped — but it does mean no single-axis
  claim about centring can be read off this pair.
