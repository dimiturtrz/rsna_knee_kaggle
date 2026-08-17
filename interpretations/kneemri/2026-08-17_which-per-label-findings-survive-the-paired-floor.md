# Re-read against the paired floor: one of the four Fracture findings survives, and almost everything that clears the floor is a LOSS

2026-08-17 · `python -m kneemri.evaluation.run_comparison --control wsfold0_e6.pt --arms extended_ws
intensity392 res392 slices16 schedule_8 crop80_tissue pair_memory`. No training, no GPU: each pair's
interval is a bootstrap over the studies both runs kept. Control `wsfold0_e6` = 0.8457, its scores
reconstructed from its kept predictions, so **every pair below is marked INDICATIVE** — one of the two
runs had its predictions re-derived rather than recorded live.

`*` = the delta clears THIS PAIR's own paired half-width (0.0190 to 0.0366 depending on the pair).

## The claim this bead was opened on

knee-3vh recorded that four arms with unrelated mechanisms each "improved Fracture" by +0.012 to
+0.037 against an unpaired interval of ±0.061 — read as one noisy label rather than four mechanisms.
The paired floor is two to three times tighter. It does not rescue them:

    arm             Fracture   clears its own paired floor?
    intensity392     +0.0372   YES  (*, floor 0.0311)
    res392           +0.0330   CANNOT BE ASKED — kept no per-study predictions, unpaired bound only
    extended_ws      +0.0200   no   (floor 0.0288)
    crop80_tissue    +0.0183   no   (floor 0.0366)
    schedule_8       +0.0158   no   (floor 0.0254)
    slices16         +0.0120   no   (floor 0.0190)
    pair_memory      +0.0003   no

**One of six resolves.** The original reading — one noisy label, not four mechanisms — survives the
tighter ruler, and it survives it in the strongest form: the arms whose Fracture gain was quoted in
the ledger (extended_ws, slices16, schedule_8) are all UNREAD even against a floor built from their
own shared studies. Unread is not refuted; it is the ledger's per-label Fracture rows being claims
nothing has yet supported.

`res392` is the one that cannot be asked at all. It predates `HoldoutRecord`, so its per-label numbers
— including the +0.0273 MCL that the resolution argument leans on — still print against the unpaired
bound. That is a second reason its row is unsafe, on top of the confounded encoder rate (knee-a8x).

## What the paired floor makes READABLE that the unpaired bound hid

The tighter interval was supposed to promote deltas the conservative bound suppressed. It does, and
the promoted set is almost entirely NEGATIVE:

    extended_ws     PF OA +0.0156*   Medial Meniscus -0.0095*
    intensity392    Fracture +0.0372*   Synovitis -0.0229*   Medial OA -0.0185*
    slices16        Medial OA -0.0103*
    schedule_8      Lateral Meniscus -0.0146*
    crop80_tissue   Baker's -0.0270*   Effusion -0.0173*   Synovitis -0.0172*   Medial Meniscus -0.0157*
    pair_memory     nine of twelve starred, all negative

Across seven arms, exactly **two** positive deltas clear their pair's own floor: `PF OA +0.0156` under
extended slots and `Fracture +0.0372` under intensity392. Every other resolvable per-label move in the
table is a loss. That is the shape of a set of arms whose MEAN barely moved (+0.0002, +0.0003, -0.0005,
-0.0027): they are not lifting some findings and holding the rest — they are trading, and the trades
that resolve are the ones going the wrong way.

## pair_memory is a refutation, not noise

    Lateral Meniscus -0.0602*  Medial Meniscus -0.0597*  Baker's -0.0482*  PF OA -0.0373*
    Lateral OA -0.0271*  ACL -0.0264*  Synovitis -0.0241*  Medial OA -0.0217*  Contusion -0.0213*
    MEAN -0.0302

Eleven of twelve findings down, nine of them past a 0.0332 floor. This is far outside anything a
sampling story explains, and it is the clearest negative result the paired harness has produced.
Its diff line reads "matched on every option both runs recorded", so the mechanism to blame is
`pair_memory=8` itself — an option the control predates rather than one it set differently.

## What this does NOT say

- INDICATIVE is not a hedge for politeness. The control's scores were reconstructed from stored
  predictions, and no arm here was scored twice under identical conditions. A star means "clears the
  interval a bootstrap over shared studies gives", not "replicated".
- It does not re-price any arm's MEAN. Those were already read against the 0.0062 three-seed floor and
  nothing here changes them.
- It says nothing about `res392` per label, by construction — see above.
- Nothing here was submitted.
