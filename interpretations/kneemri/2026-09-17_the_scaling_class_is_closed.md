# The scaling class is closed: depth, schedule and trunk are all flat-to-negative on our stack

*2026-09-17. Measured entirely off checkpoints already on disk — zero card cost. The numbers are
`ledger75` over `D:/data/volumetric/mri/models/rsna_knee_abnormality`, 909 held-out studies, fold 0 / split_seed 2026.*

## Why this class and not another

`experiments/README.md:14829-14890` closed the per-column axis on four spent slots: the board metric is
a plain unweighted mean over twelve findings, the board's 0.056 of headroom is **distributed, not
pooled**, and driving a weak column from 0.932 to a perfect 1.000 buys `(1.000 − 0.932)/12 = 0.0057`.
Its own verdict named what survives — *"exactly the class of change that lifts all twelve columns at
once: what the model sees and how much of it, and what it was pretrained on"* — i.e. **depth, more
corpus, stronger trunk**.

This page prices all three. None of them carries.

## Depth: flat, and NEGATIVE across the 3x jump

Three ladders already existed; nobody had put them side by side.

```
ladder   px   ep  calib   shallow                                  deep                                      delta
A       336   12  True    labelwise_dense_s2026         0.8964     labelwise_depth24_s2026        0.8998     +0.0034
B       336   24  True    labelwise_dense_drop40_long   0.9048     labelwise_depth24_drop40_long  0.9053     +0.0005
C       224   12  False   labelwise224_s2026            0.8832     cover_labelwise_s2026          0.8802     -0.0030
```

Every delta is inside the 0.0062 paired floor, and ladder C — the only one carrying a **3x** slice jump
(12 to 36) — is the one that goes negative. The belief that the depth curve was "still steepening at 20"
came from the FOREIGN curve, not from ours.

## Schedule: peaks at 24 epochs, regresses at 36

Same corner (sl20 / 336 / labelwise / calib=True), three budgets:

```
ep12   labelwise_dense_s2026                  0.8964   peak epoch 7
ep24   labelwise_dense_drop40_long_s2026      0.9048   peak epoch 23    +0.0084 over ep12
ep36   labelwise_dense_drop40_long36_s2026    0.9030   peak epoch 17    -0.0018 vs ep24
```

The schedule moves more than depth does, and it is the one sub-lever that has actually been paying —
but it has stopped. The 36-epoch run does not merely fail to improve, it peaks a third of the way in
and lands below the 24-epoch run. Longer is spent.

## Trunk: non-monotone, and it REVERSES at our operating resolution

Every labelwise member this project owns is `variant: small` — 306 of 364 scoreable members are small,
and **no labelwise member has ever run on a bigger trunk**. That looked like an untested gap. It is
not: the capacity question has been answered three times at `max` pooling, and the answer depends on
resolution in the direction that kills the lever.

```
224 / 12sl / max / ep12   DINOv2   base224_control_s2026  0.8754  ->  base224_s2026          0.8795   +0.0041
336 / 20sl / max / ep12   DINOv2   dense family    ~0.8900     ->  base family        ~0.8828   -0.0072
224 / 12sl               DINOv3   dinov3_base_lora_s2026 0.8488  ->  v3large_dense_s2026    0.8491   +0.0003
```

The 224 pair is an explicit matched control (`base224_control_s2026` was built to be exactly that). The
336 row is the group mean over the three `base` members at that recipe against the small dense-family
controls at the same recipe (`band_dense` 0.8877, `through_dense` 0.8867, `calib_dense` 0.8916,
`clean_calib` 0.8940).

Read together: **capacity buys something only where resolution is starving the trunk, and at 336 px —
where we actually ship — the smaller trunk wins.** And the one ladder that reaches `large` shows large
flat against base. A bigger trunk on 3.6k training studies is not being limited by its parameter count.

## What this leaves

The ledger named three survivors. Depth is measured flat-to-negative, schedule is measured saturated,
trunk is measured flat-to-negative and adversarial at our resolution. Corpus is the only one of the
three not settled on this page, and it is settled elsewhere and negatively: in-domain SSL is behind its
matched control on every objective that has been fit (latent MIM −0.0258, volume-native `vjepa_warm`
−0.0204, cross-corpus `mrnet_warm` −0.0104 against `band_dense` 0.8877), and `mae_unitfix` reads 0.128
BEHIND its control at the matched epoch (knee-zxm9z, not armed).

So the entire class that the per-column closure licensed is now spent. That is a real finding and it is
the expensive kind: it says the project is at a local maximum of the SCALING axis, and that further
wall-clock spent on bigger/longer/deeper is spent against a measured null.

## What it does NOT close

The per-column closure's inference was *the board gain must come from something lifting all twelve
columns at once*, and it then equated that with scaling. **Those are not the same set.** A change to the
TARGET GEOMETRY also lifts all twelve at once and is not a scaling knob at all.

The concrete candidate is the one the v3 page ended on. The board metric is an unweighted mean over
twelve findings, and the reader's own error rate is wildly non-uniform across them — over the 58
annotated studies, false-positive rate runs from 9.7% (Synovitis) to 69.6% (Effusion), and the model
already beats its own labels on 6 of 12 findings (`labels − model` reads −0.0470 [−0.0717, −0.0218]).
Equal-weighted BCE therefore spends its largest gradients on the noisiest column. A per-column
reliability weight changes the RELATIVE gradient across findings in a way no monotone rung remap and no
amount of capacity can.

The blocker the v3 page correctly raised: any weight fitted on the 58 gold cells is unreadable on the 58
gold cells — that is the `raptor_solo` mechanism again under a new name. **The way past it is to fit the
weights somewhere other than gold.** Per-column label noise can be estimated from the disagreement
between the report label and a member's own OUT-OF-FOLD prediction across the full corpus, which is
held-out by construction and never touches the 58. Gold then stays clean and becomes the adjudication
set it was supposed to be.

That is untested, it is not a scaling knob, and it has a mechanism that could plausibly move all twelve
columns. It is the next thing to price.

## Denominators

All member scores: `ledger75`, 909 held-out studies, fold 0 / split_seed 2026, mixed cut — a number in
this repo is midpoint unless it says otherwise, and every number here says `ledger75`. Paired floor
0.0062. Reader error rates: 58 annotated studies, 696 cells, 34.5% positive.
