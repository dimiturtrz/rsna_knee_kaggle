# The volume-pretrained trunk ties, and the seat axis prices itself out

`experiments/arms/vjepa_probe.sh`, bead knee-jucq. Card time 15 min against 35 priced.

## The numbers

| trunk | read | stored per slot | frozen holdout |
|---|---|---|---|
| DINOv2-large (control) | 252 px, per-slice | 12 vectors | **0.7669** |
| V-JEPA 2-large | 256 px, 12 slices as FRAMES | 1 vector | **0.7620** |

Delta -0.0049 against a holdout noise floor of 0.0062. The middle branch of the pre-registration: not
better, not worse, and by the arm's own words "do not read a tie as a kill".

## The plumbing fired, which was the link most likely to break silently

The control cache is `(4407, 6, 12, 1024)` and the V-JEPA cache is `(4407, 6, 1, 1024)`. Depth 1 is the
proof the tubelet path ran: the 12-slice group entered the encoder as FRAMES and was consumed inside it,
rather than being folded into 3 channels or encoded slice-by-slice. A depth-12 V-JEPA array would have
meant the volumetric branch never fired and the whole arm would have been a capacity comparison wearing a
pretraining-domain label. It is checked here rather than assumed because that is the failure this
project keeps paying for.

The confound the arm was built to remove is also gone: 1024 wide against 1024 wide, 304M against a
comparable ViT-L, each trunk reading the size its own patch grid tiles (14 -> 252, 16 -> 256). What is
left in the difference is pretraining domain and depth handling, which is what was wanted.

## What the tie actually says, which is more than "nothing"

The two sides are not symmetric, and reading this as "no difference" throws away the asymmetry.

V-JEPA 2 was pretrained to predict in latent space, an objective with no reason to produce linearly
separable features; DINOv2 was pretrained with an objective that does. On the ruler that most favours
DINOv2 -- a frozen linear-ish probe -- they tie. And V-JEPA reaches that tie while handing the head ONE
vector per slot where the control hands it twelve to max over. The volumetric trunk does its depth
reduction inside the encoder, with learned weights, and loses nothing against a max over twelve.

So the honest reading is: the asset is alive, its cheap path is neither opened nor closed, and the
frozen probe is a weak instrument for exactly this objective. It does not license a claim that the trunk
is better.

## But the seat question it was launched to answer has since priced itself out

The arm was justified as buying DISTANCE for a seventh seat, on the 2026-08-21 split where a
recipe-distant sixth took 0.931 and a near-twin cost a tick. That framing survives the tie -- a tie on
quality plus large distance is exactly the profile of `lr4e-5_ws`, which was WORSE by both offline rulers
and took the entry.

What does not survive is the SIZE of the prize. The split it appeals to is 0.931 against 0.930. The whole
champion progression across many seats is 0.925 -> 0.931. A seat, played perfectly, is worth about one
board tick, and this project's bar for a result is ~1.5% absolute. The seat axis cannot clear that bar
even when every decision on it is correct.

That is the night's real finding and it is a measurement, not a mood. The distance sweep over 119
checkpoints says every working member orders the studies 0.87-0.92 alike, so there is little disagreement
left to buy; and the paired board history says what disagreement remains is worth ~0.001 when bought.
Both rulers now agree the ensemble axis is near its cap.

## Where the mass actually is

Frozen probes sit at 0.762-0.767. Fitted members sit at 0.85+. Fine-tuning carries roughly +0.09 --
two orders of magnitude more than a seat swap. Every question about which trunk to STAND ON is therefore
a question about the fine-tuned number, and the frozen probe can only say which trunks are worth the
fine-tuning arm.

This one is. A trunk that ties on the ruler biased against it, while compressing 12 slices into 1 vector
with learned weights, is the candidate for the backbone question the owner has asked for twice. The next
arm is a FINE-TUNED V-JEPA member against a matched fine-tuned DINOv2-large -- read as MEMBER QUALITY,
not as a seat -- because that is the only axis in reach whose mechanism could plausibly deliver 1.5%.

## What this does NOT rule out, so the asset stays re-testable

- Fine-tuned V-JEPA. Untested. The frozen probe is a known-weak instrument for a latent-prediction
  objective and this result is evidence about the probe as much as about the trunk.
- A tubelet size other than 2, or a slice window other than 12. Neither was varied.
- V-JEPA as a SEAT rather than as a backbone. The tie still licenses fitting one member and pricing it,
  and it would be the most distant candidate in the pool if its ordering follows its pretraining.
  Deprioritised because the prize is ~0.001, not because it was refuted.

## Blocking consequence

knee-2nxh stops being deferred. `Encoder.activation_units` is 2D-only and undercounts a tubelet trunk by
frames-per-tubelet -- 6x at a 12-slice window with tubelet 2 -- so the VRAM cap will size a fine-tuning
run on a number that is 6x too small. The frozen probe paid no backward pass and no optimiser state, which
is why it was safe to defer; a fine-tuning arm pays both. Fix before the arm, not after the OOM.
