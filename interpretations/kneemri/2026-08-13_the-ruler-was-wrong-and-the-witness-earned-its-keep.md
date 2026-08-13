# The ruler was wrong, the witness earned its keep, and a bad result nearly got believed

2026-08-13. Three defects found in one day, one of them in the scoring function itself. The
interpretation of 2026-08-12 catalogued **seven bugs of one shape** — a value travels in the config and
a consumer reads a default — and predicted that the witness would be what catches the next ones. It
was, on its first real use.

## The witness worked, and it found three things nothing else could

`witness_check` re-derives what training predicted for eight named studies through the *inference*
path. Running it for the first time on real members:

1. **`torch.compile` renamed every parameter.** `compile_encoder` did `self.backbone =
   torch.compile(self.backbone)`, and the rebinding puts `_orig_mod` into every state-dict key. Five
   members — `schedule_8`, `extended_ws` and all three DINOv3 runs, 223–235 keys each — were
   **unloadable by the inference path**. Invisible by construction: training never reloads its own
   weights, so they ran, scored and entered the ledger while being unsubmittable. `nn.Module.compile()`
   compiles in place and leaves the names alone.
2. **Every bf16 member was served in fp16.** `Ensemble.predict_member` ran a dtype-less
   `torch.autocast`, whose CUDA default is float16, while the trainer carries it per run.
   `config['bf16']` had been recorded since the flag landed, under a comment saying the two are
   different arithmetic, and read by nobody — **the eighth instance of the 2026-08-12 shape.** The
   **ninth** was the same defect in `HoldoutPredictions`, i.e. in the tool that chooses which members to
   submit.
3. **Its own tolerance was tighter than bf16's epsilon.** 2e-3, inherited from the fingerprint and
   documented as "above GPU noise" — true until bf16 landed, whose eps is 2⁻⁷ = 7.8e-3. The gate would
   have refused every bf16 member. Measured across four members and two dtypes, the residual tracks the
   *witness's* eps at 1.6–2.4× of it, so the bound is now a factor on eps rather than a constant.

Then it was proven rather than argued: `sliceattn` with `whole_stack` flipped in a copied manifest is
**refused at 0.407 against a 0.00391 bound** — 104×. A reading change and arithmetic noise are two
orders of magnitude apart.

## The ruler itself was wrong, which is a different and worse shape

`LabelScores.area_under_curve` built Mann-Whitney U from `argsort().argsort() + 1` — **ordinal** ranks.
Tied predictions are ordered by their position in the array, so **a tie was settled by study order**. A
constant predictor, carrying no information at all, scored **1.0** when the positives happened to sit
late in the corpus and 0.0 when they sat early.

Found by a unit test written for something else: the oracle ceiling must be ≥ the plain average, and it
failed because the average was scoring 1.0 on predictions that were a constant 0.5.

- **Single-member numbers never moved** — raw float64 sigmoids effectively never tie, so every
  per-member score in the ledger stands.
- **Ensemble numbers did.** `Ensemble.combine` maps each member onto k/(n−1) and averages, which ties by
  construction: **20.5% of 881 studies at two members**, 14.4% at five, 7.4% at eleven.

Study order is arbitrary against the labels, so this is noise rather than bias — but it corrupts the
*ranking of subsets*, which is the one thing `member_curve` exists to produce and the thing that picks
a submission. The 0.8601 five-member peak should not be quoted again until recomputed.

## A bad result nearly got believed, and the rule caught it

v3+LoRA sat at **0.6867** after four epochs against fine-tuned v3's 0.8057, loss flat at 1.23. That
reads as *"rank 8 underfits"* — which is the refutation the arm was **designed to accept**.

`LoRALinear` freezes the `nn.Linear` it wraps and nothing else. Everything that is not a Linear stayed
trainable and was optimised at the **adapter rate of 5e-4**, sixty times the fine-tuning rate: position
embeddings (526k), the patch projection (226k), every norm, every LayerScale gamma — **781,440
parameters against the adapters' 663,552.** More than half of what the optimiser opened was not an
adapter.

**The validation that missed it looked like it passed**: 663,552 + 781,440 = 1,444,992, and the arm
header quoted "1.44M trainable, 6.4% of the encoder" as proof the method was wired correctly. Counting
a total without asking *which* parameters is how a wrecking ball reports the expected number. The unit
test missed it for the same reason — its stand-in backbone was made only of Linears, so the leak had
nothing to leak.

Under the fix, the same arm reaches **0.8217 against fine-tuned v3's 0.8057, +0.0160**. A false kill
became a confirmed result.

## What that result says, and what it does not

DINOv3's deficit was **about a third our adaptation recipe and about two thirds the encoder**. The
morning's interpretation ruled out patch grid (441 patches against 196 bought 0.0023) and capacity
(v3-S+ scored 0.0045 *worse* than v3-S) and named adaptation as the one axis never varied. It was the
right axis, and worth more than the two that had been tested.

DINOv2 stays: v3+LoRA at 0.819 is still 0.027 short of 0.8457. And on DINOv2 the same method is
**behind at every matched epoch** — 0.8127 against 0.8318 at epoch 3 — which is that arm's own
predicted refutation for the encoder whose recipe was tuned on it. The contrast is the finding: our
fine-tuning is right for v2 and wrong for v3.

## Two P0s I filed against things the repo already knew

Recorded because the pattern matters more than either bead. I raised a P0 claiming whole-stack members
had lost a measured +0.0156 of TTA; `experiments/README.md`, bead `knee-eeo` (closed) and the
2026-08-12 interpretation all already said that whole-stack and overlap TTA are **one mechanism bought
twice**. I then raised `knee-4ir` to P0 on the grounds that "does diversity dominate quality" was open;
the same README settles it across six leaderboard entries.

I audited the beads and not the ledger. The rule is now in `CLAUDE.md` and `AGENTS.md`: beads are the
work queue, `experiments/README.md` and `interpretations/` are where measured results live, and
`bd search` will not find them.
