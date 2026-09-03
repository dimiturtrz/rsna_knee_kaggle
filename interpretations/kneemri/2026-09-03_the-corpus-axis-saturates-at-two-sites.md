# The corpus axis saturates at two sites

2026-09-03. `merged3_corpus` ran, landed, and was never written up. This is the reading it is owed.

## What the arm was

`merged_corpus` concatenated Rijeka's 736 studies into ours and read +0.0020 mean — nothing on its own
number, but with a structure-scale split that WAS the result: deficit six +0.0061, focal +0.0056 against
regional −0.0031, and **Lateral Meniscus +0.0196\*** on a finding Rijeka never labels. The ledger's
conclusion was that the gain could not be supervision and had to be the encoder, and that MRNet should
therefore be argued on KIND rather than on volume.

`merged3_corpus` is that argument run. One variable against `merged_corpus`: the 6,393-study cache
(canonical + Rijeka + MRNet) in place of the 5,143-study one. Same recipe character for character.
It carried a second hypothesis of its own — Rijeka fills ONE of six canonical slots, MRNet ships
sagittal, axial AND coronal and fills three — so if the corpus term is slot COVERAGE rather than study
count, this arm reads well above what +24% studies would buy.

Harness check the header demanded, run before reading: `scheme: merged3`, `image_size: 336`,
`epochs: 12`, converged 1/1. The flag landed.

## The reading

**Mean −0.0024**, inside the floor. That is the weaker half. The split is the result, and it is the
`merged_corpus` split INVERTED:

| finding | `merged_corpus` (+Rijeka) | `merged3` (+MRNet) |
|---|---|---|
| Lateral Meniscus | **+0.0196\*** | **−0.0217\*** |
| Synovitis | **−0.0206\*** | **+0.0234\*** |
| ACL | +0.0161 | −0.0013 |
| by structure scale | focal +0.0056 / regional −0.0031 | focal −0.0076 / regional +0.0048 |

Both starred deltas flip sign. The structure-scale signature flips. A second expert-image-read corpus
did not compose with the first — it **cancelled the first one's only starred gain**, on the exact
finding that carried the KIND argument.

## The line that closes it

**MRNet supervises ACL and nothing else, with 1,248 expert image reads — and ACL moved −0.0013.**

That is the cleanest refutation available on this axis, because it is the one column where the mechanism
is not indirect. Rijeka's interesting result was a gain on a structure it does NOT label, which had to be
encoder transfer. Here the supervision is direct, present, and 1.7× Rijeka's volume, and the column it
supervises did not move. The supervision landed and bought nothing.

## What this closes

* **Slot coverage is refuted.** Three filled slots against one, and the arm read −0.0024. The corpus
  term is not slot coverage.
* **Volume is refuted a third time.** +24% studies, nothing. `half_data` (−0.0234 at half) and
  `quarter_data` (concave, each doubling buying ~0.48 of the last) already priced the next doubling at
  ~+0.009; this arm is a 0.24 step and read negative.
* **KIND does not compose.** The one lane `quarter_data` left open was a corpus differing in kind rather
  than in count. Two such corpora now exist in one cache and the second undid the first.

The arm's own pre-registration puts this on the **FLAT** branch, not the down branch — that one required
"down past −0.0062, and ACL specifically down", and ACL is −0.0013. So `MRNET_LEVEL` (the 0.82 discount
for a binary label lumping partial with complete tears) is NOT implicated and stays unsuspected.

## What it does not close, and what it repriced

The flat branch names its own successor: whether the foreign block is being **drowned** rather than
whether more data helps — `--corpus-oversample` exists and has never been run on a merged cache. That
question is open. It is also a knob on a lane that has now produced +0.0020 and −0.0024 across two
sites, so it is priced as an instrument, not as a gain.

**It reprices an owner action I raised earlier today.** MRNet's `meniscus` and `abnormal` label tables
were never built and would need a gated Stanford re-download; I flagged that as ~1,130 image-read labels
against our 58. This arm says those labels would be more of a supervision that measurably did not
transfer on the column we already have. The request stands at low priority, not as a lever.
