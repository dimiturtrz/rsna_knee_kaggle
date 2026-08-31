# Distilling a teacher's LOGITS into a trunk is not pretraining — and it does not answer the feature question

`swin_distill_pre` → `swin_distill_fit` (knee-8009 a), read 2026-08-31. Two stages, both on disk,
both converged in the sense `convergence_check` cares about: stage two ran its full 12 epochs, peaked
at epoch 6, and is not on the truncated list.

## What was asked, and why this shape

The owner's stated ORDER was distil FIRST into the trunk, then fit labels warm from it. Both previous
distill arms mixed a teacher term into a normal labelled fit at weight 0.3 (`distill_pairs` +0.0016,
`raptor_distill` +0.0035) and landed inside the 0.0062 floor, which says nothing about the order
because the order was never varied. `DistilledObjective.__call__` is additive, so weight 20 makes the
teacher term almost all of the gradient and the twelve label logits a probe rather than a target.

Stage one: swinbase, 4 epochs, `--distill-weight 20.0` against `raptor_teacher.npz`. Reached 0.8177
on its own holdout (0.7578 → 0.7923 → 0.8123 → 0.8177, still rising when the budget ended —
`convergence_check` calls it UNDERTRAINED, which is a real caveat and is stated below).

Stage two: that encoder exported and fitted on labels, 12 epochs, everything else `swin_supervised`'s
line.

## Measured

| | mean |
|---|---|
| `swin_supervised_s2026` (cold control) | 0.8331 |
| `swin_distill_fit_s2026` (warm from the distilled trunk) | **0.8255** |
| delta | **-0.0077** |

Per label against the control: Contusion -0.0288\*, Lateral OA -0.0225\*, MCL +0.0201, Fracture
-0.0194, Medial Meniscus -0.0168\*, ACL -0.0139, PF OA -0.0138, Effusion -0.0127, Baker's +0.0101,
Lateral Meniscus +0.0077, Medial OA -0.0059, Synovitis +0.0038. Half-widths reach ±0.0429. Three
starred negatives, zero starred positives.

## The harness branch fired, and it came back clean

The arm pre-registered `<= -0.0062` as SUSPECT THE HARNESS FIRST, so that was checked before anything
was written:

- Stage one's recorded recipe carries `distill_weight: 20.0` and a non-null `teacher_targets`. The
  teacher term was live.
- Stage two's recorded recipe carries `encoder_weights` pointing at the exported encoder, with
  `distill_weight 0.0` and `teacher_targets None`. It was genuinely warm and genuinely label-only.
- `arm_read` prints the diff itself and it is ONE flag: `encoder_weights None→…enc.pt`. Nothing else
  moved between the two arms.
- The export wrote 423 tensors from the 0.8177 checkpoint, so the trunk that was fitted is the trunk
  that was distilled.

Only after those does the negative read as a fact about the method rather than about the plumbing.

## What it means, stated narrowly

**A trunk driven to imitate a teacher's twelve output logits is a WORSE initialisation than the
pretrained trunk it started from.** That is not surprising once said plainly: the teacher targets are
head-level, so what the trunk was asked to become is a feature extractor whose readout happens to
reproduce raptor's label mapping. Fitting labels from there starts inside a basin shaped by another
model's decision boundary, and the -0.0077 is the cost of climbing back out of it.

## What this does NOT close

knee-8009 **(b)** — feature/representation distillation, label-free — is untouched, and that was
written down before the run rather than after it. (b) distils the teacher's REPRESENTATION, not its
verdict; the failure named above is specifically a failure of the logit target, and it is the reason
(b) is a different arm rather than a rerun of this one.

Two further caveats that a later reader should not have to rediscover:

- Stage one is UNDERTRAINED. Four epochs, still rising at the last one. A longer stage one might reach
  a better trunk; it would still be a logit-shaped trunk, so it changes the size of this number and
  not its sign.
- The control is COLD by design, so this arm never separated "distilled trunk" from "warm start at
  all". It did not need to: the arm lost, and a losing warm start is a losing warm start either way.
  A positive would have owed that follow-up.

## Cost

~30 min stage one + ~3.3 h stage two, 0 slots. Bought as an INSTRUMENT and it paid as one: it retires
the ORDER question (distil-then-fit, at the only teacher target this project owns) without spending a
submission, and it licenses nothing about (b) in either direction.
