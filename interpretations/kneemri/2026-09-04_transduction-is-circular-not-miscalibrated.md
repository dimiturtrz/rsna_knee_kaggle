# Transductive adaptation buys nothing, and the reason is circularity — not the labelling scheme

**What was measured.** Forty cells: five trunks × two pseudo-labelling schemes × four tail sizes, on the
fold-0 holdout (909 studies). `kneemri/evaluation/transduction.py`, run 2026-09-04, log in
`logs/transduction_prev.log`.

**The result.** No cell in either scheme clears +0.0005, and none clears the 0.0062 noise floor in
either direction. The correct reading is **no gain**, not damage.

## The suspect the module named against itself, and why it was worth an hour

The 2026-09-03 negative on this axis did not stop at the number. Its own docstring named a harness
suspect: the fixed-tail scheme ties TAIL SIZE and IMPLIED PREVALENCE to a single knob. At share 0.25, a
finding present in 5% of studies is handed twenty percentage points of invented positives, and the refit
is then fitting that invention rather than the test distribution. That is a real defect, and it is
exactly the class of thing that turns a good idea into a bad measurement — a faulty-harness reading, not
a refutation.

So the scheme became a strategy (`kneemri/evaluation/labelling.py`) and the alternative got built:
`PrevalenceTail` caps the positive tail at the finding's own training prevalence, leaving `share` to mean
confidence and nothing else. Two knobs where there was one. The negative tail is untouched, because a
study in the bottom `share` is confidently negative whatever the prevalence.

**The suspect is refuted.** The two schemes track each other everywhere:

| trunk | fixed (0.05 / 0.1 / 0.25 / 0.4) | prevalence (same shares) |
| :--- | :--- | :--- |
| dinov2-large | −0.0036 / −0.0026 / −0.0020 / −0.0029 | −0.0036 / −0.0022 / −0.0021 / −0.0015 |
| v3large | −0.0058 / −0.0033 / −0.0021 / −0.0027 | −0.0058 / −0.0029 / −0.0009 / −0.0016 |
| vjepa2-large | −0.0042 / −0.0031 / +0.0005 / +0.0003 | −0.0042 / −0.0035 / −0.0015 / −0.0012 |
| ortholarge | −0.0055 / −0.0038 / −0.0033 / −0.0030 | −0.0055 / −0.0042 / −0.0031 / −0.0032 |
| radnet121@focal-max | −0.0047 / −0.0028 / −0.0004 / −0.0015 | −0.0047 / −0.0034 / −0.0022 / −0.0019 |

Fixing the prior distortion did not move the axis. Whatever is stopping this from paying, it is not the
shape of the invented label distribution.

## What the shape says instead

The delta shrinks toward zero as the tail GROWS, and it does so under **both** schemes. That is the
signature of circularity, not of a prior:

- Pseudo-labels read off a ridge lie along that ridge's own decision direction. Adding more of them pulls
  the refit back toward the first fit, so the delta goes to zero as the tail grows.
- A small tail moves the fit least but adds only its most extreme rows — which is where a linear fit is
  most wrong. So the small-tail end is where the damage concentrates, and it does, uniformly:
  every trunk's worst cell is at share 0.05, in both schemes.

A linear head refitted on its own confident predictions learns what it already held. Measured, not
argued.

## What this does and does not close

**Closes:** naive self-training with a linear head under ZERO shift. The holdout here is a fold of our
own corpus, so the simulation deliberately sets the distribution shift to zero and bounds the mechanism
from above. It came back at zero, and it cost no card and no submission slot.

**Does not close:** test-time adaptation with a MEMBER head under the real test set's actual shift,
which is the case a kernel would be in. That is a different head (non-linear, high capacity) and a
different amount of shift (non-zero, and unmeasured). Recorded as a measured negative on the cheapest
version of the axis, not as a closed lane. Re-opening it means answering the circularity argument, not
re-quoting a number.

**Bears on:** any future proposal to pseudo-label the Kaggle test set. The circularity argument applies
to it unchanged, and applies harder to a head that can memorise its own labels.

## Method note

The suspect was named by the module that failed, before anyone tried to defend the result — which is why
it was cheap to test. Writing the leading harness suspect into the artefact at the moment of the negative
is what let the follow-up be forty CPU-minutes instead of a re-derivation.

Related: `kneemri/evaluation/transduction.py` docstring (the numbers), `knee-tta1` (the axis).
