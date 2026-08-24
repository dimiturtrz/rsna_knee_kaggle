# We have no denominator that both measures the right truth and resolves a per-label gap — 2026-08-24

Written three hours after `2026-08-24_the_goal_priced_against_the_inventory.md`, and it corrects that
note's second section. The correction was found by accident: a routine look for a member whose Lateral
Meniscus strength the ensemble might be averaging away returned numbers that could not be reconciled
with the ones I had published that afternoon.

## The two numbers that cannot both be about the same thing

```
Lateral Meniscus, this project's members     0.8453     (median over 44 members)
Lateral Meniscus, quoted in the goal note    0.6497     ("ours are very uneven")
```

They are not about the same thing, and the goal note's attribution — "ours" — is wrong. `0.6497` comes
from `kneemri.evaluation.supervision_gap`, and its docstring is explicit about what it measures:
**PUBLISHED members, scored against RADIOLOGIST ANNOTATIONS, on the studies `train.csv` annotates.** Our
own members cannot be scored there at all, because they are fitted with the annotated studies upweighted
and would be scoring their own training data. So the goal note took a table about somebody else's models
on somebody else's ruler and introduced it as a description of ours.

## The denominator, which is the part that decides everything downstream

`train.csv` annotates **58 studies**. For Lateral Meniscus that is **23 positives and 35 negatives**.

A Hanley-McNeil standard error at that split, at AUC 0.65, is **0.0754** — a 95% halfwidth of **±0.148**.
Across all twelve findings the halfwidths run **0.14 to 0.21**:

```
finding             n1   n0   95% halfwidth        finding             n1   n0   95% halfwidth
ACL                 24   34       0.146            Effusion            35   23       0.142
MCL                  9   49       0.209            Synovitis           27   31       0.143
Medial Meniscus     26   32       0.144            Baker's             12   46       0.185
Lateral Meniscus    23   35       0.148            Contusion           19   39       0.156
Medial OA           15   43       0.169            Fracture            18   40       0.159
Lateral OA          11   47       0.192            PF OA               21   37       0.151
```

The whole spread across the twelve findings on that ruler is 0.6497 to 0.9547 — **0.305, about twice a
typical halfwidth.** So the table barely resolves its own ordering, and the goal note's proposal reads,
in its own units, as moving Lateral Meniscus to the **top of its confidence interval**:

> *"Lateral Meniscus alone, moved 0.65 → 0.80, is +0.0125 on the mean — very nearly the whole gap."*

0.80 is inside [0.50, 0.80]. That is not a deficit with a lever under it; it is an interval.

This was not unknown here. `2026-08-20_label_contest_pilot.md` says in its own words that each AUC is
"drawn from ~58 studies with a per-label SE" and refuses to read an ordering off the table. Four days
later I read an ordering off the same table and planned the project's next move with it.

## What the resolvable denominator says instead, and it says something different

Our own holdout scores report labels rather than annotations, but it scores every study rather than 58.
Median per-label over the 44 converged members at or above 0.83:

```
MCL               0.7858        Synovitis         0.8648
Fracture          0.8051        Contusion         0.8685
Lateral OA        0.8154        Medial OA         0.8722
PF OA             0.8399        ACL               0.8739
Lateral Meniscus  0.8453        Medial Meniscus   0.8973
Effusion          0.8523        Baker's           0.9020
```

**Lateral Meniscus is mid-pack — fifth from the bottom, not the floor.** The worst three are MCL,
Fracture and Lateral OA. And the shape of the problem changes with the ordering: the spread here is
**0.116**, not 0.305, and lifting the worst label to the median of the rest is worth **+0.0066 on the
mean** — at the 0.0062 noise floor, and less than a third of the +0.015 the leaderboard gap costs.

**This ordering was checked against the same objection that killed the other one**, because MCL had the
smallest annotation count of the twelve (n1=9) and "the worst label is the rarest label" is exactly the
artefact I had just finished criticising. It survives. The report labels are soft 5-level values
({0.08, 0.28, 0.68, 0.82, 0.94}); binarised at 0.5 over 4406 studies, a 20% holdout carries:

```
finding            holdout pos   95% hw        finding            holdout pos   95% hw
Fracture                    62    0.061        Contusion                  259    0.031
Synovitis                  116    0.045        Medial OA                  323    0.029
MCL                        153    0.040        PF OA                      398    0.026
Lateral OA                 217    0.034        Medial Meniscus            469    0.025
Baker's                    222    0.033        Effusion                   533    0.025
Lateral Meniscus           224    0.033        ACL                        245    0.032
```

**0.025 to 0.061, against the annotation ruler's 0.14 to 0.21** — four to six times finer, and the 0.116
label spread is about three times a typical halfwidth. MCL at 0.7858 ± 0.040 sits genuinely below a pack
median near 0.865. Only Fracture (± 0.061, the rarest) is too wide to place confidently, and it is not the
one the ordering turns on. So the ordering here is real in a way the annotation ordering is not.

So on this ruler there is no single label carrying the gap. The "one focal structure is nearly the whole
deficit" story is a property of a 58-study table, not of the model.

## The structural problem, which is the actual finding

Neither ruler can settle it, and they fail in opposite directions:

| ruler | truth | studies | can it resolve a per-label gap? |
|---|---|---|---|
| annotations (`supervision_gap`) | radiologist — **the competition's own truth** | 58 | no, ±0.15 |
| our holdout | LLM-read report labels — a proxy, ~65% agreement on positives | all | yes, but not against the truth that scores us |

**This project has no measurement that is both the right truth and large enough to rank findings.** That
is why per-label reads keep producing orderings that disagree with each other — the crop80 per-label
gate, the 2026-08-20 re-read, and now this. It is one missing denominator showing up as three separate
puzzles.

Worth saying that one cross-ruler agreement does survive and is not noise-sized: **MCL is a ligament at
the medial margin, and `crop80_span` lost precisely there (MCL −0.0265\*, ACL −0.0263\*) because an 80 mm
crop cuts the margin out of the field.** The worst label on the resolvable ruler and the label the crop
experiment destroyed are the same structure, which is a mechanism rather than a coincidence.

## What this changes

- **Delete the per-label targeting argument from the goal note.** It ranked the inventory on an
  unresolvable table. The rest of that note — the closed-inventory arithmetic and the +0.0082 pricing —
  does not depend on it and stands.
- **Do not build for Lateral Meniscus.** It is not the floor on the only ruler that can see a floor.
- **Do not build for MCL either, on this evidence alone.** +0.0066 for lifting the worst label is inside
  the noise floor, and the proxy-truth objection applies to that ordering too.
- **The one thing worth buying is the denominator.** More annotated studies — or a defensible way to
  score our own members against the existing 58 without scoring their training data — would convert three
  unresolvable puzzles into one measurement. That is infrastructure, not a knob, and it is the first item
  in a long time that is sized like the goal rather than like a tweak.

  **Priced 20 minutes later, and the aggregate half of that recommendation does not survive** — see the
  next section, which was written against a measurement this repo already contained.

## Then the denominator was priced, and 58 studies cannot see the goal either

The paragraph above proposed buying more annotated studies. Before filing that as work I ran
`kneemri.evaluation.ruler_transfer`, which already builds exactly that instrument for the **published**
members — the ones not fitted on the annotations — and therefore already runs on all 58 rather than on
the 13 our own members hold out. It asks a different question from `supervision_gap`: not whether the two
rulers agree in LEVEL, but whether a DELTA between two arms survives the change of ruler.

```
58 annotated studies, 58 of them with a held-out prediction; 1000 paired resamples
     arms    report              annot               ratio
     1->2   +0.0021 [-0.0087,+0.0142]   +0.0041 [-0.0072,+0.0153]   +1.97 [-2.03,+3.31]
     2->4   +0.0043 [-0.0037,+0.0124]   +0.0054 [-0.0025,+0.0139]   +1.27 [-1.24,+3.38]
perm 0.40   +0.0430 [+0.0183,+0.0692]   +0.0730 [+0.0501,+0.0996]   +1.70 [+1.07,+3.48]
   share    report     annot    ratio across 5 permutation seeds
    0.05   +0.0006   +0.0017   +0.97 [+0.97,+0.97]
    0.10   +0.0010   +0.0032   +0.98 [+0.75,+1.22]
    0.20   +0.0078   +0.0114   +1.17 [+0.75,+1.64]
    0.40   +0.0577   +0.0635   +1.15 [+0.79,+1.70]
```

**Two readings, and they cut opposite ways.**

**The resolution.** A delta on the annotation ruler carries a 95% halfwidth of about **0.011**. That is
finer than the per-label ±0.15 — a mean over twelve labels is a better-behaved statistic than any one of
them — but it sets a hard floor on what 58 studies can adjudicate. Both REAL comparisons here are the
axis this project spends most of its slots on, adding members, and both land at +0.002 to +0.005 with
intervals spanning zero **on both rulers**. Only a large synthetic perturbation resolves at all. So the
instrument can gate "is this a result at all" at roughly this house's own 0.015 bar, and cannot rank
anything below it. **First place costs +0.0082 offline. The ruler cannot see the thing the goal is made
of.** That is the argument against spending a fitting run to unlock it, and it is why `knee-74uo` went
P1 → P3 rather than being worked.

**The direction, which is worth more than the bead.** Wherever the ratio resolves it is **above 1**: the
large perturbation gives +1.70 with an interval excluding 1, and the share sweep runs 0.97 / 0.98 / 1.17 /
1.15 as the perturbation grows. **The report ruler UNDERSTATES deltas relative to radiologist truth
rather than overstating them.** If that holds for real gains and not only for permuted ones, our offline
numbers are a conservative floor on what the board sees — the same direction as `board_transfer`'s +1.82,
reached from an entirely independent construction. That is the first cross-check that transfer constant
has ever had.

**And the caveat that keeps it a direction.** The only rows that resolve are permutation perturbations —
predictions degraded on purpose, not models improved. A degradation and an improvement need not travel
between rulers at the same rate. **Do not put 1.70 in a document as a constant**; that is precisely how
4.67 / 2.14 / 1.92 came to be quoted for three months while the tool read +1.82. It is a sign, not a size.

The structural table above is unchanged by this and gets one row sharper: the annotation ruler cannot
resolve a per-label gap (±0.15) **and cannot resolve an aggregate arm delta below ~0.011 either**. What it
can do is confirm a result that already cleared the house bar, on the competition's own truth, for free —
it costs no GPU. That is a real use, and it is a smaller one than "the first item sized like the goal".

## What I did wrong, since that is the reusable part

I quoted a number from a document instead of from the tool that produces it, and the document had
dropped the qualifier the tool's own docstring carries in its first paragraph. Then I attributed
somebody else's models to us, and read an ordering off a table whose own note four days earlier says the
ordering is unreadable. Three failures, one shape: **the number travelled and its denominator did not.**
