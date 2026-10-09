# Levers before the final picks (2026-10-09)

Final picks are due 2026-10-22, and teams can merge until 10-15. Read the board numbers from `comp/board.jsonl` and
the pin from `comp/champion.json`. The numbers here are the KEYS to grep for, as of 19:00Z today.

## Where we stand

- Champion `mirrorseat_v1` reads board 0.953. It blends our seven (logit mean) at 1/2 with nartaa's CoAtNet 384
  at 1/2, a rank blend.
- Silver and bronze both cut at 0.950, where ~440 teams are tied. Gold cuts at 0.960, the top 21. The leader reads 0.964.
- The gap that matters is +0.007 to gold. Holding the medal means not dropping below 0.950 on PRIVATE.

## Closed today, and the mechanism behind each

| Lever | Read | Why it closes |
|---|---|---|
| foreign-seat SHARE (1/3, 1/2, reader in or out) | 0.952 / 0.952 / 0.953 | Flat at board resolution. The share is a knob worth ±0.001, so `mirrortwo`, `mirrorthird` and `mirrortilt` stay unspent |
| reliability-gated share by stratum | no stratum flips | `2026-10-09_no-stratum-flips-the-blend-share.md` |
| JEV / field label tables | gold58 0.9020 vs ours 0.8995 | in-sample to the 58, and gold gains have not transferred before (README:17448) |
| more public families | heliosli dead (weights retired); MaxViT fold0 flat | knee-dbj03 |
| distilling OUR ensemble into one model | ceiling +0.0049 | the labelwise trio's logit mean is 0.9102 vs best solo 0.9053 ledger75 on 909 rows. A perfect student gains less than the bar |

## The finding: the offline ruler is blind to the field's single-model lever

The forum's top single models (735304) are 2.5D ConvNeXt-T at 0.953 and DINOv2 ViT-S at 0.951. Both are trained on
**pseudo labels**, with no external data. Their author: "model is not the key. it is your labels". Our best solo is ~0.946.

Our holdout truth IS the v2 report table. A lever that moves the training target away from the report labels is
scored against the very thing it moves away from. It reads null or negative offline by construction, which is the
"faulty TEST" case. Every distillation null in the ledger (AXES §D) was read on that ruler.

This was already reasoned out on 2026-09-30 and filed as **knee-p0s4f** (P1, arm `lw24_targetblend.sh` written,
blend table built leak-free and calibrated). The 15:13Z note on knee-dbj03 called the "labels are the key" lever
spent. That conflated choosing a label TABLE (closed above) with training on a model-derived soft target (untested
on the board).

What tempers it:
- `oofsolo_v1` 0.926 vs `holdsolo_v1` 0.928 on the board. That was one read, with a leaky teacher, on the 0.85-era
  recipe.
- `hisrecipe_distill` pulled its member ONTO our pool (corr 0.871 → 0.89). That is the Deotte caveat: it can lift
  a solo while lowering its ensemble value.
- P(board solo > 0.946) ~0.10. It is an INSTRUMENT on the ruler question first.

A stronger teacher than `teacher_clean` now exists: the champion blend itself, which includes nartaa's
OAI-supervised trunk. Relabelling the 4,407 train studies with it would run in a Kaggle GPU kernel with no local
card. But a student trained on it is foreign-corpus-derived, and the 09-15 ruling says that is instrument-only and
never shipped. That needs a new ruling (see below).

## Owner decisions this blocks on

1. **GPU go** for `lw24_targetblend` (~1.6 h card). It also needs **72 GB of disk** to rebuild the 24-slice cache
   (`images_336px_24sl_*` is still absent today), or a re-anchor on 20sl with its own control, which costs two
   arms.
2. **Can a student distilled from the champion blend (which includes nartaa's OAI seat) be shipped?** It is a
   different question from the blending approval already given.
3. **Folds for OOF teachers.** The field's 0.959 is 5-fold. Folds are banned today.

## Recommendation for the two final picks

- **Pick 1:** the board-best entry (today `mirrorseat_v1`, or `mirrorpair_v1` if it re-pins).
- **Pick 2:** a hedge that is structurally different on the axis most at risk of public overfit, the foreign
  seat. Take the best entry with ours carrying a larger share (`triseat_v1`-style), not a second copy of pick 1's
  structure. Two picks that agree on everything hedge nothing.
