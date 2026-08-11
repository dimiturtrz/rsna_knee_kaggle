# What actually moved the number — and what conspicuously did not

*Fold 0, every arm matched on seed, data and fold; scored on the pilkwang label table so that the
training-label arms earn no self-agreement credit. Baseline is the recipe that produced our 0.868
leaderboard submission.*

## The table

| change | holdout | delta | verdict |
|---|---|---|---|
| baseline (constant LR, no augmentation) | 0.8092 | — | what we submitted at 0.868 |
| + OneCycle schedule | 0.8153 | **+0.0061** | shipped |
| + augmentation actually applied | 0.8202 | **+0.0110** | shipped — it was never called |
| + intensity augmentation (gamma/blur/contrast/noise) | 0.8221 | +0.0019 | kept, not defaulted |
| better label table (v4_blend, +0.028 vs gold) | 0.8194 | −0.0008 | **neutral** |
| 32 inference slices instead of 12 | 0.8137 | **−0.018** | refuted |
| overlap TTA (inference only) | +0.0035 | +0.0035 | shipped |

Five folds on the shipped recipe: **0.8269** against **0.8162**, +0.0107, every fold improved.

## The two findings that were bugs, not ideas

**Augmentation was written, unit-tested, and never called.** `fit_epoch` did not invoke it. Both
leaderboard submissions and all five original folds trained unregularised. The evidence was in a curve
nobody had looked at: a 40-epoch run peaked near epoch 16 and then declined monotonically to 0.8043,
which is what unregularised training looks like, and explains how the public baseline trains 60 epochs
where we degraded after 16.

Both quality gates passed it. `vulture` saw the method as *used* — because its own test referenced it.
The mirror gate saw it as *tested* — because it had a `test_augment`. **A method can be fully covered
and completely disconnected: every gate checked the callee and none checked the caller.** The fix was a
test on `fit_epoch` asserting it routes pixels through augmentation.

**The data path read four times what it used.** `images[batch][:, :, s:s+3]` fancy-indexed first,
materialising all twelve cached slices (97.5 MB per batch) before discarding three quarters. One
indexing operation instead: 35 ms → 3.3 ms, and the epoch fell from ~40 s to ~30 s.

## Four public ideas that needed adapting, not copying

Each came from a real, measured result elsewhere. None transferred unchanged, and the *reason* was
different every time.

1. **Flips** (cardiac-seg augments with both axes). A cardiac label is not side-defined; five of our
   twelve targets are. A horizontal flip would undo the laterality normalisation they depend on.
2. **Translation** (cardiac-seg measured it NEUTRAL and disabled it). Their encoder is a
   fully-convolutional U-Net and therefore translation-equivariant; ours is a ViT with position
   embeddings and is not. The measurement cannot be inherited — only the method for making it.
3. **Label fusion** (the 0.903 system's headline mechanism). Fusing our four public readers scores
   0.8966, *below* the best single reader at 0.8982. Their fusion presumably worked because their
   private Qwen reader was independent of pilkwang's; ours are all derived from the same reports, and
   the best table is already a blend.
4. **32 inference slices** (the 0.903 recipe trains at 12, infers at 32). Worth −0.018 here. Our
   encoder eats three consecutive slices *stacked as channels*, so raising the slice count over a fixed
   band shrinks each group's physical thickness and shows the model an input distribution it never saw.
   Their pooling is max over independently-encoded slices, which has no such coupling. The idea is
   conditional, not dead: it needs the group's physical extent held fixed, or per-slice encoding.

## Two hypotheses of mine that the data refuted

**Laterality was not the cause of the medial/lateral asymmetry.** Both medial/lateral pairs favour
medial (−0.043 meniscus, −0.066 OA), which is exactly what a broken mirror would produce. But the
asymmetry is *identical* across tag-resolved and geometry-resolved studies (the two rules agree 98.6%
of the time on 2,135 studies), and it is **smaller on the knees we actually mirror** (+0.015 against
+0.071). A broken mirror does the opposite. The cause is most likely prevalence: medial pathology is
commoner and more often named in a report.

**Label quality did not become model quality.** The tables differ by +0.028 in agreement with the 58
gold studies and the models differ by −0.0008. Both readers work from the *same reports*, so their
disagreement concentrates on studies whose reports are genuinely ambiguous — where neither label is
reliable and the image cannot settle it either. Unbiased confirmation needs a cross-fitted evaluation
on the gold studies, which costs five folds per arm.

## Two methodology errors worth more than the gains

**Different rulers.** The first ablation queue would have scored the v4_blend arm against v4_blend
targets and the baseline against pilkwang targets. A holdout AUC rises when labels agree with
*themselves* more, and that would have read as a better model. Forty-five minutes of running GPU were
killed to add `--eval-labels` and restart. The ruler chosen is deliberately the one that *disfavours*
the interesting arm.

**A refactor moved a boundary the deployment path depended on.** Extracting `Submission` was right on
every local measure — LCOM4, testability, 95% duplication between two kernels — and it broke the next
kernel run with `ModuleNotFoundError`, because the kit carrying our source was not rebuilt.
`verify_kit.py` catches this in a second and was skipped, because the change felt like it was "only in
the kernels". The kernels *import* the kit; a change to either is a change to both.
