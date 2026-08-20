# The fitting recipe every arm shares, in ONE place. Sourced by lib.sh AND stub_lib.sh.
#
# WHY IT IS ITS OWN FILE. `kneemri.experiments.arm_check` sources each arm against stub_lib.sh, so anything defined
# only in lib.sh would be EMPTY under the check - and an unset variable expands to nothing, which means
# the checker would validate an invocation missing exactly the flags it was meant to prove. Both libs
# source this, so there is one definition and the check sees what the run sees.
#
# WHY IT EXISTS AT ALL. Twenty arms repeated these flags by copy-paste, and the drift has already cost:
# knee-7tq, the final full-data fit, said it would run "with whatever recipe survives" and needed FOUR
# corrections when audited, because no survivor had ever propagated into it. It would have shipped a
# model worse than members already on disk.
#
# HISTORY IS NOT PRESERVED HERE, DELIBERATELY. An arm that ran under an older recipe is recoverable from
# git; keeping a second, frozen copy of the flags in the tree to protect that would reintroduce the
# duplication this removes. One dev, no live system: fix it, and read the old invocation out of history
# if it is ever needed.

# Every flag below was measured, not assumed:
#   --fold 0        the shared split, so members are directly averageable (folds and seeds are banned)
#   --one-cycle     defined over total_steps; compressing the schedule cost 0.0028 (schedule_8)
#   --weight-floor  the confidence floor the published baseline uses
#   --rank-weight   the pairwise term, the only loss change positive in 3 of 3 matched arms
RECIPE="--fold 0 --one-cycle --weight-floor --rank-weight 1.0"

# Free throughput, measured on this card: gradient checkpointing trades 19% per-step for nearly all the
# encoder's activation. It does not change what is computed.
#
# COMPILE IS NO LONGER HERE because it is no longer optional: `--compile-encoder` was a flag twenty-two
# of thirty-six arms simply forgot to name, paying 21% of every step for the omission, so the trainer
# compiles by default and `--eager-encoder` is the escape.
THROUGHPUT="--recompute-activations"

# The early stopper, DERIVED rather than picked: `python -m kneemri.experiments.stale_gap` reads every
# curve on disk for the longest NON-improving streak a run later recovered from, and patience must
# EXCEED that or it stops a run that was about to get better. Across 56 runs that streak is 3 epochs
# (lora_r8_v3small_224_s2026: 16 epochs, best at 12, survived 3), so the safe value is 4.
#
# SEVEN ARMS PASSED 3, which is exactly the value that fires on that run's third stale epoch — the one
# before its best. resolution.sh carried a comment claiming 3 could not change a recorded best; that
# replay was run before the curve that set the floor existed. One constant here so the next arm inherits
# the measurement instead of copy-pasting a number that ages.
#
# NOT FOR EVERY ARM. An arm whose question IS the schedule (band_dense_long, aug_long) is measuring the
# shape of the tail, and a stopper deletes the measurement; final_fit has no holdout under --fold -1, so
# there is nothing for a stopper to read. Re-derive after a run with a longer recovered stall lands.
PATIENCE="--patience 4"
