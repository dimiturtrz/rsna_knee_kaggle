# The fitting recipe every arm shares, in ONE place. Sourced by lib.sh AND stub_lib.sh.
#
# WHY IT IS ITS OWN FILE. `kneemri.arm_check` sources each arm against stub_lib.sh, so anything defined
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

# Free throughput, all measured on this card: gradient checkpointing trades 19% per-step for nearly all
# the encoder's activation, and compile is -21% per step. Neither changes what is computed.
THROUGHPUT="--recompute-activations --compile-encoder"
