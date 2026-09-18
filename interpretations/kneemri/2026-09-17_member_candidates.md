# Candidate members, ledger75, 909-study pool, foreign raptor_* EXCLUDED (98.6% in-sample, knee/board 2026-08-29)
# 382 honest members with a record. Champion seats with a record: 3 of 7.

## Champion's measurable seats
```
0.9053  labelwise_depth24_drop40_long_s2026
0.9048  labelwise_dense_drop40_long_s2026
0.9045  labelwise_ext_drop40_long_s2026
```

## Top 25 honest solo
```
0.9053  SEAT labelwise_depth24_drop40_long_s2026
0.9051       labelwise_drop_drop40_long_s2026
0.9048  SEAT labelwise_dense_drop40_long_s2026
0.9045  SEAT labelwise_ext_drop40_long_s2026
0.9044       labelwise_depth24_negpull_s2026
0.9030       labelwise_dense_drop40_long36_s2026
0.9021       dinov3f0_solo
0.9018       roinative_labelwise_drop40_long_s2026
0.9007       labelwise_dense_stem_llrd_s2026
0.9004       recal_holdannot_s2026   WRONG RULER - 0.8888 on the corpus table, see the banner below
0.9000       warm_calib_lowlr_drop40_long_s2026_run1
0.8998       labelwise_depth24_s2026
0.8994       warm_calib_lowlr_drop40_long_s2026
0.8986       fieldwide_s2026
0.8986       labelwise_extended_s2026
0.8984       roi_labelwise_drop40_long_s2026
0.8984       labelwise_drop_s2026
0.8982       chainfour_drop40_long_s2026
0.8978       clean_calib_drop40_long_s2026
0.8974       dropstack_light_s2026
0.8972       dropstack_long_s2026
0.8971       foreign_distill_calib_s2026
0.8970       labelwise_depth20_b10_s2026
0.8964       labelwise_dense_s2026
0.8964       foreign_dinov3_matched
```

## Per-finding leader among honest members, and its margin over the best CHAMPION SEAT

**CORRECTED 2026-09-18T01:14Z — EVERY `recal_holdannot_s2026` ROW IN THIS TABLE IS VOID.** Its holdout
truth is `report_labels_v3.csv`, whose fourth rung (0.6879) falls on the other side of the 0.75 recut
from the corpus table's (0.82), so it was graded against a different positive set than the members it is
compared to here. Re-scored on the corpus truth it reads **0.8888, not 0.9004**, and it loses ELEVEN of
twelve columns to `headlr_ship_s2026` — it leads nothing. The six leads below are the ruler.
`comp/members.declines:84` had already ruled this number out and this table quoted it anyway;
`LedgerSeventyFive.population` now refuses the comparison outright (knee-g2vm9). Evidence:
`interpretations/kneemri/2026-09-18_the_cut_is_not_what_exhausted_our_ruler.md`.
ACL                    0.9692 recal_holdannot_s2026                        +0.0226 over labelwise_dense_drop40_long_s2
MCL                    0.9472 recal_holdannot_s2026                        +0.0423 over labelwise_ext_drop40_long_s202
Medial Meniscus        0.9364 recal_holdannot_s2026                        +0.0282 over labelwise_ext_drop40_long_s202
Lateral Meniscus       0.9218 recal_holdannot_s2026                        +0.0252 over labelwise_depth24_drop40_long_
Medial OA              0.9298 foreign_labels_dense_s2026                   +0.0037 over labelwise_depth24_drop40_long_
Lateral OA             0.8827 foreign_dinov3_matched                       +0.0007 over labelwise_ext_drop40_long_s202
PF OA                  0.9021 labelwise_depth24_drop40_long_s2026          SEATED
Effusion               0.9420 ctl_nofloor_s2026                            +0.0023 over labelwise_ext_drop40_long_s202
Synovitis              0.9291 fieldwide_s2026                              +0.0077 over labelwise_dense_drop40_long_s2
Baker's                0.9599 recal_holdannot_s2026                        +0.0410 over labelwise_depth24_drop40_long_
Contusion              0.9147 recal_holdannot_s2026                        +0.0073 over labelwise_dense_drop40_long_s2
Fracture               0.9253 dinov3f0_solo                                +0.0830 over labelwise_depth24_drop40_long_
## The DINOv3 seat is a leak, not a trunk (2026-09-17 06:4xZ)

On the champion's three measurable seats (`champ3` = 0.9100, ledger75, 909 studies):

| addition | champ3 + it | delta | provenance |
|---|---|---|---|
| `dinov3f0_solo` | 0.9215 | **+0.0116** | VOID — `members.declines:736`, 703/909 in-sample |
| `foreign_dinov3_tta3` | 0.9204 | +0.0104 | VOID — same served weights |
| `foreign_dinov3_matched` | 0.9196 | +0.0096 | VOID — same served weights |
| both of the above | 0.9267 | +0.0167 | the leak compounds |
| `v3base_dense_s2026` | 0.9078 | **-0.0022** | ours |
| `v3small_control_s2026` | 0.9072 | -0.0028 | ours |
| `v3large_stack_s2026` | 0.9071 | -0.0029 | ours |
| `lora_r8_v3small_224_s2026` | 0.9068 | -0.0032 | ours |
| `v3large_dense_s2026` | 0.9063 | -0.0037 | ours |
| `labelwise_drop_drop40_long_s2026` | 0.9104 | +0.0004 | ours, same family as the seats |
| `recal_holdannot_s2026` | 0.9100 | 0.0000 | ours, and on a MOVED RULER - void, see the banner above |

Every DINOv3 member that ADDS is in-sample on 703 of the 909. Every DINOv3 member that is OURS
subtracts. The isolation that made this look like a mechanism — "two independent DINOv3 members both
carry ~+0.01 while same-family DINOv2 members carry nothing" — was not an isolation: both carriers are
`mattiaangeli`'s released fold weights, one family, one leak. The DINOv2 controls reading flat is the
clean-vs-leaked contrast, not the same-trunk-vs-different-trunk one.

This is the `raptor_solo` mechanism a second time (`foreign_overlap`, 98.6% in-sample, 2026-08-29), and
it was caught by `members.declines` rather than by any measurement — the combine reads beautifully and
says nothing. **A candidate's provenance is read BEFORE its delta, not after.**

What this does NOT close: whether a DINOv3 trunk trained HERE would carry diversity. Our best honest
DINOv3 member is 0.8724 against seats at 0.9045-0.9053; a member 0.03 below the pool adding nothing is
what the seat curve predicts for any weak member, so the trunk has never been tested at seat strength.
That is a GPU question (train a DINOv3 member to ~0.90), not a seating question, and it is the only
form in which this axis is still open.
