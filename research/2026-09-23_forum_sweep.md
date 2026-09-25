# Forum + public-code sweep — 2026-09-23 ~20:07Z

Delta against `2026-09-13_host_rulings_forum_sweep.md` and `2026-08-31_public-frontier-sweep.md`. Pages read
through r.jina.ai; quotes marked near-verbatim where they passed a summariser. Unreadable: 740610, 742050,
742190, 733965.

## Board (read 20:07Z)
Top 0.958 (Scott Willis, rhoskeri), 0.956 ×2, 0.955 ×8, 0.954 ×7, #20 0.953. Top was 0.952 on 08-31.

## Rulings and data
- **741819 (staff, ~09-21):** external data passes the RULES if free and self-serve, without institutional
  sign-off; staff will not judge the dataset's own DUA. Replies read OAI as needing sign-off, so no.
  Changes nothing for us: the owner's 2026-09-15 ruling (knee-vgz5b.7) is stricter.
- **741916 (staff):** a host update meant to fix corrupted train DICOMs "may have failed".
- **742122:** hidden-rerun failure on series with mixed slice sizes (a robustness trap; our entries score).

## Claimed mechanisms, priced against our record
| Claim (source) | Our record |
|---|---|
| Slice DENSITY, not span: 44→80 slices 0.917→0.932 (737696, Raptor CoAtNet window reader) | Slice count 12→24 flat, 24→36 −0.0224 on our DINOv2 whole-stack (AXES:87-88). Architecture-bound claim. |
| Eval windows 48→72 → board 0.944 (737696) | Window TTA; our whole-stack members gain nothing from overlap TTA (ledger). |
| Train/inference pooling mismatch costs ~0.02 (737696) | Serving matches training by fingerprint + witness checks. |
| RadImageNet R50, centre slice ×3 beats adjacent-slice channels +0.020 (Ziad, fold 0, 737696) | CNN lane closed on measurement (`interpretations/kneemri/2026-09-05_the-cnn-axis-is-maximally-distinct-and-still-worthless.md`); there is no DINOv2-only rule. |
| InstanceNumber against position in >1/3 of series; laterality missing in ~half (741310) | Already handled: order by `ImagePositionPatient·normal`, InstanceNumber only as fallback (`core/data/corpus/acquisition.py:340-372`); laterality is Type 2C-aware (`:12`). |
| Flip aug must swap medial/lateral labels (741310) | We don't flip-augment across laterality (no flip in `core`/`kneemri`). |
| Fracture back into RadImageNet blend, +0.0118 Fracture AUC (notebook) | Ensemble-blend knob; not a solo mechanism. |
| "Fast Parent 0.957" notebooks | NOT a board score: in-sample macro AUC on gold58 (`local_fullset_diagnostic_macro_auc`). |

Nothing here is a solo DINOv2 mechanism priced near +0.015 that the record lacks.
