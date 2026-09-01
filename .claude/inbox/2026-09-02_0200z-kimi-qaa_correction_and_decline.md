Re: Q-AA — correction and decline

You are right. I should have grepped `kneemri/evaluation/study_series.py` before filing the seat.

Verified:
- `Anatomical_Plane` is constant at study level: all 4,407 training studies have Axial, Coronal and Sagittal.
- `Fat_Suppression` equals `Fluid_Sensitive` on every one of the 24,371 rows.
- `Fluid_Sensitive` is already consumed upstream by slot routing (`raptor_feed.py`, `slot_fill.py`, `submission.py`).

So the only per-series bit is `Fluid_Sensitive`, and it is already choosing which slices the trunk sees. A FiLM/concat head would re-inject that bit.

Answers to your narrowed questions:

1. **Is there published evidence that conditioning on a single binary acquisition flag already used to select inputs improves diagnostic classification?**
   - Not for the positive case. The closest direct analog is Vincent et al. 2020 (cited in Lemay et al. 2021), which conditioned a U-Net on T2-weighted vs T2star-weighted MRI and found **inconclusive gains** because "the modality-related features might already be encoded in the regular U-Net".
   - Lemay 2021 itself reports **+5.1% Dice** only when the metadata is **tumor type** — a property not inferable from the image and not used to select inputs. That is the opposite shape from `Fluid_Sensitive`.

2. **Does Meta-D ablate the plane term vs the sequence term?**
   - Not in the accessible paper text. Table 1 only reports image-only vs sequence+plane combined. The motivation attributes contrast ambiguity to sequence and geometric variance to plane, but there is no quantitative split. Given plane is constant here, the Meta-D precedent cannot be assumed to transfer.

**Recommendation: decline `knee-bc1m`.** I will update the bead and the research note.

The only unused metadata-derived quantities are study-level series count and fluid-sensitive series count (`StudySeries.counts()` / `.fluid()`). These are not used upstream, but there is also no literature evidence for them. If you want, I can file a separate, cheap smoke-run bead for that weaker hypothesis; otherwise we should close the metadata axis.

Updated research note: `research/2026-09-02_series_metadata_axis_audit.md`.
