# External knee-MRI data access audit

This note is the practical companion to the external-data and 3D-model research notes. It lists the knee-MRI corpora we have mentioned, how to actually download them, what their licenses require, and whether they can legally feed a Kaggle submission. The goal is to turn the Scheme B recommendation into a concrete procurement checklist.

## Summary table

| Corpus | Size | Access route | License / restriction | Usable for Kaggle? |
|--------|------|--------------|----------------------|--------------------|
| MRNet | 1,370 exams | Stanford AIMI / competition page | Research-use agreement; various sources list CC-BY or public domain, but official page requires agreement | Likely yes, if agreement allows competition use |
| OAI | ~4,800 subjects, many timepoints | NDA via Login.gov | Public-domain NIH data | Yes |
| Rijeka / KneeMRI | 917 volumes | Zenodo / Kaggle | **CC BY-NC-ND 4.0** on Zenodo; the Kaggle mirror declares `unknown` | **Not in a selected entry** — 2.6a passes, rule 2.5's winner-redistribution obligation is the open question (§3) |
| fastMRI knee | ~1,000+ scans | fastMRI.med.nyu.edu | Custom NYU agreement, research/education | Likely yes, verify competition use |
| SKM-TEA | 155 scans | Stanford AIMI | Custom Stanford AIMI agreement | Likely yes, verify competition use |
| KMAR-50K | 1,444 scans | Mendeley Data | CC-BY (open access) | Yes |

## 1. MRNet (Stanford)

- **What:** 1,370 knee MRI exams from Stanford University Medical Center (2001-2012), with labels for abnormal, ACL tear, and meniscal tear. Train/val/test splits are public; the official test set is hidden and evaluated on Codalab (competition now closed).
- **Where:** [Stanford ML Group competition page](https://stanfordmlgroup.github.io/competitions/mrnet/) and [Stanford AIMI catalog entry](https://aimi.stanford.edu/datasets/mrnet-knee-mris).
- **Access:** The official download requires signing a research-use agreement on the Stanford AIMI portal. Some third-party sources (e.g., TensorFlow Datasets issues) list the license as CC-BY or public domain, but the canonical path is the AIMI agreement.
- **Competition risk:** The Kaggle host rule 2.6a generally requires external data to be "freely and publicly available." MRNet is publicly available under an agreement, but we need to confirm the agreement does not restrict competitive/commercial use. The bead `knee-2sbk` already flags this uncertainty.
- **Action:** Download and read the AIMI Data Use Agreement before any model trained on MRNet is shipped to Kaggle.

## 2. Osteoarthritis Initiative (OAI)

- **What:** Multi-center longitudinal knee OA study with ~4,800 participants, MRI (sagittal DESS), X-rays, clinical data, and MOAKS/WORMS-derived structural labels.
- **Where:** [NDA NIH OAI portal](https://nda.nih.gov/oai/).
- **Access:** Register via [Login.gov](https://login.gov/) on the NIH NDA site, request data permissions under "My Account → Data Permissions," add studies to a filter cart, and create a data package. Approval is usually fast for public datasets.
- **License:** Public-domain NIH data. No commercial restriction.
- **Competition risk:** Low. OAI is explicitly public and unrestricted.
- **Action:** Create an NDA account and request the baseline sagittal DESS MRI package plus KL grades. This is the safest external corpus to start with.

## 3. Rijeka / KneeMRI dataset

- **What:** 917 sagittal PD-weighted knee MRI exams from Clinical Hospital Centre Rijeka (2006-2014), labeled for ACL injury.
- **Where:** [Zenodo record 14789903](https://zenodo.org/records/14789903) and [Kaggle mirror](https://www.kaggle.com/datasets/sohaibanwaar1203/kneemridataset).
- **License:** **CC BY-NC-ND 4.0** — non-commercial and no-derivatives.
- **Competition risk:** **Medium, and NOT via 2.6a — this line said "almost certainly violate 2.6a" until 2026-09-07 and that was wrong.** D75 §2 quotes 2.6a's own words: external data must be public, equally accessible to all participants, and available at no cost. Rijeka passes all three. The real exposure is rule **2.5**, which makes the WINNER license the winning Submission *and its source code* under CC-BY-NC 4.0 — and weights fitted on ND-licensed images are exactly the artefact an ND clause makes hard to redistribute cleanly. That is an ambiguity in the winner's obligation, not a breach of the external-data rule.
- **What the mirror declares — TWO first-party reads that disagree, both recorded rather than resolved.** The rendered page, read through a text proxy (D75 §1, 2026-09-07), carries "This dataset is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)". The Kaggle API's metadata for the same slug (`kaggle datasets metadata -d sohaibanwaar1203/kneemridataset`, 2026-09-07 20:46Z) returns `"licenses": [{"name": "unknown"}]` — the machine-readable licence field is EMPTY, and the description it returns carries only the acknowledgement request (Clinical Hospital Centre Rijeka, Štajduhar et al. 2017), not the CC sentence. The structured field is what a downstream tool would read; the prose is what a human sees. Neither is evidence of a MORE permissive term, so this changes nothing about the decision — it is recorded because a future check reading the API field will see `unknown` and must not conclude the licence question was ever open. The Zenodo record's `metadata.license` is unambiguous: CC BY-NC-ND 4.0.
- **Action:** Do not use Rijeka in any shippable model unless the host explicitly waives the restriction. It can be used for *internal method development only* if we keep the model out of submissions. Update `comp/members.declines` with this reasoning.

## 4. fastMRI knee

- **What:** Fully anonymized knee MRI raw k-space and DICOM images from NYU Langone Health, originally released for accelerated MRI reconstruction.
- **Where:** [fastMRI.med.nyu.edu](https://fastmri.med.nyu.edu/).
- **Access:** Request access through the NYU fastMRI portal; a custom agreement must be signed.
- **License:** Royalty-free license for internal research and educational purposes (custom NYU agreement).
- **Competition risk:** Medium. The agreement is research-oriented; we must verify it permits use in a public competition with prize implications.
- **Action:** Read the fastMRI Data Use Agreement. If it restricts competitive use, either decline or request a waiver.

## 5. SKM-TEA (Stanford)

- **What:** 155 clinical quantitative 3D knee MRI scans with dense segmentations (cartilage, meniscus) and pathology bounding boxes. Two tracks: raw k-space and DICOM images.
- **Where:** [Stanford AIMI dataset page](https://stanfordaimi.azurewebsites.net/datasets/4aaafb9-c6e6-4e3c-9188-3aaaf0e0a9e7) (link from SKM-TEA paper) and [GitHub](https://github.com/StanfordMIMI/skm-tea).
- **Access:** Create a Stanford AIMI account, log in, and download. Dataset is 900 GB compressed / 1.6 TB expanded.
- **License:** Custom Stanford AIMI agreement.
- **Competition risk:** Medium, same as MRNet and fastMRI — verify the agreement allows competition use.
- **Action:** The main value of SKM-TEA is not scale (only 155 scans) but *dense anatomy labels*. It is most useful for pretraining a segmentation-aware 3D model or CSNet-style surface network, not for supervised 12-label classification.

## 6. KMAR-50K

- **What:** 1,444 multi-parameter knee MRI scans from 1,190 patients, paired with motion-artifact and artifact-free images.
- **Where:** [Mendeley Data](https://data.mendeley.com/datasets/xw7mrg7ntg).
- **Access:** Direct download from Mendeley Data after registration.
- **License:** Open-access / CC-BY (verify exact version on download page).
- **Competition risk:** Low. CC-BY is the most permissive common research license.
- **Action:** Useful for SSL / reconstruction pretraining and for data augmentation (motion-artifact simulation). Not labeled for our 12 pathology classes, so treat it as unlabeled data for Scheme A.

## 7. Kaggle rule 2.6a reminder

Kaggle competitions typically require that any external data used in a submission be:
- freely and publicly available to everyone, and
- usable under its license for the competition purpose.

A dataset that requires a research-use agreement is not automatically disqualified, but it must not impose restrictions that make it less available than the competition data itself. Custom agreements need a line-by-line check.

**"NC and ND licenses are generally disqualifying" stood here until 2026-09-07 and is not what 2.6a says.** 2.6a's three tests are public, equally accessible, no cost — a licence term restricting what the USER may then do with the data is not one of them. What NC/ND actually bites is rule **2.5**, the winner's obligation to license the winning Submission and its source code under CC-BY-NC 4.0. See the D75 appendix below for the quoted clauses; §3 restates Rijeka on that basis.

## 8. Recommended procurement order

1. **OAI** — lowest risk, largest volume, public domain. Start here for Scheme B.
2. **KMAR-50K** — low risk, large volume, good for SSL / augmentation.
3. **MRNet** — moderate risk, verify agreement; valuable labels overlap our task.
4. **fastMRI knee** — moderate risk, verify agreement; mostly unlabeled but high-quality volumes.
5. **SKM-TEA** — moderate risk, small volume, high label density; useful only if we pursue 3D segmentation pretraining.
6. **Rijeka** — high risk (CC BY-NC-ND); decline for submissions unless host waives.

## 9. Local inventory update

Most of the corpora above are already on disk under `D:/data/volumetric/mri/raw/`. This changes the practical next step from "download" to "inspect format and decide license risk."

| Corpus | Local path | Size on disk | Format on disk | Notes |
|--------|-----------|--------------|----------------|-------|
| MRNet | `D:/data/volumetric/mri/raw/mrnet/` | 7.3 GB | 1,250 exams × 3 planes as `uint8` `.npy` stacks (axial ~44×256×256, sagittal/coronal ~36×256×256) + `train/valid-{abnormal,acl,meniscus}.csv` | Already preprocessed and ready to load. Only license check blocks use. |
| Rijeka / KneeMRI | `D:/data/volumetric/mri/raw/kneemri_rijeka/` | 3.1 GB | 917 volumes as `uint16` `.pck` files (32×320×320) + `metadata.csv` with `aclDiagnosis` and ROI coords | License remains CC BY-NC-ND; data is available for internal experiments only. |
| fastMRI knee | `D:/data/volumetric/mri/raw/fastmri/` | 194 GB | 973 train + val `singlecoil_*.h5` files; `labels/Annotations/knee.csv` has 16,167 bounding boxes across 974 files and 21 pathology labels | This is the strongest immediate lever — labels map directly to RSNA classes (ACL, meniscus, cartilage, bone lesion, effusion, etc.). |
| SKM-TEA | `D:/data/volumetric/mri/raw/skmtea/` | 19 GB | 155 DICOM studies + annotations + segmentation masks | Small but dense; useful for 3D segmentation pretraining. |
| KMAR-50K | `D:/data/volumetric/mri/raw/kmar50k/` | 67 GB | ~3,000 `.nii` volumes across `ArtifactData_*` and `GroundTruthData_*` | Unlabeled for pathology; use for SSL/reconstruction/motion augmentation. |
| knee_clinical_seq | `D:/data/volumetric/mri/raw/knee_clinical_seq/` | unknown | DICOM series (batch2) | Unknown labels; needs inventory. |
| RSNA label tables | `D:/data/volumetric/mri/raw/knee_label_tables/` | unknown | Multiple CSVs with LLM/consensus labels and calibrated soft targets for RSNA studies | Potential pseudo-label / teacher sources for Scheme C. |

### fastMRI knee labels are the big surprise

`fastmri/labels/Annotations/knee.csv` contains 16,167 bounding-box annotations over 974 knee scans, with labels such as:

- Meniscus Tear (5,658 boxes)
- Cartilage — Partial / Full Thickness loss/defect (2,985 + 615)
- Joint Effusion (1,311)
- Bone — Fracture/Contusion/dislocation (1,060)
- Bone — Subchondral edema (986)
- Ligament — ACL Low Grade sprain / High Grade Sprain (765 + 678)
- Ligament — MCL, PCL, LCL sprains
- Bone — Lesion, Soft Tissue Lesion, etc.

These map well onto the RSNA 12-label set. Because the boxes are slice-local, they can be used for partial-label supervised pretraining (Scheme B) without requiring full 3D reconstruction. The NYU license still needs verification, but the data is already local.

### Revised procurement order given local state

1. **fastMRI knee** — already local, rich labels, strongest immediate lever. Verify NYU license first.
2. **MRNet** — already local and preprocessed. Verify Stanford AIMI license.
3. **KMAR-50K** — already local, use for SSL/augmentation.
4. **SKM-TEA** — already local, use for 3D segmentation pretraining if pursued.
5. **OAI** — not yet local (to our knowledge), but public-domain and low-risk; worth downloading for volume.
6. **Rijeka** — already local, but license likely bars competition use.

## 10. What we still need to verify

- Exact text of the Stanford AIMI agreement for MRNet and SKM-TEA.
- Exact text of the NYU fastMRI agreement.
- Whether the RSNA 2024 host has published an explicit external-data whitelist or FAQ.
- Whether Rijeka can be used under a separate research-only agreement without the NC/ND clauses.
- Whether `knee_clinical_seq` has any labels and what its license is.

The next concrete step is no longer downloading — it is reading the fastMRI and MRNet data-use agreements, then building a dataloader that can consume the local `.npy` / `.h5` / `.pck` / `.nii` formats under Scheme B.


---

## 2026-09-07 20:50Z — licence audit appendix (researcher; quotes verbatim, no ship ruling)


To main (copy mastermind). Quotes verbatim from the fetched pages (Kaggle pages read through the
`r.jina.ai` text proxy — direct fetch returns the JS shell). No ship recommendation, as asked. The
append to `research/2026-09-02_external_knee_mri_data_access_audit.md` and the `licence` bead are your
writes — appendix text ready to paste is in `inbox/main/attach/d75_licence_appendix.md`.

## 1. The Kaggle mirror's licence, quoted

Local copy: `D:/data/external/kneemri_rijeka/kneemridataset.zip` (no `dataset-metadata.json` beside it;
slug recovered from the zip name). Page: <https://www.kaggle.com/datasets/sohaibanwaar1203/kneemridataset>
(title "kneeMRIdataset", owner Sohaib Anwaar). Licence field and description:

> This dataset is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
> International License (CC BY-NC-ND 4.0).

> If you are using this dataset in your work, please acknowledge the source (Clinical Hospital Centre
> Rijeka, Croatia) and reference this paper

Upstream, same terms — Zenodo record 14789903 "KneeMRI dataset" (Štajduhar, Mamula, Miletić, Ünal),
<https://zenodo.org/api/records/14789903>, `metadata.license`:

> Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)

So the mirror does NOT carry a different licence from upstream: both say CC BY-NC-ND 4.0.

## 2. Rule 2.6 and 2.5, quoted from the competition rules page

Source: <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules>, §2.6 EXTERNAL DATA
AND TOOLS. (Our own `research/2026-09-02_fastmri_mrnet_competition_use_licence_audit.md:11,13` quotes
2.6a/b with the same words.)

2.6.a:
> You may use data other than the Competition Data ("External Data") to develop and test your Submissions.
> However, you will ensure the External Data is either publicly available and equally accessible to use by
> all Participants of the Competition for purposes of the competition at no cost to the other Participants,
> or satisfies the Reasonableness criteria as outlined in Section 2.6.b below.

> The ability to use External Data under this Section does not limit your other obligations under these
> Competition Rules, including but not limited to Section 2.8 (Winners Obligations).

2.6.b:
> The use of external data and models is acceptable unless specifically prohibited by the Host.

> Because of the potential costs or restrictions (e.g., "geo restrictions") associated with obtaining rights
> to use external data or certain software and associated tools, their use must be "reasonably accessible
> to all" and of "minimal cost".

> ...the costs of potentially procuring a license for software used to generate a Submission, must also be
> considered.

2.5 WINNER LICENSE (the clause that touches NC/ND directly):
> You hereby license and will license your winning Submission and the source code used to generate the
> Submission under CC-BY-NC 4.0, an Open Source Initiative-approved license...

> In the event that input data or pretrained models with an incompatible license are used to generate your
> winning solution, you do not need to grant an open source license in the preceding Section for that data
> and/or model(s).

### NC and ND separately — the two readings, both quoted above, neither resolved here

- **NC (NonCommercial).** Reading A: 2.6a's test names three properties — publicly available, equally
  accessible to all participants, no cost — and Rijeka meets all three (public Zenodo + public Kaggle
  mirror, free). NC is not one of the three words, and 2.5 explicitly anticipates "input data ... with an
  incompatible license" among WINNING solutions without disqualifying them. Reading B: 2.6a's phrase
  "to use ... for purposes of the competition" — a prize competition is arguably commercial use under CC's
  NC definition ("primarily intended for or directed towards commercial advantage or monetary
  compensation"), so a participant cannot lawfully "use" the data for the competition, and 2.6a's "you will
  ensure" is then unmet. The rules text does not say which reading the host takes.
- **ND (NoDerivatives).** Reading A: a trained model weight is arguably not a "derivative" of the images in
  CC's sense (the licence governs the licensed material and adaptations of it; CC's own FAQ on AI training
  is non-committal), and Kaggle rules nowhere mention ND. Reading B: `merged_blend_cf` is a set of weights
  fitted on the images and a model trained on Rijeka is distributed (kit dataset + winner licence), which ND
  forbids ("if you remix, transform, or build upon the material, you may not distribute the modified
  material"). 2.5's carve-out says the incompatible-licence INPUT need not be re-licensed; it says nothing
  about whether the derived weights may be distributed.
- Our 2026-09-02 audit line "NC and ND clauses almost certainly violate Kaggle rule 2.6a" is a
  PARAPHRASE: 2.6a contains neither "non-commercial" nor "derivative". The bar it states is
  public / equally accessible / no cost. Whether NC reaches it goes through "for purposes of the
  competition", not through the clause's own words. Ambiguous; both readings quoted.

## 3. MRNet — Stanford Research Use Agreement, and the NAME LIST

RUA quote (from `research/2026-09-02_fastmri_mrnet_competition_use_licence_audit.md:25`, itself quoting
stanfordmlgroup.github.io/competitions/mrnet):
> 1. Permission is granted to view and use the MRNet Dataset without charge for personal, non-commercial
> research purposes only. Any commercial use, sale, or other monetization is prohibited.
plus (:40) "non-commercial + no redistribution + no derivative works".

Members whose training set includes MRNet images or MRNet-derived labels:

| member | how MRNet enters | on disk (`D:/models/rsna_knee_abnormality`) | roster / declines |
|---|---|---|---|
| `mrnet_acl_pretrain_s2026` | trunk pre-trained on MRNet ACL task | yes | neither |
| `mrnet_warm_s2026` | `mrnet_warm.sh:52 --init mrnet_acl_pretrain_s2026.pt --init-scope trunk` | yes | declines:298 |
| `merged3_corpus_s2026` | `--extra-scheme mrnet` (README:332), Rijeka + MRNet images | yes | roster:613 |
| `noisy_student_s2026` | student of `teacher_merged3` (teacher trained on merged3) | **no .pt** | — |
| `merged_blend_dm`, `calib_span_dm`, `lr4e-5_ws_dm` | `teacher_merged3` distill (merged3_tranche_one) | **no .pt** | — |

Members whose training set includes RIJEKA images (`--scheme merged`, `merged_blend_v1.csv`):
`merged_blend_cf` (+ `_n12b0.1-0.9`, `_r224`, `_r280`, `_r392` variants), `merged_blend_s2026`,
`merged_dense_cf`, `merged_span_cf`, `merged_span_ff`, `merged_corpus_s2026`, `merged_labelwise_cf`,
`merged_distill_cf`, `merged_blend_df` (`retrain_six_distill.sh`, `--scheme merged $CALIB $DISTILL_SIX
--labels "$MERGED_BLEND"`), `blend_ws`/`blend_calib`/`blend_ladder`/`blend_through`/`blend_base`
(`blendsix.sh:98`), `merged_smoke`. Distill-tainted only (teacher trained on Rijeka, student on our
images): `*_dm` above. NOT tainted (checked): `final_ws_df`, `calib_span_df`, `grade_ladder_df`,
`through_dense_df`, `lr4e-5_ws_df` — `--labels "$LABELS"`, teachers `teacher_oof5`/`teacher_clean_full`
are our own fold twins.

## 4. The other three foreign sets, declared licence quoted

| set | Kaggle slug | licence field (verbatim) |
|---|---|---|
| RaptorAI weights | `dreaddevelopment/raptor-knee-widedense` ("Knee MRI model weights: twelve findings") | "CC0. Use them for anything, no attribution required." — trained on 4,349 competition studies + LLM soft labels; "No external data was used." |
| mattiaangeli | `mattiaangeli/rsna-knee-coat-resgated-ep10-top3` | "CC0: Public Domain" |
| pilkwang LLM labels | `pilkwang/rsna-knee-llm-labels` | "CC0: Public Domain" |

`D:/data/external/raptor/fetch.sh:5-6` pulls `dreaddevelopment/$d`, so
`raptor/labels/labels_llm_soft.csv` (consumed by `foreign_labels_dense.sh:53`, `hisrecipe_distill*.sh:62`)
is a dreaddevelopment (CC0) artefact. Our own re-upload `dimiturnt/mattia-all-public` declares
`"licenses": [{"name": "CC0-1.0"}]` (`D:/data/external/mattia_all_public/dataset-metadata.json`). No
`dataset-metadata.json` exists under `kaggle_foreign/*` — licences above are from the live pages.

## 5. Inventory — where a Rijeka/MRNet-trained member sits

**Champion** (`comp/champion.json`): `raptorweight_v1`, kernel `kneemri-raptorweight`, submission
55875622, board 0.944 — members `final_ws_cf, calib_span_ff, grade_ladder_cf, merged_blend_cf,
lr4e-5_ws_cf, through_dense_cf`. Contains `merged_blend_cf`.

**Roster** (`comp/members.roster`): :390 `merged_blend_cf`, :420 `merged_blend_s2026`, :438-440
`blend_ws`/`blend_calib`/`blend_ladder`, :469 `merged_dense_cf`, :490 `merged_span_cf`, :559
`merged_corpus_s2026`, :588 `merged_span_ff`, :613 `merged3_corpus_s2026` (MRNet+Rijeka).
**Declines**: :72 `merged_smoke`, :115 `merged_distill_cf`, :128 `merged_labelwise_cf`, :298
`mrnet_warm_s2026`.

**Kernel `chosen` tuples** (`comp/kernels/*/*.py`) — 47 kernels name a merged member; unless noted the
member is `merged_blend_cf`:
aucvote, bend, blendseat (`merged_blend_s2026`), blendthree (`_s2026`), calibmix (`_cf`+`_s2026`),
calibsix, champviews, cleancalib, colbaker, colhalf, colmcl, colsyn, corpusdose (`_cf`, `merged_dense_cf`,
`merged_span_cf`, `merged_span_ff`), corpusfull (`_cf`, `_s2026`, `merged_corpus_s2026`, `merged_dense_cf`,
`merged_span_cf`, `merged_span_ff`), distinct14 (`_cf`, `merged_dense_cf`), eff224, fiveship, fleettwo,
foreignlogit, foreignnine, foreignseat, foreignthree, foreigntwo, greedyseat (`merged_span_cf`), halfcham,
halfseat, holdswap, honest12 (`_cf`, `merged_dense_cf`), lensseat, monotone, onegroup, oofseat, pixgroup,
proxyseat, rankvote, rapthalf, raptorseat, raptorweight, relabel7, resseat (`_s2026`), rootmove, spanseat
(`_cf`, `merged_span_ff`), **stack**, stackvote, twinchamp (`_s2026`), twinfive, uncalib, vjepaseat.
No kernel seats an mrnet/merged3/noisy_student/`_dm` member.

Colweight: `inbox/main/attach/d63_W8.csv` row `merged_blend_cf` (mass 1.81); the clean refit without it
is `d63_W8_clean_lambda0.75.csv` (D74 §3).

— researcher, 20:50Z
