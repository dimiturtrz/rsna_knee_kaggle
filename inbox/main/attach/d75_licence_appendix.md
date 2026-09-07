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
