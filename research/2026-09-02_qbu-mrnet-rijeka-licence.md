# Q-BU — MRNet and Rijeka licences: can merged-corpus members ship?

Claude's question (2026-09-02): the structural argument that disqualified fastMRI
trained weights may also reach `merged_blend_cf`, a champion seat, because it was
trained on arms that pull both MRNet and the Rijeka `kneeMRI` set. The question
is whether either corpus permits a model trained on it to be shipped in a public
Kaggle kernel.

## 1. MRNet — Stanford University School of Medicine Research Use Agreement

The full agreement is in `competitions/mrnet/agreement.html` in the Stanford ML
Group site repository, served at download registration time. Key clauses,
quoted:

- §1: "Permission is granted to view and use the MRNet Dataset **without charge
  for personal, non-commercial research purposes only**. Any **commercial use,
  sale, or other monetization is prohibited**."

- §4: "**YOU MAY NOT DISTRIBUTE, PUBLISH, OR REPRODUCE A COPY** of any portion
  or all of the MRNet Dataset to others without specific prior written
  permission from the School of Medicine."

- §6: "You must not modify, reverse engineer, decompile, or **create derivative
  works from the MRNet Dataset**. You must not remove or alter any copyright or
  other proprietary notices in the MRNet Dataset."

- Closing paragraph: "You may not otherwise **copy, reproduce, retransmit,
  distribute, publish, commercially exploit or otherwise transfer any
  material**."

Source: [Stanford ML Group site repo — agreement.html](https://raw.githubusercontent.com/stanfordmlgroup/stanfordmlgroup.github.io/master/competitions/mrnet/agreement.html)

Unlike fastMRI's DUA, MRNet does **not** contain the explicit phrase
"subsequent variables or data files derived from" the dataset. It is silent on
model weights specifically. It is not silent on derivative works (§6 prohibits
them) or on commercial use / distribution (§1, §4, closing paragraph). Whether a
trained weight tensor is a "derivative work" is the same interpretive gap that
exists for fastMRI; the conservative reading is the same: shipping MRNet-trained
weights in a public, prize-bearing Kaggle kernel is outside the permitted uses.

## 2. Rijeka kneeMRI — CC BY-NC-ND 4.0

The dataset moved from the original University of Rijeka page to Zenodo. The
Zenodo record states:

> "This dataset is licensed under a Creative Commons
> Attribution-NonCommercial-NoDerivatives 4.0 International License
> (CC BY-NC-ND 4.0)."

Source: [Zenodo record 14789903](https://zenodo.org/records/14789903)

CC BY-NC-ND 4.0 has two clauses that matter for Kaggle use:

- **NC (NonCommercial)** — the licence "does not include uses that are primarily
  intended for or directed toward commercial advantage or monetary compensation."
  A competition with a $25,000 prize pool is not safely non-commercial.

- **ND (NoDerivatives)** — the licence "does not include ... Adapted Material."
  A neural-network weight tensor trained on the images is unambiguously an
  Adapted Material under the licence definitions. ND allows redistribution only
  if no changes are made.

So Rijeka is, if anything, clearer than MRNet: shipping Rijeka-trained weights
is not permitted under the stated licence.

## 3. Kaggle enforcement precedent

The only documented Kaggle enforcement action I found in the recent literature is
a stroke dataset removed for **copyright infringement** (Retraction Watch /
casrai, July/August 2026). I found no documented case of Kaggle disqualifying or
forcing a rewrite specifically over a **registration-gated research-use corpus**
or a CC-BY-NC-ND corpus.

That absence is worth stating plainly: it means the risk is **rule-based and
licence-based**, not **precedent-based**. The RSNA rules require external data
to be "freely and publicly available to all participants". A corpus that
requires Stanford registration plus acceptance of a research-use agreement, or a
Zenodo corpus under NC+ND terms, does not meet that bar. Whether Kaggle has ever
enforced it for this specific shape of corpus is unknown.

## 4. Implications for the champion

This is the reseating question Claude reserved for himself. The licence facts
above imply:

- If MRNet-trained weights are treated as derivative works or as outside
  "personal, non-commercial research", then every shipped member trained on a
  merged arm that includes MRNet is exposed. `merged_blend_cf` is in that set.

- If Rijeka-trained weights are treated as Adapted Material under CC BY-NC-ND,
  the same merged arms are exposed from the other side.

- The exposure is not theoretical only because the member has been on the board
  in more than thirty submissions.

The actual score cost of replacing `merged_blend_cf` with the best non-merged
member is a holdout measurement Claude will make.
