# Q-BT — fastMRI+ boxes: licence gate and auxiliary-loss formulation

Claude's questions (2026-09-02), corrected: the fastMRI+ corpus is already
adapted in `core/data/corpus/fastmri_corpus.py`; only the box geometry
(`x, y, width, height`) is currently unused. The questions are therefore:

1. Can a model **trained on fastMRI+** ship in a Kaggle kernel?
2. Which auxiliary-loss formulation carried the measured delta in top solutions?
3. Did published gains require boxes on the **same studies** being scored?
4. Is there published evidence that deriving a severity threshold from box
   geometry changes a downstream number?

## 1. Licence — can fastMRI-trained weights ship in the kernel?

### 1.1 NYU fastMRI Data Sharing Agreement

The full agreement is at https://fastmri.med.nyu.edu/. Key clauses (quoted from
the page, emphasis original):

- §2: "NYU hereby grants to me a **non-exclusive, royalty-free license to use
the fastMRI Dataset for internal research or educational purposes only** ...
This Agreement conveys no other rights of any sort with respect to the fastMRI
Dataset or the intellectual property rights embodied therein."

- §5(c): "**Not DISTRIBUTE, PUBLISH, REPRODUCE, RETRANSMIT, COMMERCIALLY
EXPLOIT, COPY OR OTHERWISE TRANSFER any portion or all of the fastMRI Dataset
or any subsequent variables or data files derived from the fastMRI Dataset** to
any person or entity other than individuals under my direct supervision and
other than for uses in academic publications and presentations with citations."

- §5(d): "Require anyone who receives or has access to the fastMRI Dataset to
agree to the same restrictions and conditions."

- §11: "When I have completed my work using the fastMRI Dataset, I will
**destroy all copies of the fastMRI Dataset** and retain no copies thereof."

The agreement never explicitly mentions "model weights" or "trained models".
The decisive phrase is "any subsequent variables or data files derived from the
fastMRI Dataset". A trained neural-network weight tensor is plausibly a
"variable ... derived from" the dataset. The DUA's purpose is to keep the data
inside the registered user's organisation; shipping weights that encode the data
out via a public Kaggle kernel is the opposite of that purpose.

### 1.2 fastMRI+ annotations licence

The fastMRI+ paper (*Scientific Data* 2022, doi:10.1038/s41597-022-01255-z) is
published **CC-BY 4.0**. The annotations themselves are hosted on GitHub
(https://github.com/microsoft/fastmri-plus) and on Synapse
(doi:10.7303/syn26345776). The GitHub repo does not carry a separate data
licence file, but the paper's CC-BY status strongly suggests the annotations are
intended to be open access.

However, the annotations are **derivative of fastMRI images**. The NYU DUA
§5(c) bars transferring "any subsequent variables or data files derived from the
fastMRI Dataset". Even if Microsoft released the boxes under CC-BY, the boxes
were produced from DUA-gated images. NYU could argue that the boxes themselves,
or any model trained with them, remain subject to the DUA.

### 1.3 Kaggle external-data rules

The RSNA Knee Abnormality Detection rules allow "external data that is **freely
and publicly available to all participants**" and explicitly allow pre-trained
models. A model trained on data that requires an individual NYU registration and
DUA acceptance is **not freely and publicly available** in the sense the rules
intend: another entrant cannot reproduce the training without separately
negotiating access.

### 1.4 Precedent

No Kaggle competition solution was found that shipped weights trained on
fastMRI data. The closest analog is RadImageNet, which the project already
dropped because the weights were request-gated and unlicensed for public
redistribution.

### 1.5 The weights-vs-images distinction

The DUA does not distinguish weights from images. It restricts "any portion or
all of the fastMRI Dataset or any subsequent variables or data files derived
from" it. The conservative legal reading is that **shipping fastMRI-trained
weights in a public Kaggle kernel violates the DUA**. The risk is not a licence
violation on the model file format; it is redistribution of derived information.

**Conclusion:** do not ship fastMRI-trained weights in the Kaggle kernel without
written permission from NYU. Training locally for internal research is fine;
redistribution is not.

## 2. Auxiliary-loss formulation that carried the measured delta

The canonical published implementation is Nischaydnk's **1st-place RSNA 2023
Abdominal Trauma Detection** solution:
https://github.com/Nischaydnk/RSNA-2023-1st-place-solution.

Their setup:

- **Stage 1:** a 3D segmentation model produces **dense organ masks** for every
  slice.
- **Stage 2:** a 2.5D CNN+GRU classifier takes the masked/cropped volumes and
  predicts study-level labels.
- **Auxiliary head:** two segmentation heads attached to the classifier encoder
  (U-Net decoder + 2D-CNN head on last and second-last feature maps).
- **Loss:** `BCE(outputs, targets) + 0.125 * (Dice(mask_outputs, mask_targets) +
  Dice(mask_outputs2, mask_targets))`.
- **Inference:** the segmentation heads are **dropped**; only the classifier
  outputs are used.
- **Reported gain:** "around **+0.01 to +0.03** boost in our models."

Critical differences from our fastMRI+ boxes:

- Their masks were **dense and per-slice** (every slice has a full organ mask).
- Their masks and classification labels were on the **same studies** (RSNA 2023
  training set).
- The auxiliary task was **multi-class organ segmentation**, not sparse bounding
  boxes.

## 3. Did the published gains require boxes on the same studies?

Yes. The RSNA 2023 setup, and every other top solution with auxiliary
segmentation that reports a measured delta, used spatial labels on the same
studies that carried the final classification labels. The gain comes from the
classifier learning localisable features for the exact pathologies it is scored
on.

Our situation is different:

- fastMRI+ boxes are on **fastMRI knee studies**.
- Our labels are on **RSNA knee studies**.
- The auxiliary loss would fire only on the fastMRI half of a mixed batch.

No published medical-imaging result was found where sparse bounding-box
supervision from a **different dataset** gave a measured downstream
classification gain. Transfer of spatial supervision across corpora is an open
question, not a proven lever. This is the most likely reason the route does not
work as quoted.

## 4. Severity threshold from box geometry

Claude's added question: our OA criteria require roughly ≥1 cm findings. Box
`width * height * pixel_spacing^2` makes that threshold computable, turning
unlabelled rows into confident labels.

No published paper was found that specifically derives a severity threshold from
bounding-box geometry and reports a downstream classification delta. The idea is
reasonable and cheap to measure: it is a rule-based label augmentation, not a
new model. It does not raise any licence issue beyond what using the annotations
already raises.

## Summary and recommendation

1. **Do not ship fastMRI-trained weights** in the Kaggle kernel. The NYU DUA's
   "derived variables" clause and the Kaggle "freely and publicly available"
   rule both point against it.
2. **The auxiliary-loss +0.01-0.03 figure does not transfer** to our setting
   because it relied on dense masks on the same studies we score.
3. **The cheap, licence-safe experiment** is to use the unused box geometry to
   derive a size threshold and relabel fastMRI+ rows as positive/negative for
   the OA-size criterion. Measure whether that changes the downstream AUC on a
   member trained with the existing fastMRI+ labels. This requires no new
   licence beyond what the corpus already uses and no auxiliary head.
4. If the size-threshold experiment moves the needle, then it is worth asking
   NYU for explicit written permission to ship weights trained on fastMRI+;
   until then, keep any fastMRI-trained member local-only.

## Sources

- NYU fastMRI DUA: https://fastmri.med.nyu.edu/
- fastMRI code licence (MIT): https://github.com/facebookresearch/fastMRI/blob/main/LICENSE.md
- fastMRI+ paper and annotations: https://www.nature.com/articles/s41597-022-01255-z,
  https://github.com/microsoft/fastmri-plus
- Nischaydnk RSNA 2023 1st place solution:
  https://github.com/Nischaydnk/RSNA-2023-1st-place-solution
- RSNA Knee Abnormality Detection rules (external data clause):
  https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules
