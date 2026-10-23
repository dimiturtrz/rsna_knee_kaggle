# External knee corpora — what is on disk, what joins, what it is worth

Every number this project owns is measured on a 909-study holdout carved from the SAME corpus the
members trained on. That is the gap these corpora exist to close, and it is a different job from
"more training data" — most of what follows will never enter a training set.

Credentials for all three live OUTSIDE the repo under a credentials directory (`$CREDS` below), `chmod 600`, and are
read by the fetch scripts rather than embedded. Never echo one into a log or a chat.

## SKM-TEA (Stanford AIMI) — the external validation set

- Access: a time-limited Azure blob **SAS URL** for the `qdess` container, saved at
  the file `SKMTEA_URL_FILE` names. `sp=rl` is read+list only. Stanford recommends AzCopy or
  Storage Explorer; **neither is installed here**, and neither is needed — the container answers the
  plain REST list endpoint, `"$BASE?$QUERY&restype=container&comp=list"`, which costs no transfer and
  is how the breakdown below was taken before anything was downloaded.
- Fetch: `core/data/corpus/fetch_skmtea.sh` (resumable, skips what is already on disk).

Container, 471 blobs / 798.0 GB:

| prefix | files | size | verdict |
| --- | --- | --- | --- |
| `v1-release/files_recon_calib-24.tar.gz` | 1 | 763.4 GB | raw k-space — SKIP |
| `v1-release/dicoms` | 155 | 18.4 GB | the usable track |
| `v1-release/image_files.tar.gz` | 1 | 16.1 GB | h5 mirror of the same volumes |
| `v1-release/segmentation_masks` | 310 | 0.1 GB | fetched |
| `v1-release/annotations` | 3 | 0.0 GB | fetched first, to prove the join |
| `v1-release/all_metadata.csv` | 1 | 0.0 GB | |

**The join is proved and self-contained.** `annotations/v1.0.0/train.json` is COCO-like: 86 train
images, 242 train annotations, 16 categories over 6 tissues. An image row carries
`scan_id "MTR_001"`, which is exactly the blob name `dicoms/MTR_001.tar.gz` — same namespace, labels
shipped inside the release. Annotations are 3D boxes `[x, y, z, w, h, d]` with a `confidence` and a
`labeler`, and `tissue_id` gives compartment and laterality directly.

Categories: Meniscal Tear (8 subtypes), Ligament Tear (3 grades), Cartilage Lesion (grades 1/2A/2B/3),
Effusion.

**Priced honestly: this is not a training lever.** 155 studies against our 3,498, and qDESS is a
single 3D dual-echo sequence that fills roughly one of our six slots. It will not move 0.944. Its
value is that it is the only corpus here whose labels were produced by other people on other scanners,
which is the one thing our own holdout can never be.

## fastMRI (NYU) — the label namespace, and the steering error that cost a download

- Access: presigned download links, extracted from the NYU mail to
  `$CREDS/fastmri_urls.txt` (231 URLs across knee, brain, breast, prostate).
- Fetch: no script. The links file IS the fetch step — pick the archive by name and call it:
  `curl -sfL -C - "$(grep -F '/knee_singlecoil_train.tar.xz?' $CREDS/fastmri_urls.txt \
  | head -1)" -o /d/data/volumetric/mri/raw/fastmri/knee_singlecoil_train.tar.xz`. `-C -` resumes. A wrapper around
  three curl calls would be a file to maintain and a second place for the credential path to be
  written down.
- Archives worth pulling, in order: `knee_singlecoil_train.tar.xz` (~89 G, carries the annotated
  studies), `knee_singlecoil_val.tar.xz`, `labels.tar.gz`. NOT the multicoil batches — ~1 TB for a
  join singlecoil already provides. Check free space first; `D:` needs ~150 G of headroom for the
  train archive plus its extraction.

**fastMRI+ annotates by `fileNNNNNNN`, and that identifier exists only in the h5 releases.** The
120 GB DICOM release on disk (`knee_mri_clinical_seq_batch{1,2}`, layout
`<batch>/<patient>/study_<hash>/MR<n>_<hash>/NNNNN_<hash>.dcm`) is anonymised to opaque hashes —
`PatientID = 1FB`, `StudyInstanceUID = 9999.1102...` — with no `fileNNNNNNN` anywhere. It therefore
cannot join to a single fastMRI+ box.

That download happened because this repo's agent said the singlecoil/multicoil archives were "k-space
reconstruction" and could be skipped. Both halves of that were wrong: those h5 files carry
`reconstruction_esc` / `reconstruction_rss`, which are finished image volumes, AND they are the only
carrier of the label namespace. The advice steered toward the archive that cannot join and away from
the one that can. It is written down here so the next reader does not re-derive it from the same
guess.

The DICOM corpus is not worthless — it is a large in-domain UNLABELLED corpus with real
`SeriesDescription` (e.g. `pd_tse_fs_SAG_p2_320`, 1.5T, 384x384, 0.39 mm in-plane), which makes it the
only source of true sequence identity and a candidate SSL pretraining set. It is just not free boxes.

## Disk

`D:` had 472 G free at the start of these fetches. SKM-TEA takes 18.4 G. The fastMRI script refuses to
start an archive below 150 G free, because `knee_singlecoil_train.tar.xz` is ~89 G before extraction.
