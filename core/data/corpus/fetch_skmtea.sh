#!/usr/bin/env bash
# One-off fetch of the SKM-TEA DICOM track + segmentation masks. The SAS is a credential and is read
# from outside the repo, never embedded here. Promoted to core/data/corpus/fetch_skmtea.py once the
# corpus earns an ingest; this exists so the 18.4 GB moves while that is written.
#
# TAKES A TRACK, AND THE MASKS COME FIRST, BECAUSE THE FIRST RUN BURIED THEM. Listed order is
# dicoms then segmentation_masks, so a fetch killed for disk had pulled 19 GB of images and ZERO
# masks -- and the masks are the only reason this corpus is here. They are 310 files totalling
# ~84 MB, four orders of magnitude under the DICOMs, so the disk objection that parked this fetch
# was never an objection to them.
#
#   bash core/data/corpus/fetch_skmtea.sh segmentation_masks   # 84 MB, the supervision signal
#   bash core/data/corpus/fetch_skmtea.sh                      # both tracks, ~57 GB
#
# THE PER-BLOB mkdir IS NOT DEFENSIVE, IT IS THE BUG THIS SCRIPT ALREADY HAD. Mask blobs are nested
# one level deeper than the two directories created up front (segmentation_masks/dicom-track/...),
# curl -o will not create a parent, and -sf silences the error -- so the first masks run reported
# 310 of 310 blobs attempted, printed a progress line for every one, exited 0, and wrote nothing at
# all. The only trace was the FAILED line, which reads like a handful of bad blobs rather than a
# total no-op. Count the files on disk, never the progress lines.
set -u
TRACK="${1:-dicoms|segmentation_masks}"
SAS=$(cat "${SKMTEA_URL_FILE:?set SKMTEA_URL_FILE to the file holding the SAS URL}")
BASE="${SAS%%\?*}"; QUERY="${SAS#*\?}"
source experiments/roots.sh
OUT="$SKMTEA_RAW"
mkdir -p "$OUT/dicoms" "$OUT/segmentation_masks"
python_list() { grep -oE "<Name>v1-release/($TRACK)/[^<]*</Name>" logs/skmtea_list.xml | sed 's/<[^>]*>//g'; }
total=$(python_list | wc -l); done_count=0
for blob in $(python_list); do
  target="$OUT/${blob#v1-release/}"
  done_count=$((done_count+1))
  if [ -s "$target" ]; then continue; fi
  mkdir -p "$(dirname "$target")"
  curl -sf "$BASE/$blob?$QUERY" -o "$target.part" && mv "$target.part" "$target" \
    || { echo "FAILED $blob"; rm -f "$target.part"; }
  echo "[$done_count/$total] $blob"
done
echo "SKMTEA FETCH COMPLETE"
du -sh "$OUT"
