#!/usr/bin/env bash
# One-off fetch of the SKM-TEA DICOM track + segmentation masks. The SAS is a credential and is read
# from outside the repo, never embedded here. Promoted to core/data/corpus/fetch_skmtea.py once the
# corpus earns an ingest; this exists so the 18.4 GB moves while that is written.
set -u
SAS=$(cat "D:/life/creds/machine/skmtea_url.txt")
BASE="${SAS%%\?*}"; QUERY="${SAS#*\?}"
OUT=/d/data/volumetric/mri/raw/skmtea
mkdir -p "$OUT/dicoms" "$OUT/segmentation_masks"
python_list() { grep -oE "<Name>v1-release/(dicoms|segmentation_masks)/[^<]*</Name>" logs/skmtea_list.xml | sed 's/<[^>]*>//g'; }
total=$(python_list | wc -l); done_count=0
for blob in $(python_list); do
  target="$OUT/${blob#v1-release/}"
  done_count=$((done_count+1))
  if [ -s "$target" ]; then continue; fi
  curl -sf "$BASE/$blob?$QUERY" -o "$target.part" && mv "$target.part" "$target" \
    || { echo "FAILED $blob"; rm -f "$target.part"; }
  echo "[$done_count/$total] $blob"
done
echo "SKMTEA FETCH COMPLETE"
du -sh "$OUT"
