# fastMRI+ labels against our twelve targets

Measured on the MIT label repo (github.com/microsoft/fastmri-plus), cloned to
`D:/data/volumetric/mri/raw/fastmri/labels/Annotations/knee.csv` on 2026-09-01.

## What the corpus actually is

974 distinct studies, 16,167 annotation rows, `study_level = "No"` on 16,154 of them. Every row is a
per-slice bounding box: `file, slice, study_level, x, y, width, height, label`. This is LOCALIZED
supervision, which is a different KIND of signal from our study-level targets, not merely more studies.
The survey figure of 1,172 studies is wrong for the label file; 974 is what the annotations cover.

## The mapping

Ours is `TARGETS` in `kneemri/evaluation/protocol_prior.py:58` (twelve). fastMRI+ carries 22 classes.
More conditions, but not a superset.

| our target | fastMRI+ class(es) | boxes | status |
|---|---|---|---|
| ACL | Ligament - ACL Low Grade sprain; ACL High Grade Sprain | 1,443 | direct, and GRADED |
| MCL | Ligament - MCL Low-Mod Grade Sprain; MCL High Grade sprain | 296 | direct, and GRADED |
| Effusion | Joint Effusion | 1,311 | direct |
| Baker's | Periarticular cysts | 864 | superset; Baker's is one periarticular cyst among others |
| Medial Meniscus | Meniscus Tear (unsided) | 5,658 | laterality not in the label; recoverable from box x |
| Lateral Meniscus | Meniscus Tear (unsided) | (same) | as above |
| Medial OA | Cartilage partial/full loss; Bone- Subchondral edema | 4,586 | component of OA, compartment from box position |
| Lateral OA | as above | (same) | as above |
| PF OA | as above | (same) | as above |
| Contusion | Bone-Fracture/Contusion/dislocation | 1,060 | POOLED with Fracture; the distinction we score is not in the label |
| Fracture | Bone-Fracture/Contusion/dislocation | (same) | POOLED with Contusion |
| Synovitis | none | 0 | absent from the vocabulary |

Classes with no counterpart in our twelve, available as extra supervision: PCL (low-mod 142, high 18),
LCL complex (low-mod 130, high 14), Bone - Lesion 183, Soft Tissue Lesion 90, Muscle Strain 65,
Joint Bodies 38, Patellar Retinaculum 24, Displaced Meniscal Tissue 232, artifact 13.

## What follows

1. The meniscus block is the largest single chunk (5,658 boxes, 35% of the corpus) and is unsided in the
   label text only. Medial vs lateral is which side of the joint the box centre falls on, so the
   laterality is a geometric derivation, not missing data. The same derivation gives OA compartment
   from cartilage box position. Deriving it is a design decision that must be validated against a
   held-out subset before it is trusted, since it assumes a consistent left/right convention per series.
2. ACL and MCL arrive GRADED (low vs high). We currently train those as binary. Ordinal supervision on
   two targets is available for free and is not a knob.
3. Two of our targets cannot be helped by this corpus at all: Synovitis has no counterpart, and
   Contusion/Fracture are pooled into a single class, which is precisely the distinction we are scored
   on. Those remain our own data's job.
4. Box slice indices point into the ORIGINAL series. Our cache decimates roughly 35 slices to 12, so the
   annotated slices survive only if the retained-slice mapping is applied when ingesting. Ingesting
   without that mapping silently drops the localization, which is the entire reason to use this corpus.
