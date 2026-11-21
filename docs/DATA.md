# Data, weights and checkpoints

This repository contains **no** radiographs, annotation files, patient identifiers, derived
masks or model weights. Anyone who wants to re-run verification, evaluation or the demo has to
obtain the data themselves under the original terms of use.

## 1. RSNA data

Source: RSNA Pneumonia Detection Challenge (2018), adjudicated-label release with expert
opacity boxes.

- Challenge page: <https://www.kaggle.com/c/rsna-pneumonia-detection-challenge>
- Rules / terms: <https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/rules>

Accepting the data terms is your responsibility. Do not redistribute the DICOM files or the JSON
metadata.

Required files:

| File | Expected SHA-256 |
|---|---|
| `pneumonia-challenge-dataset-mappings_2018.json` | `803ce79e3bc9c66d3631738e91e62e1175730e98ad1415e8dc4d6292ba10bf27` |
| `pneumonia-challenge-annotations-adjudicated-kaggle_2018.json` | `f8f5c54bae158fc0e2f019af979471803151a5922463ff3a378d7734e01b18fb` |
| 30,000 DICOM studies (`<StudyUID>/<SeriesUID>/<SOPUID>.dcm`) | checked by the P0 audit |

## 2. Expected layout

```
<repo root>/
├── data/
│   ├── raw/
│   │   ├── RSNA Pneumonia/raw/<StudyUID>/<SeriesUID>/<SOPUID>.dcm
│   │   ├── pneumonia-challenge-dataset-mappings_2018.json
│   │   └── pneumonia-challenge-annotations-adjudicated-kaggle_2018.json
│   ├── processed/          # created by P0 / P1 (see below)
│   └── lung_masks/train/   # created by P2 (PSPNet masks, *.npz)
├── models/pretrained/pspnet_chestxray_best_model_4.pth
└── results/runs/{b0_dev_s42,b1_boxpos_s42,a3_l100_s42}/best.pt
```

`code/paths.py` resolves the manifest's frozen `RSNA Pneumonia/raw/...` paths to
`data/raw/RSNA Pneumonia/raw/...`. Do not edit the frozen CSVs to change paths.

## 3. Frozen processed files

The P0 audit (`code/audit_data.py`) and the P1 split (`code/split_patients.py`) produce
`data/processed/`. The study used these exact files:

| File | SHA-256 |
|---|---|
| `master_manifest.csv` | `4c690ddf7918d0890b6715dba290be85a16959adc0dcaa70b5b4c5bd387d6c9b` |
| `split.csv` | `ccabcf432b0d266264e4ea96ca018b05a67d944af65fca4ad159f5130da7f9a7` |
| `boxes.csv` | `6372a1facc3e1b78fbcd97a5bd103d2fc33ed257c92f8076b1d36e9168078e45` |

The training, evaluation and demo code refuse to run when these hashes do not match.
An independent re-run of P0 + P1 on a second copy of the data reproduced the identical
`split.csv` hash.

## 4. Weights and checkpoints (not distributed here)

| File | SHA-256 |
|---|---|
| `models/pretrained/pspnet_chestxray_best_model_4.pth` (PSPNet lung segmentation, ~260 MB) | `019b167eac6b729fc1bb92bbbc185fc1730aaa65819f4e3fe718186cadc044fc` |
| `results/runs/b0_dev_s42/best.pt` | `f6bda8a74c5d72b948a9ee4ba35cf53c1c0176e7d15dbb457ecc78a4d79f2929` |
| `results/runs/b1_boxpos_s42/best.pt` | `f1b2ab19211d22dc4ebe53cb978d36e15fa436b58e16f389954f5d2da0e688f1` |
| `results/runs/a3_l100_s42/best.pt` | `7bd87ba3841f0b8c8287a859dc3c942027e6fb318b2d5122dbfe1321bca2743f` |

The ImageNet MobileNetV2 weights are loaded from the torchvision cache
(`MobileNet_V2_Weights.IMAGENET1K_V1`). They must be downloaded once before training.

## 5. What was deliberately left out

The following outputs exist in the original project but are not published because they contain
per-patient rows or machine-specific paths:

- per-image predictions (`predictions.csv`)
- per-lesion regional scores (`regions.csv`, `per_image.csv`)
- mask QA tables, sampler traces (`results/qa/`)
- run logs (`results/logs/`)

The aggregate metrics, bootstrap results and ROC/PR curves in `results/evaluation/` are complete
and are what the README tables are based on.
