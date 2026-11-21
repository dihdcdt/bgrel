# BG-REL: Lung-Aware Box-Guided Relative Evidence Learning

**Chest radiograph lung-opacity classification with training-time relative regional supervision**

BG-REL is a research project on binary *Lung Opacity* classification on the RSNA Pneumonia
Detection Challenge chest radiographs. A classifier can reach a high image-level AUROC while
relying on shortcuts rather than on the abnormal region. BG-REL uses the expert lesion boxes
**only during training** and asks a different question from plain box supervision: *does the
model score the annotated lesion higher than comparable lung regions in the same image?*
At inference the model takes the radiograph alone, with no boxes or masks.

> **Research prototype, not a medical device.** The models predict the RSNA *Lung Opacity*
> label, not a clinical pneumonia diagnosis. Scores are `sigmoid(logit)` of research models and
> are not calibrated clinical probabilities.

---

## Method

All three models share one backbone and one linear head. Only the training signal differs.

| Model | Training objective | Extra supervision |
|---|---|---|
| **B0** – baseline | image-level BCE | image label only |
| **B1** – positive-box | BCE + BCE on each expert lesion box score | expert lesion boxes |
| **A3** – BG-REL | BCE + λ · listwise Relative Evidence Loss (REL) | boxes + frozen lung-aware reference regions |

- **Backbone:** ImageNet-initialised MobileNetV2 modified to output stride 16 (dilation 2),
  giving a `1280 × 14 × 14` feature map. Global average pooling feeds one shared
  `Linear(1280, 1)` head (`code/model.py`).
- **Fractional ROI pooling:** a box in 1024 × 1024 source coordinates is mapped without rounding
  onto the 14 × 14 grid, and features are averaged with area-overlap weights. The *same* head
  then gives a region logit on the same scale as the image logit (`code/regions.py`).
- **Lung-aware reference sampling:** for each lesion, up to K = 2 reference regions of the same
  size are chosen in the same lung or the contralateral lung. They must lie inside the predicted
  lung (PSPNet masks via TorchXRayVision, occupancy ≥ 0.80), avoid all annotated boxes and meet
  vertical-alignment and overlap rules. Selection never uses model scores. When no reference
  qualifies (K = 0) the lesion simply has no REL term (`code/sample_controls.py`).
  References are *geometric* candidates and are **not** confirmed healthy tissue.
- **Relative Evidence Loss:** a softmax over one lesion and its references,
  `ℓ_REL = log(e^{s(L)} + Σ_k e^{s(C_k)}) − s(L)`, averaged over eligible lesions and images
  (`code/losses.py`).

**Training recipe (all models):** AdamW (lr 3e-5, weight decay 1e-3), per-step cosine schedule,
batch size 4, BF16 autocast, brightness/contrast augmentation in [0.9, 1.1], at most 20 epochs
with early stopping (patience 4), seed 42. The checkpoint with the best validation AUROC was
kept; for all three models this was epoch 3.

## Data and protocol

- RSNA Pneumonia Detection Challenge (2018) with adjudicated labels and expert boxes.
  After audit: **29,684 radiographs from 12,233 patients**, 7,106 Lung Opacity positives and
  11,390 expert boxes.
- Patient-disjoint 80 / 10 / 10 split: TRAIN 23,748 images / 9,786 patients, VALIDATION 2,969 /
  1,223, TEST 2,967 / 1,224. No patient appears in more than one partition. The split is frozen
  and SHA-256 sealed.
- Preprocessing: DICOM decode (MONOCHROME1 inversion when needed), per-image min–max
  normalisation, bilinear resize to 224 × 224, 3-channel expansion and ImageNet normalisation.
- VALIDATION is used for checkpoint selection and for the regional analysis. TEST was evaluated
  **once**, at image level only, under a frozen protocol.

**No data, images, patient identifiers or model weights are included in this repository.**
See [docs/DATA.md](docs/DATA.md) for how to obtain the data and where to place it.

## Results

All numbers below are read from the archived files in `results/evaluation/`.

### Image level (held-out TEST, n = 2,967 images, 1,224 patients)

| Model | AUROC [95% CI] | AP [95% CI] | Sensitivity @0.5 | Specificity @0.5 |
|---|---|---|---|---|
| B0 | 0.8788 [0.8605, 0.8962] | 0.7180 [0.6622, 0.7635] | 0.534 | 0.932 |
| B1 | 0.8771 [0.8590, 0.8943] | 0.7098 [0.6531, 0.7550] | 0.569 | 0.911 |
| A3 (BG-REL) | 0.8785 [0.8607, 0.8951] | 0.7126 [0.6572, 0.7572] | 0.577 | 0.911 |

Paired AUROC differences: A3 − B0 = −0.0003 [−0.0038, +0.0034], A3 − B1 = +0.0014
[−0.0028, +0.0057]. **Every interval includes 0: BG-REL shows no image-level gain.**
The 0.5 threshold is descriptive and was not tuned. CIs come from 2,000 paired patient-cluster
bootstrap resamples (seed 42). Source: `results/evaluation/final_test/`.

### Regional level (VALIDATION only, 344 eligible lesions / 168 patients)

Only 344 of 1,122 validation lesion boxes had at least one admissible reference. All models
were scored on exactly the same lesions.

| Model | Mean REM (lesion − mean reference logit) | Top-1 (lesion > all references) |
|---|---|---|
| B0 | 1.653 | 61.9% |
| B1 | 1.797 | 67.4% |
| A3 (BG-REL) | **3.075** | **76.5%** |

Patient-averaged A3 − B1 difference: **REM +1.148 [+0.764, +1.514]**, which excludes 0.
Top-1 +0.070 [−0.006, +0.145] includes 0. Source: `results/evaluation/regions_val/` and
`results/evaluation/b1_regions_val/`.

### What this shows, and what it does not

- BG-REL separates annotated lesions from matched references more strongly than direct box
  supervision, on the eligible validation subset.
- It does **not** improve image-level discrimination.
- **Limitations:**
  - References are not confirmed healthy tissue, and the PSPNet masks were not validated
    against expert lung masks.
  - Regional results cover eligible lesions only.
  - Each model was trained with a single seed.
  - There is no regional TEST evaluation and no external cohort.

## Repository layout

```
code/              29 research modules (pipeline phases P0–P9, model, losses, regions, evaluation)
tests/             57 unit / regression tests for the research code (no data needed; see Tests)
reproducibility/   archived historical sources and SHA ledgers (used by the tests)
scripts/           read-only provenance checkers
demo/              local Streamlit demo (B0/B1/A3 scores, regional evidence, Grad-CAM) + tests
results/runs/      per-model training config, summary, per-epoch metrics, training log
results/evaluation/ archived metric summaries, bootstrap comparisons, ROC / PR curves
docs/              data instructions (DATA.md) + historical development records (environment, technical review, test report)
run_pipeline.sh    runbook: status / plan / verify / tests / per-phase preflight checks
```

Main modules:

| Module | Purpose |
|---|---|
| `code/audit_data.py`, `code/split_patients.py` | P0 data audit, P1 patient-level split |
| `code/dataset.py` | split/manifest seal checks, DICOM decoding, preprocessing, augmentation |
| `code/model.py` | MobileNetV2-OS16 with shared head |
| `code/regions.py`, `code/losses.py`, `code/training_objectives.py` | fractional ROI pooling, REL loss, B1 box loss |
| `code/generate_lung_masks.py`, `code/mask_qa.py`, `code/mask_policy.py`, `code/build_train_lung_masks.py` | PSPNet lung masks and QA |
| `code/sample_controls.py`, `code/rel_adapter.py` | lung-aware reference sampler, frozen-reference loader |
| `code/train_baseline.py`, `code/train_ablations.py`, `code/train_bgrel.py` | training of B0, B1, A3 |
| `code/evaluate_p8a.py`, `code/evaluate_regions.py`, `code/evaluate_b1_*.py`, `code/evaluate_final_test.py` | validation, regional and final test evaluation |

## Environment

Developed and run on Windows 11, Python 3.12, NVIDIA RTX 4050 Laptop GPU.

```powershell
py -3.12 -m venv .venvs\bgrel
.\.venvs\bgrel\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.\.venvs\bgrel\Scripts\python.exe -m pip install -r requirements.txt
```

Use `--index-url https://download.pytorch.org/whl/cpu` for a CPU-only install.
`requirements.txt` pins numpy 1.26.4, pydicom 3.0.1, Pillow 11.3.0, pandas 2.3.3, scipy 1.17.1,
scikit-learn 1.7.2, scikit-image 0.26.0, torchxrayvision 1.5.7 and tqdm 4.70.1. It lists the
top-level packages observed in the original environment and is not a full lock file.
`run_pipeline.sh` expects the venv at exactly `.venvs/bgrel`.

## Usage

Run from Git Bash in the repository root:

```bash
bash run_pipeline.sh status   # layout, venv and asset presence report (data/model items show as missing until you add them)
bash run_pipeline.sh tests    # python -m unittest discover -s tests -v
bash run_pipeline.sh help     # all commands, including per-phase --preflight checks
```

Full verification (`verify`, `check`) and training or evaluation need the RSNA data, the PSPNet
weights and the checkpoints placed as described in [docs/DATA.md](docs/DATA.md). Training
scripts include guards against overwriting the archived runs.

### Demo

The Streamlit demo runs the three checkpoints on one DICOM image, either a TRAIN sample or an
upload. It shows the image-level scores, the expert boxes, the frozen reference regions with
their region logits / REM / Top-1, and Grad-CAM maps. It needs the local data and checkpoints.

```powershell
.\.venvs\bgrel\Scripts\python.exe -m pip install -r demo\requirements-demo.txt -c demo\constraints-bgrel.txt
.\.venvs\bgrel\Scripts\python.exe -m pip uninstall -y pyarrow   # see demo/requirements-demo.txt
cd demo
..\.venvs\bgrel\Scripts\python.exe -m streamlit run app.py      # binds to 127.0.0.1:8501
```

By default the demo uses the parent folder of `demo/` as the repository root. Set `BGREL_ROOT`
only if the code lives somewhere else; the demo stops with a clear error if `code/model.py` is
missing.
TRAIN-sample scores show that the pipeline works. They are not evidence of generalisation,
because the models were trained on those images.

## Tests

| Suite | Location | Tests in this repository | Need external data / checkpoints? |
|---|---|---|---|
| Research | `tests/` | **57** | No. They use synthetic tensors and masks, temporary folders and `reproducibility/` |
| Demo | `demo/tests/` | **74** | **14 do not**; the other 60 need the RSNA DICOMs, the frozen CSVs and the three `best.pt` checkpoints |

> **Test status: no tests have been executed on this repository copy.** The counts above come
> from reading the test files. The original research suite contained 61 tests; this repository
> includes 57 research tests. Test results recorded in `docs/` (for example 42 tests in
> `docs/TECHNICAL_REVIEW.md`) are historical development records for earlier versions of the
> suite, not results for this repository.

Notes:
- The research tests still need every package from `requirements.txt` plus torch and torchvision.
- Run the research tests through `discover` (`bash run_pipeline.sh tests`):
  `test_rel_adapter_restored.py` relies on `code/` being put on the import path by the test
  modules loaded before it.
- A source-migration contract test that depended on internal deployment tooling is not included
  here, which accounts for the difference between the original 61 and the 57 included here.
- Without data, the 60 data-dependent demo tests report errors; they are not skipped
  automatically. Details are in [demo/README.md](demo/README.md#tests).

## Authors

- **Đinh Đức Đạt** – FPT University, Ho Chi Minh City, Vietnam
- **Trần Ngọc Minh** – FPT University, Ho Chi Minh City, Vietnam

Both authors are co-developers. They jointly carried out all parts of the project: data
preprocessing and splitting, model and loss implementation, reference sampling, training and
evaluation, the demo and Grad-CAM, the technical report and the presentation.

## Acknowledgements

- RSNA Pneumonia Detection Challenge data and expert annotations (Shih et al., *Radiology: AI*, 2019).
- MobileNetV2 (Sandler et al., CVPR 2018) via torchvision.
- PSPNet lung segmentation via [TorchXRayVision](https://github.com/mlmed/torchxrayvision) (Cohen et al., MIDL 2022).
- Grad-CAM (Selvaraju et al., ICCV 2017).

## License

No license has been chosen yet. All rights reserved by the authors until a license is added.
