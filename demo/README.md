# BG-REL research demo

Local Streamlit dashboard that runs the three frozen BG-REL checkpoints (B0, B1, A3) on one
uploaded chest-radiograph DICOM and compares their image-level Lung Opacity scores.

> **Research demonstration only.** Not a medical device and not for diagnosis or clinical
> decisions. Scores are `sigmoid(logit)` of research models trained on RSNA labels; they are
> **not** clinically calibrated probabilities.

## What it uses

| Model | Run | Training objective |
|---|---|---|
| B0 | `results/runs/b0_dev_s42/best.pt` | Image-level BCE |
| B1 | `results/runs/b1_boxpos_s42/best.pt` | BCE + positive-box BCE |
| A3 | `results/runs/a3_l100_s42/best.pt` | BCE + lung-aware listwise relative evidence (BG-REL) |

The checkpoints, source code and frozen data are read in place from the repository root
(the parent folder of `demo/` by default; override with the `BGREL_ROOT` environment variable). The demo never
writes to that project: it also disables `__pycache__` creation in the research `code/` folder.
No model is retrained.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (sections A–D) |
| `bgrel_infer.py` | Verified loading, DICOM decoding, preprocessing and inference |
| `tests/test_regression.py` | Regression tests on 3 TRAIN images |
| `.streamlit/config.toml` | Binds to `127.0.0.1:8501`, usage statistics off, light theme |
| `requirements-demo.txt` | The one added package (`streamlit==1.67.0`) |
| `constraints-bgrel.txt` | `pip freeze` of `.venvs\bgrel` before Streamlit was added |

## Image input

Section **A. Image input** offers two sources; both feed the identical
decode → preprocess → B0/B1/A3 pipeline.

**Select Sample from TRAIN Dataset** (default)

- `TrainCatalog` opens `data/processed/split.csv` and `master_manifest.csv` read-only, after
  verifying their frozen SHA-256 (`ccabcf43…`, `4c690ddf…`), and joins them on `image_id`.
- Only rows with `split == "train"` are kept; VAL/TEST rows are discarded while reading and no
  VAL/TEST image is ever listed or opened. Expected TRAIN counts are enforced:
  23,748 images = 5,685 **Positive** (`Lung Opacity`) + 18,063 **Negative**
  (`Normal`, `No Lung Opacity / Not Normal`).
- A category switch, a searchable dropdown (verified examples `00000032_007`, `00000003_003`,
  `00000003_000` first) and a *Random sample* button.
- The DICOM file is resolved from the manifest `dicom_path` (Study/Series/SOP-UID layout) via the
  research `paths.dicom_from_manifest`; the image ID is **not** the DICOM filename. The bytes go
  through the same `decode_dicom_bytes()` as uploads, and the research `decode_dicom()` is run on
  the manifest row as a cross-check (SOP UID / PatientID / photometric) — both decodes must be
  bit-identical.
- Displayed: image ID, ground-truth label, Positive/Negative category, view, the X-ray, the
  non-identifying DICOM attributes and the three predictions. Never displayed: `patient_id`,
  DICOM PatientID, Study/Series/SOP UIDs or the file path (the `TrainSample` object shown to the
  UI does not even carry them).
- These are training images, seen by the models: the scores illustrate the pipeline, not
  generalisation.

**Upload a DICOM file** — unchanged manual upload (1024×1024, MONOCHROME1/2).

**Upload PNG/JPG (external raster)** — a *separate* branch labelled
**"External raster – non-standard preprocessing"**, with an out-of-distribution warning. It uses
its own function, `decode_raster_bytes()`; the DICOM decoder and every other existing function
are unchanged (their source hashes were compared before and after). Steps, all shown in the UI:

1. Pillow decode; PNG and JPEG only (GIF, BMP, DICOM bytes etc. are rejected). A fully opaque
   alpha channel is dropped losslessly; any real transparency is rejected.
2. Greyscale: 8/16/32-bit and float greyscale kept as-is; colour via Pillow luminance
   `convert("L")` (0.299 R + 0.587 G + 0.114 B).
3. Aspect-preserving resize so the longer side is exactly 1024 (Pillow BILINEAR, antialiased when
   shrinking, no overshoot). Never stretched, never cropped. Skipped for 1024×1024 input.
4. Centred padding to 1024×1024 with the **minimum grey value of the resized image**, so the
   canvas adds no new intensity minimum.
5. The same per-image min–max normalisation as the DICOM path (no MONOCHROME1 inversion, no EXIF
   rotation), then the unchanged research `prepare_image()` via `preprocess()`.

Examples: 2048×1536 → 1024×768 + 128 px top/bottom; 1536×2048 → 768×1024 + 128 px left/right.
External rasters are **never** matched to TRAIN records — not even with a TRAIN-like filename and
identical pixels — so no ground truth, lesion boxes, reference regions, REM or Top-1 are shown.
Scores are labelled as research-model outputs, not validated probabilities or diagnoses.
For a 1024×1024 8-bit MONOCHROME2 image the raster path is bit-identical to the DICOM path (tested).

## Ground-truth lesion boxes

- Source: the frozen P0 `boxes.csv` (SHA-256 `6372a1fa…`), i.e. the adjudicated RSNA
  **"Calculated"** label group of the annotation JSON (verified identical in the tests). Other
  reader-level groups (Team 1/2/2a, Adjudication) are not used.
- Only TRAIN boxes are loaded (9,161). On load the catalog checks, per TRAIN image, that the box
  count equals the manifest `num_boxes`, that positives have ≥ 1 box and negatives none, that the
  canvas is the 1024×1024 DICOM frame and that `x2 = x + width`, `y2 = y + height`.
- Boxes are drawn in red on the **same decoded 1024×1024 array the models receive** — no resizing
  before drawing. Coordinates are continuous (x = column, y = row); the outline covers the pixels
  `floor(x) … ceil(x+width)−1` and is drawn inward. A test checks the outline corners sit exactly
  on those bounds and that no pixel outside the boxes is changed.
- Legend: *"Ground-truth RSNA expert annotation — not model-generated localization."*, the number
  of annotated regions, and a table of x, y, width, height and area. A toggle hides the boxes.
- Negatives: no box is drawn; the panel states that no ground-truth lesion box is available.
- Uploads: ground truth is shown **only** if the upload is confidently the same image as a TRAIN
  record — SOP Instance UID is in TRAIN, Study/Series UIDs and DICOM PatientID equal the manifest
  row, and the decoded pixels are bit-identical to the archived DICOM. Otherwise the app says there
  is no confident match and shows no label and no boxes. Non-TRAIN records are never looked up.

## D. Regional evidence (TRAIN only, frozen P3 reference regions)

**Data — nothing generated or resampled.** `RegionIndex` builds on the research loader
`rel_adapter.load_frozen_controls()` (SHA-pinned `results/qa/qa_controls.csv` `59ac71c8…`,
`sampler_trace.csv` `2932a246…`, split, manifest, boxes; full integrity checks) and re-reads the
two P3 CSVs under the same pins for per-lesion K, skip reason, tier, side and occupancy. Every
lesion box is cross-checked against `boxes.csv`, every control against the research loader.
A3/B1 record source hashes for `rel_adapter.py` and `regions.py` that match the current files.
2,667 TRAIN lesions in 2,421 images have K ≥ 1 reference regions; 6,494 have K = 0 (their
recorded skip reason is shown). PSPNet is not used; VAL/TEST are never touched (non-TRAIN IDs
raise before any file is opened). The saved regional CSVs are VAL-only and are not used.

**Scoring — research functions unchanged.** For each model: `forward_features` → research
`regions.scale_boxes_xyxy` (1024 → 224, continuous) → `regions.shared_region_scores`
(fractional area-weighted pooling on the 14×14 map, then the same linear head as the image
logit). REM = lesion logit − mean(reference logits); Top-1 = lesion logit > max(reference logit)
— the definitions in `evaluate_regions.py`. Two self-checks run on **every** request and withhold
the section on failure: a whole-image box must reproduce the image logit (|Δ| < 1e-4), and K ≥ 1
scores must equal the training path `rel_adapter.pair_scores_from_frozen` (|Δ| < 1e-5).

**Display.** Red solid = expert lesion box; blue dashed = frozen reference region (labels `2a`,
`2b` = lesion 2's references). A table lists lesion logit, reference logits, REM and Top-1 for
B0, B1 and A3. K = 0 lesions show only their lesion logit and why no reference exists. A toggle
in the Positive list restricts it to the 2,421 images with K ≥ 1; the reference image
`00000032_058` (one K = 2 lesion, one K = 0 lesion) is listed first. Negatives and unmatched
uploads show no regions.

**Interpretation caveats shown in the app.** TRAIN, in-sample: A3 was trained to rank these
exact lesions above these exact references, so a positive A3 REM is expected and is not evidence
of generalisation. Reference regions are geometry matches (`visual_status = UNKNOWN`, P3 status
"pending independent review"), not verified healthy tissue. Not Grad-CAM.

## F. Grad-CAM Explainability (`gradcam.py`)

Post-hoc visualisation of **image-level** prediction evidence for B0, B1 and A3 on the same
input, generated on request with the **Generate Grad-CAM** button.

- **Target layer:** `features.18` — the final `Conv2dNormActivation` (1×1 conv 320→1280, BN,
  ReLU6), verified to be `model.features[-1]`. Its 1280×14×14 output is the map whose global
  average feeds the single linear head (`forward = classifier(features(x).mean((2, 3)))`). Same
  layer for all three models.
- **Target:** the image-level Lung Opacity **logit** (before sigmoid), from a forward pass on the
  exact preprocessed tensor used for the predictions (no separate preprocessing).
- **Computation:** forward hook on the target layer (always removed in `finally`),
  `torch.autograd.grad(logit, A)` (no parameter `.grad` accumulated), channel weights = spatial
  mean of the gradients, weighted sum, ReLU. Works inside `no_grad`/`inference_mode`; eval mode
  preserved; CUDA with a CPU fallback on out-of-memory; finiteness checked.
- **Architecture note:** with GAP + one linear layer, ∂logit/∂A is `w_c/196` everywhere, so
  Grad-CAM equals the classic CAM `ReLU(Σ_c w_c A_c)/196`; the pre-ReLU map sums to
  `logit − bias`. Both identities are tested independently.
- **Display:** original 1024×1024 model view + three overlays (bilinear upsampling of the 14×14
  map, `align_corners=False`), an inferno-style colour map built in numpy (no new packages),
  per-pixel transparency proportional to the value, and a colour bar. Default **common scale**
  across the three models (comparable intensities); per-model scale is optional and labelled
  as not comparable. Zero maps (no positive evidence) are shown empty, never divided by ~0.
- **Box overlays (visual only, `gc_overlays.py`):** checkboxes *Show Ground Truth lesion boxes*
  (red) and *Show frozen reference regions* (blue dashed) draw outlines on the original and all
  three Grad-CAM panels. They reuse the section A/D drawers exactly: `draw_ground_truth` /
  `draw_regions` are rendered on a black sentinel canvas and the drawn pixels are pasted over
  each panel (identical coordinates, rounding, line widths and labels; heatmap untouched
  elsewhere). Available only for verified TRAIN samples or DICOMs matched to a TRAIN record;
  reference regions only when frozen K ≥ 1 data exists. Never shown for unmatched DICOMs or
  PNG/JPG. Boxes are never passed to Grad-CAM, which is computed before and independently of them.
- **Caching:** results are stored with a SHA-256 of the input tensor; option changes do not
  recompute, and a new image never shows the previous image's heatmaps.
- Works for TRAIN samples, DICOM uploads and PNG/JPG rasters (the external-raster and
  out-of-distribution warnings are repeated). It never adds ground truth, boxes, reference
  regions, REM or Top-1, and none of those are derived from heatmaps. Grad-CAM is not attention
  and does not show that a model has learned medically meaningful features.

## Integrity guarantees (checked at every start-up)

`bgrel_infer.ModelBundle()` refuses to load unless all of the following hold:

1. **Strict loading:** `load_state_dict(strict=True)` into the research `MobileNetV2OS16`.
2. **Provenance**, as in `evaluate_p8a.verify_run` / `evaluate_b1_val.verify_b1`:
   checkpoint SHA-256 and `metrics.csv` SHA-256 against `summary.json`; run status;
   `best_epoch == 3` matching the checkpoint; checkpoint config equal to `config.json`;
   saved VAL AUROC matching the summary; frozen split SHA; TEST-seal flags `False`;
   source hashes of `model.py`, `metrics.py` (and `regions.py` for B1).
3. **The one documented exception — `dataset.py`.** R5 changed only the DICOM path lookup
   (`from paths import dicom_from_manifest` + one line). The demo undoes exactly those two
   edits in memory and requires the result to hash to the training-time
   `7e1ecaa18d1c…` (`reproducibility/pre_r3_source/dataset.py`). Any other change fails.
4. **Preprocessing parity:** the upload decoder is checked bit-for-bit against the research
   `RSNAImageDataset(train=False)` on TRAIN image `00000003_000.png`.

`torchxrayvision` is never imported (it registers a top-level `model` module that shadows the
research `code/model.py`); the module asserts this.

## Preprocessing (same as the research pipeline)

1. `pydicom.dcmread`; require 1024×1024, `SamplesPerPixel == 1`, `MONOCHROME1`/`MONOCHROME2`,
   finite, non-constant pixels. Other sizes are **rejected, not resized**.
2. Per-image min–max normalisation to [0, 1]; invert for `MONOCHROME1`.
3. Research `prepare_image(train=False)`: bilinear resize to 224×224 (`align_corners=False`),
   grey → 3 channels, ImageNet mean/std. No augmentation.

The research `decode_dicom()` additionally cross-checks SOP UID / PatientID against the frozen
manifest row; an uploaded file has no manifest row, so that identity check cannot apply.

## Inference and TF32 policy

Batch size 1, `eval()`, `torch.inference_mode()`, FP32 weights and inputs, CUDA if available.
Models are loaded once (`st.cache_resource`) and reused. On CUDA out-of-memory a prediction
falls back to a CPU FP32 copy of that model and the UI says so.

**TF32:** `torch.backends.cudnn.allow_tf32 = True`, `torch.backends.cuda.matmul.allow_tf32 = False`
— the PyTorch defaults, which the archived research evaluators and the GPU smoke test also used.
On Ampere GPUs (RTX 30xx) convolutions may therefore run in TF32; CPU and GPU scores differ by
up to ~8e-4. The demo sets these values explicitly so they cannot drift.

## Privacy

The server binds to `127.0.0.1` only. Uploaded files stay in memory in the local Streamlit
process; they are not written to disk and not sent anywhere. Streamlit usage statistics are
disabled. Patient identifiers, dates and UIDs are not displayed.

## Setup

Run from the `demo/` folder of the repository:

```powershell
$py = '..\.venvs\bgrel\Scripts\python.exe'
& $py -m pip install -r requirements-demo.txt -c constraints-bgrel.txt
& $py -m pip uninstall -y pyarrow
```

**Why pyarrow is removed.** Streamlit depends on pyarrow, and the current pyarrow (26.0.1)
refuses to import with the research-pinned NumPy 1.26.4. Because some research import chains
load pyarrow when it is present, having it installed can break research unit tests. The demo therefore avoids every
pyarrow-backed Streamlit element: tables are markdown, the bar chart is inline SVG.
`pip check` reports `streamlit 1.67.0 requires pyarrow, which is not installed` — expected.
If pyarrow is ever reinstalled (e.g. by upgrading Streamlit), uninstall it again and re-run
`bash run_pipeline.sh tests` from the repository root.

## Launch

```powershell
..\.venvs\bgrel\Scripts\python.exe -m streamlit run app.py   # from demo/
```

Then open <http://localhost:8501>. Stop with `Ctrl+C`. Start-up takes a few seconds (provenance
hashing, model loading, GPU warm-up). Run the command from `demo/` so Streamlit picks up
`.streamlit/config.toml` (loopback binding, no telemetry).

Inference times are per forward pass. On a laptop GPU the first model of a request can take
~30 ms while the GPU clocks up from idle; the others typically take ~6–7 ms.

## Tests

```powershell
..\.venvs\bgrel\Scripts\python.exe -m unittest discover -s tests -v   # from demo/
```

> **Status: no tests have been executed on this repository copy.** The counts below come from
> reading the test files.
>
> Research suite (root `tests/`): the original research suite contained 61 tests; this
> repository includes 57 research tests. See the Tests section of the root [README](../README.md#tests).

14 of the 74 tests need no data or checkpoints: 9 raster-geometry tests in `test_raster.py`,
4 in `test_regression.py` (no torchxrayvision loaded, TF32 policy, dataset.py path-only diff,
invalid DICOM bytes) and the zero-heatmap test in `test_gradcam.py`. The other 60 need the RSNA
DICOMs, the frozen CSVs and the three `best.pt` checkpoints (see `../docs/DATA.md`); without them
they report errors rather than skipping.

74 tests: 7 in `test_gc_overlays.py` (section A/D drawers reproduced pixel-exactly, identical red/blue outlines on all four panels, heatmap untouched outside outlines, no reference layer for K = 0, Grad-CAM independent of boxes), 9 in `test_gradcam.py` (target layer feeds the logit, all three models, shapes, finiteness, [0, 1] normalisation, equality with independent CAM and logit sum, hooks removed incl. on exceptions, parameters and eval mode unchanged, no_grad/inference_mode/CPU, zero heatmaps, overlay transparency, raster input), 12 in `test_raster.py` (1024×1024, 2048×1536, 1536×2048 and other shapes; aspect
ratio; padding value and extent; no cropping/distortion via an independent Pillow resize and
corner markers; RGB luminance; 16-bit; JPEG; opaque-alpha handling; invalid inputs; never matched
to TRAIN; PNG of TRAIN DICOM pixels bit-identical to the DICOM path), 25 in `test_regression.py` (provenance, preprocessing parity, smoke-test values,
TRAIN-only catalog, join and counts, refusal of VAL/TEST IDs, no identifier fields, ground-truth
boxes vs the annotation JSON, pixel-level overlay alignment, no boxes on negatives, upload
matching incl. tampered identifiers; region geometry vs the raw P3 CSVs, whole-image and REM
self-consistency, independent recomputation with the research functions, K = 0 handling for the
verified examples, VAL/TEST refusal, region overlay alignment) and 11 headless UI tests in
`test_app.py` (default TRAIN sample, verified examples, ground-truth panel, matched vs unmatched
upload, regional evidence for the reference image / verified examples / unmatched uploads,
TRAIN-only dropdown contents, random button, upload flow, invalid upload) — plus 3 UI tests for
the raster branch (landscape upload, TRAIN-named raster gets no ground truth, invalid raster),
21 in total (incl. 3 box-overlay UI tests: K = 0 sample, K = 1 sample with both checkboxes on/off, no boxes for raster or unmatched DICOM; and 4 Grad-CAM UI tests: TRAIN sample with no stale results after changing the image, raster upload keeps OOD warnings and shows no ground truth, DICOM upload, no image). The UI tests also
scan everything the page renders for the sample's DICOM PatientID, Study/Series/SOP UIDs and path.

Inference runs only on the three verified TRAIN images (`00000032_007.png`, `00000003_003.png`,
`00000003_000.png`); the dropdown-content test reads only TRAIN metadata. They are training
examples, so the results show reproducibility of the pipeline, not model performance.

## Out of scope

No training, lung-mask generation, control sampling, regional evaluation or TEST evaluation.
The known R5 issues of the regional evaluation (torchxrayvision import collision,
`sample_controls.py` source hash) are not touched.

## Authors

Developed by **Đinh Đức Đạt** and **Trần Ngọc Minh** (FPT University, Ho Chi Minh City,
Vietnam), the two co-developers of BG-REL, who jointly built the whole project including this
demo and its Grad-CAM view. See the root [README](../README.md#authors).
