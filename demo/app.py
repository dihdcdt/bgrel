"""BG-REL research demo (local Streamlit app).

Launch (from demo/):  python -m streamlit run app.py   (set BGREL_ROOT if the repo root is elsewhere)
Binds to 127.0.0.1:8501 (see .streamlit/config.toml). Uploads stay in memory on this
machine; nothing is written to disk and nothing is sent to external services.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import html
import time

import numpy as np
import streamlit as st
import torch

import bgrel_infer as bi
import gradcam as gc
import gc_overlays as gco

# NOTE: no st.dataframe / st.altair_chart / st.vega_lite_chart. Those need pyarrow, and the
# pyarrow wheel that Streamlit pulls in requires NumPy >= 2, while the research environment is
# pinned to NumPy 1.26.4 - installing it broke 13 research unit tests. Tables are rendered as
# markdown and the chart as inline SVG instead, so pyarrow is not needed at all.

TITLE = "BG-REL: Lung-Aware Box-Guided Relative Evidence Learning"
SERIES = "#2a78d6"          # single-series bar colour (reference palette slot 1, light)
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SCORE_LABEL = "Research-model Lung Opacity score"

st.set_page_config(page_title="BG-REL research demo", page_icon="🫁", layout="wide")


def md_table(headers, rows) -> str:
    esc = lambda v: str(v).replace("|", "\\|")
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(esc(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def score_bar_svg(preds) -> str:
    """Horizontal bars on a fixed 0-1 axis; value at the bar tip; native hover tooltip."""
    W, left, right, top, row_h, bar_h = 560, 44, 56, 30, 42, 22
    plot_w = W - left - right
    H = top + row_h * len(preds) + 46
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
           f'aria-label="{SCORE_LABEL} per model" '
           f'style="font-family:\'Source Sans Pro\',\'Source Sans 3\',sans-serif;max-width:{W}px">',
           f'<text x="{left}" y="16" font-size="14" fill="{INK}">{SCORE_LABEL} for the uploaded image</text>']
    for i in range(6):  # recessive grid + axis labels at 0.0 ... 1.0
        x = left + plot_w * i / 5
        out.append(f'<line x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" y2="{H - 42}" stroke="{GRID}"/>')
        out.append(f'<text x="{x:.1f}" y="{H - 26}" font-size="12" fill="{INK_2}" '
                   f'text-anchor="middle">{i / 5:.1f}</text>')
    for k, p in enumerate(preds):
        y = top + k * row_h + (row_h - bar_h) / 2
        w = max(plot_w * min(max(p.score, 0.0), 1.0), 0.0)
        r = min(4.0, w / 2)
        # square at the baseline, 4px rounded data-end
        d = (f"M{left},{y} h{w - r:.2f} a{r},{r} 0 0 1 {r},{r} v{bar_h - 2 * r} "
             f"a{r},{r} 0 0 1 {-r},{r} h{-(w - r):.2f} z")
        tip = html.escape(f"{p.label} ({bi.MODEL_INFO[p.run_id][1]}): {p.score:.4f}")
        out.append(f'<g><title>{tip}</title>'
                   f'<rect x="{left}" y="{y - 10}" width="{plot_w}" height="{bar_h + 20}" fill="transparent"/>'
                   f'<path d="{d}" fill="{SERIES}"/></g>')
        out.append(f'<text x="{left - 10}" y="{y + bar_h / 2 + 5}" font-size="13" fill="{INK}" '
                   f'text-anchor="end">{p.label}</text>')
        out.append(f'<text x="{left + w + 6:.1f}" y="{y + bar_h / 2 + 5}" font-size="13" '
                   f'fill="{INK}">{p.score:.3f}</text>')
    out.append(f'<text x="{left + plot_w / 2}" y="{H - 4}" font-size="12" fill="{INK_2}" '
               f'text-anchor="middle">{SCORE_LABEL} (0–1)</text></svg>')
    return "".join(out)


# ---------------------------------------------------------------------------
# Load once, reuse across reruns and sessions
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Verifying provenance and loading B0 / B1 / A3 checkpoints…")
def get_bundle():
    t = time.perf_counter()
    b = bi.ModelBundle()
    parity = bi.preprocessing_parity_check()
    return b, parity, (time.perf_counter() - t)


st.title(TITLE)
st.caption("Local research demonstration of three frozen RSNA Lung Opacity classifiers "
           "(MobileNetV2-OS16, seed 42). **Not a medical device. Not for diagnosis or clinical "
           "decision-making.** Outputs are research-model scores, not calibrated probabilities.")

try:
    bundle, parity, load_s = get_bundle()
except bi.IntegrityError as e:
    st.error(f"**Provenance check failed - models were not loaded.** {e}")
    st.stop()
except Exception as e:  # missing files, CUDA init failure, etc.
    st.error(f"**Could not load the models.** {type(e).__name__}: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar: system and provenance status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("System")
    st.markdown(f"**Device:** {bundle.device_name} (`{bundle.device.type}`)")
    st.markdown(f"**PyTorch:** {torch.__version__}")
    st.markdown("**Precision:** FP32 weights and inputs, batch size 1, `eval()`, "
                "`torch.inference_mode()`, no augmentation")
    st.markdown("**TF32 policy:** cuDNN convolutions TF32 **on**, matmul TF32 **off** "
                "(PyTorch defaults; identical to the archived evaluators and the GPU smoke test).")
    st.markdown(f"**Startup:** {load_s:.1f} s")
    st.subheader("Provenance checks")
    st.markdown("✅ Strict `load_state_dict` for all three checkpoints  \n"
                "✅ Checkpoint and metrics SHA-256, epoch, config, VAL-AUROC record  \n"
                "✅ `model.py` / `metrics.py` / `regions.py` source hashes  \n"
                "✅ `dataset.py` differs from training version only in DICOM path lookup  \n"
                f"✅ Upload preprocessing bit-identical to research `RSNAImageDataset` "
                f"(TRAIN reference `{parity['image_id']}`)")
    with st.expander("Checkpoint SHA-256"):
        for run_id, rec in bundle.records.items():
            st.code(f"{bi.MODEL_INFO[run_id][0]}  {run_id}\n{rec['checkpoint_sha256']}", language=None)
    st.caption("All processing happens locally. Uploaded files are held in memory only and are "
               "never written to disk or sent to external services. TRAIN samples are read "
               "read-only from the research project; VAL/TEST images are never listed or opened.")

# ---------------------------------------------------------------------------
# A. Image input: TRAIN sample (default) or manual upload
# ---------------------------------------------------------------------------
SRC_TRAIN = "Select Sample from TRAIN Dataset"
SRC_UPLOAD = "Upload a DICOM file"
SRC_RASTER = "Upload PNG/JPG (external raster)"
OOD_WARNING = (f"**{bi.RASTER_LABEL}.** This image did not come from the research DICOM pipeline. It "
               "was converted to greyscale, resized with its aspect ratio preserved and padded to "
               "1024×1024 - steps the models never saw in training. PNG/JPG exports may also be "
               "windowed, contrast-adjusted, lossy-compressed or cropped differently from RSNA "
               "DICOMs. The image is **out-of-distribution** for these models: the scores below are "
               "research-model outputs only - **not validated probabilities and not a diagnostic "
               "result**. No ground truth, lesion boxes or regional evidence exist for external "
               "images, so none are shown.")


@st.cache_resource(show_spinner="Reading the frozen TRAIN split (read-only)…")
def get_catalog():
    return bi.train_catalog()


@st.cache_resource(show_spinner="Reading frozen P3 TRAIN reference regions (read-only)…")
def get_regions():
    return bi.region_index()


def pick_random(category_key: str, options_key: str):
    import random
    opts = st.session_state[options_key]
    st.session_state[category_key] = random.choice(opts)


st.header("A. Image input")
source = st.radio("Image source", [SRC_TRAIN, SRC_UPLOAD, SRC_RASTER], horizontal=True, key="source")
decoded, sample, x, prep_ms = None, None, None, 0.0

if source == SRC_TRAIN:
    st.subheader(SRC_TRAIN)
    try:
        catalog = get_catalog()
    except Exception as e:
        st.error(f"**The TRAIN sample list could not be loaded.** {type(e).__name__}: {e}")
        catalog = None
    if catalog is not None:
        st.caption("Images from the frozen **TRAIN** partition only (patient-disjoint split, "
                   "`split.csv` SHA-256 verified). Validation and test images are never listed or "
                   "opened. These are training images: the models have seen them, so the scores "
                   "illustrate the pipeline, not generalisation.")
        cat_col, pick_col = st.columns([1, 3], gap="large")
        with cat_col:
            category = st.radio(
                "Category", ["Positive", "Negative"], key="category",
                captions=[f"Lung Opacity · {len(catalog.by_category['Positive']):,} images",
                          f"Normal or No Lung Opacity / Not Normal · "
                          f"{len(catalog.by_category['Negative']):,} images"])
        options = catalog.options(category)
        only_k = False
        if category == "Positive":
            only_k = st.toggle("Only images with frozen reference regions (K ≥ 1, for section D)",
                               key="only_k")
            if only_k:
                with_ctrl = get_regions().images_with_controls
                ref = [s for s in options if s.image_id == bi.REGION_REFERENCE_ID]
                options = ref + [s for s in options
                                 if s.image_id in with_ctrl and s.image_id != bi.REGION_REFERENCE_ID]
        sel_key = f"sample_{category}" + ("_k" if only_k else "")
        opt_key = f"options_{category}" + ("_k" if only_k else "")
        st.session_state[opt_key] = options
        with pick_col:
            sample = st.selectbox(
                f"{category} TRAIN image", options, key=sel_key,
                format_func=lambda s: (f"{s.display_id} — {s.final_label} · {s.view_position}"
                                       + ("  ✓ verified example" if s.image_id in bi.VERIFIED_EXAMPLES
                                          else "")),
                help="Type to search by image ID. Verified examples from the smoke test are listed first.")
            st.button("Random sample from this category", on_click=pick_random,
                      args=(sel_key, opt_key))
        try:
            with st.spinner("Loading DICOM from the manifest…"):
                t0 = time.perf_counter()
                decoded = catalog.load(sample.image_id)
                x = bi.preprocess(decoded.pixel01)
                prep_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            st.error(f"**This sample could not be loaded.** {type(e).__name__}: {e}")
            decoded = None
elif source == SRC_UPLOAD:
    st.subheader("Upload DICOM file")
    uploaded = st.file_uploader("Upload one chest radiograph DICOM file (.dcm)", type=["dcm"],
                                accept_multiple_files=False)
    if uploaded is None:
        st.info("Upload a 1024×1024 single-channel DICOM (RSNA Pneumonia format) to run the three models.")
    else:
        try:
            with st.spinner("Decoding DICOM…"):
                t0 = time.perf_counter()
                decoded = bi.decode_dicom_bytes(uploaded.getvalue())
                x = bi.preprocess(decoded.pixel01)
                prep_ms = (time.perf_counter() - t0) * 1000
        except bi.DicomInputError as e:
            st.error(f"**This file cannot be used.** {e}")
            decoded = None
        except Exception as e:
            st.error(f"**Unexpected error while reading the file.** {type(e).__name__}: {e}")
            decoded = None
else:
    # SEPARATE branch: external PNG/JPG -> bi.decode_raster_bytes (letterbox to 1024x1024) ->
    # the unchanged bi.preprocess(). Never matched to TRAIN records, so `sample` stays None.
    st.subheader("Upload PNG/JPG (external raster)")
    st.warning(OOD_WARNING)
    raster_up = st.file_uploader("Upload one chest radiograph image (.png, .jpg, .jpeg)",
                                 type=["png", "jpg", "jpeg"], accept_multiple_files=False,
                                 key="raster_upload")
    if raster_up is None:
        st.info("Any size is accepted: the image is fitted into 1024×1024 without stretching or "
                "cropping, and the remaining area is padded.")
    else:
        try:
            with st.spinner("Converting image…"):
                t0 = time.perf_counter()
                decoded = bi.decode_raster_bytes(raster_up.getvalue())
                x = bi.preprocess(decoded.pixel01)
                prep_ms = (time.perf_counter() - t0) * 1000
        except bi.RasterInputError as e:
            st.error(f"**This file cannot be used.** {e}")
            decoded = None
        except Exception as e:
            st.error(f"**Unexpected error while reading the image.** {type(e).__name__}: {e}")
            decoded = None

is_raster = isinstance(decoded, bi.RasterImage)
upload_match_reason = None
if source == SRC_UPLOAD and decoded is not None:
    # Ground truth for an upload only if it is confidently the same image as a TRAIN record.
    try:
        sample, upload_match_reason = get_catalog().match_upload(decoded)
    except Exception as e:
        sample, upload_match_reason = None, f"TRAIN records unavailable ({type(e).__name__})"
    if sample is not None:
        st.success("This upload is identical to a verified TRAIN record (SOP/Study/Series UIDs, "
                   "DICOM PatientID and pixel data all match), so its ground truth is shown.")
    else:
        st.info(f"No confident match to a verified TRAIN record ({upload_match_reason}). "
                "Ground-truth label and lesion boxes are therefore **not shown** for this upload; "
                "none are inferred.")

if decoded is not None and sample is not None:
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Image ID**  \n`{sample.display_id}`")
        c2.markdown(f"**Ground-truth label**  \n{sample.final_label}")
        c3.markdown(f"**Category**  \n{sample.category}")

if decoded is not None:
    gt_boxes = get_catalog().boxes(sample.image_id) if sample is not None else ()
    left, right = st.columns([1, 1], gap="large")
    with left:
        show_gt = bool(gt_boxes) and st.toggle("Show ground-truth lesion boxes", value=True, key="show_gt")
        if show_gt:
            st.image(bi.draw_ground_truth(decoded.pixel01, gt_boxes), width="stretch",
                     caption="Decoded 1024×1024 radiograph with lesion boxes drawn in original DICOM "
                             "pixel coordinates (no resizing before drawing).")
        elif is_raster:
            o = decoded.ops
            st.image((decoded.pixel01 * 255).round().astype(np.uint8), clamp=True, width="stretch",
                     caption=f"{bi.RASTER_LABEL}: the 1024×1024 canvas the models receive "
                             f"(original {o['original_size'][0]}×{o['original_size'][1]} → "
                             f"{o['resized'][0]}×{o['resized'][1]}, padded with the image's minimum "
                             f"grey value, then min-max normalised).")
        else:
            st.image((decoded.pixel01 * 255).round().astype(np.uint8), clamp=True,
                     caption="Decoded radiograph after per-image min-max normalisation "
                             "(MONOCHROME1 inverted) - the view the models receive before resizing.",
                     width="stretch")
        if sample is not None:
            with st.container(border=True):
                if gt_boxes:
                    st.markdown(f'<span style="display:inline-block;width:14px;height:14px;'
                                f'border:3px solid rgb{bi.GT_RED};vertical-align:-2px;margin-right:6px">'
                                f'</span> **{bi.GT_LEGEND}**', unsafe_allow_html=True)
                    st.markdown(f"**Annotated lesion regions: {len(gt_boxes)}**")
                    st.markdown(md_table(
                        ["Box", "x", "y", "Width", "Height", "Area"],
                        [(b.number, f"{b.x:.1f}", f"{b.y:.1f}", f"{b.width:.1f}", f"{b.height:.1f}",
                          f"{100 * b.width * b.height / 1024**2:.1f} %")
                         for b in gt_boxes]))
                    st.caption("Pixel coordinates in the original 1024×1024 DICOM frame "
                               "(x = column, y = row, origin top-left; the box spans x … x+width, "
                               "y … y+height; area as % of the image). Source: adjudicated "
                               "RSNA 'Calculated' annotations via the frozen P0 boxes.csv. "
                               "The models never see these boxes at inference.")
                else:
                    st.markdown("**No ground-truth lesion box is available for this image.**")
                    st.caption(f"Ground-truth label: {sample.final_label}. Negative RSNA images carry "
                               "no lesion annotation, so no box is drawn.")
    with right:
        st.subheader("Image metadata")
        st.markdown(md_table(["Attribute", "Value"], decoded.metadata.items()))
        st.caption("Patient identifiers, dates and UIDs are intentionally not displayed.")
        if is_raster:
            o, p = decoded.ops, decoded.ops["pad"]
            st.subheader("Resize and padding")
            st.markdown(md_table(["Step", "Value"], [
                ("Original size", f"{o['original_size'][0]} × {o['original_size'][1]} px"),
                ("Scale factor (aspect preserved)", f"{o['scale']:.4f}"),
                ("Resized to", f"{o['resized'][0]} × {o['resized'][1]} px"),
                ("Interpolation", o["interpolation"]),
                ("Padding left / right / top / bottom",
                 f"{p['left']} / {p['right']} / {p['top']} / {p['bottom']} px"),
                ("Padding value", f"{o['pad_value']:g} (minimum grey value of the resized image)"),
                ("Final canvas", "1024 × 1024 px, then per-image min-max to [0, 1]"),
                ("Cropping / stretching", "none"),
                ("EXIF rotation / photometric inversion", "not applied")]))
        st.markdown(f"Decode + preprocessing: **{prep_ms:.1f} ms** → model input "
                    f"`{tuple(x.shape)}` float32")

    # -----------------------------------------------------------------------
    # B. Model predictions
    # -----------------------------------------------------------------------
    st.header("B. Model predictions")
    if is_raster:
        st.warning(f"**{bi.RASTER_LABEL} - out-of-distribution input.** These are raw research-model "
                   "scores for a converted external image. They are not validated probabilities, "
                   "not calibrated, and not a diagnostic result.")
    if sample is not None:
        st.markdown(f"{'Matched TRAIN record' if source == SRC_UPLOAD else 'TRAIN sample'} "
                    f"`{sample.display_id}` · ground truth (for reference): "
                    f"**{sample.final_label}** ({sample.category})")
    try:
        with st.spinner("Running B0, B1 and A3…"):
            preds = bundle.predict(x)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        st.error("GPU ran out of memory and the CPU fallback could not run. Close other GPU "
                 "applications and try again.")
        st.stop()
    except Exception as e:
        st.error(f"**Inference failed.** {type(e).__name__}: {e}")
        st.stop()

    for p in preds:
        if p.note:
            st.warning(f"{p.label}: {p.note}")

    st.markdown(md_table(
        ["Model", "Run", "Lung Opacity score", "Logit", "Time (ms)", "Device"],
        [(f"**{p.label}**", f"`{p.run_id}`", f"**{p.score:.4f}**",
          f"{p.logit:+.4f}", f"{p.ms:.2f}", p.device) for p in preds]))
    st.caption(f"Score = sigmoid(logit) of a single research model trained on RSNA labels. "
               f"It is **not** a clinically calibrated probability of disease. Inference time is "
               f"the forward pass per model at batch size 1 (CUDA-synchronised), excluding decoding.")

    # -----------------------------------------------------------------------
    # C. Model comparison
    # -----------------------------------------------------------------------
    st.header("C. Model comparison")
    cards = st.columns(3, gap="medium")
    for col, p in zip(cards, preds):
        with col, st.container(border=True):
            st.markdown(f"**{p.label}** · `{p.run_id}`")
            st.metric("Score", f"{p.score:.4f}",
                      help=f"{SCORE_LABEL}: sigmoid(logit) of this research model. "
                           "Not a clinically calibrated probability.")
            st.caption(f"{bi.MODEL_INFO[p.run_id][1]}  \nLogit {p.logit:+.4f} · "
                       f"{p.ms:.2f} ms on {p.device}")

    with st.container(border=True):
        # SVG is built only from model outputs and fixed strings (html-escaped tooltips).
        st.markdown(score_bar_svg(preds), unsafe_allow_html=True)
        st.caption("Hover a bar for its exact score. The same values are listed in the table in "
                   "section B.")
    st.markdown(
        "- **B0** - the image-only baseline: the network sees only the image-level label.\n"
        "- **B1** - B0's recipe plus a positive-target BCE on expert lesion-box regions.\n"
        "- **A3 (BG-REL)** - B0's recipe plus a listwise relative-evidence loss that ranks each "
        "lesion region above matched lung reference regions.\n\n"
        "All three share the architecture, preprocessing, optimiser recipe and seed (42); only the "
        "auxiliary training term differs. Each selected checkpoint is from epoch 3.\n\n"
        "Differences between the three scores on a single image are anecdotal; the research "
        "comparison rests on the frozen validation/test evaluation, not on individual cases.")

    # -----------------------------------------------------------------------
    # D. Regional evidence (TRAIN only, frozen P3 reference regions)
    # -----------------------------------------------------------------------
    st.header("D. Regional evidence")
    st.caption("Region logits use the research code unchanged: `regions.shared_region_scores` "
               "pools the 14×14 feature map inside each box (fractional, area-weighted) and applies "
               "the **same** linear head as the image logit. REM = lesion logit − mean(reference "
               "logits); Top-1 = lesion logit above every reference logit (definitions from "
               "`evaluate_regions.py`). This is not Grad-CAM and no region is generated here.")
    if sample is None:
        st.info("Regional evidence is available only for TRAIN images (selected from the TRAIN "
                "dataset or an upload matched to a verified TRAIN record). Lesion and reference "
                "regions are never inferred for other images.")
    elif sample.category == "Negative":
        st.info("This is a negative TRAIN image: it has no expert lesion box, so there are no lesion "
                "or reference regions to score.")
    else:
        try:
            regions = get_regions()
            lesions = regions.for_image(sample.image_id)
            with st.spinner("Scoring lesion and reference regions with B0, B1 and A3…"):
                ev = bundle.region_evidence(x, sample.image_id, lesions, regions)
        except bi.IntegrityError as e:
            st.error(f"**Regional evidence withheld - a consistency check failed.** {e}")
            lesions, ev = (), None
        except Exception as e:
            st.error(f"**Regional evidence could not be computed.** {type(e).__name__}: {e}")
            lesions, ev = (), None
        if ev is not None:
            n_k = sum(1 for l in lesions if l.k)
            st.warning("**TRAIN, in-sample.** A3 was trained to rank these exact lesions above these "
                       "exact reference regions, so a positive REM here is expected and is **not** "
                       "evidence of generalisation. Reference regions are frozen P3 geometry matches "
                       "(P3 status: pending independent review), not verified healthy tissue.")
            rl, rr = st.columns([1, 1], gap="large")
            with rl:
                st.image(bi.draw_regions(decoded.pixel01, lesions), width="stretch",
                         caption="Lesion boxes (red, solid) and frozen P3 reference regions "
                                 "(blue, dashed) on the decoded 1024×1024 radiograph.")
                st.markdown(f'<span style="display:inline-block;width:14px;height:14px;border:3px solid '
                            f'rgb{bi.GT_RED};vertical-align:-2px;margin-right:4px"></span> lesion &nbsp; '
                            f'<span style="display:inline-block;width:14px;height:14px;border:3px dashed '
                            f'rgb{bi.CONTROL_BLUE};vertical-align:-2px;margin-right:4px"></span> reference '
                            f'region', unsafe_allow_html=True)
                st.caption(bi.REGION_LEGEND)
            with rr:
                st.markdown(f"**{len(lesions)} lesion region(s); {n_k} with frozen reference regions "
                            f"(K ≥ 1).**")
                for les in lesions:
                    if les.k:
                        ctl = ", ".join(f"{c.label} (tier {c.tier}, {c.side} lung, occupancy "
                                        f"{c.occupancy:.2f})" for c in les.controls)
                        st.markdown(f"- Lesion **{les.box_number}**: K = {les.k} → {ctl}")
                    else:
                        why = bi.SKIP_REASON_TEXT.get(les.skip_reason, les.skip_reason or "not recorded")
                        st.markdown(f"- Lesion **{les.box_number}**: K = 0 - no reference region "
                                    f"({les.skip_reason}: {why}). Only its lesion logit is shown; "
                                    "REM is undefined and this lesion had no REL term in A3 training.")
                rows = []
                for run_id in bi.RUN_IDS:
                    for rec in ev[run_id]["lesions"]:
                        c = rec["control_logits"]
                        rows.append((f"**{bi.MODEL_INFO[run_id][0]}**", rec["box_number"], rec["k"],
                                     f"{rec['lesion_logit']:+.3f}",
                                     " / ".join(f"{v:+.3f}" for v in c) if c else "—",
                                     f"**{rec['rem']:+.3f}**" if rec["rem"] is not None else "—",
                                     ("yes" if rec["top1"] else "no") if rec["top1"] is not None else "—"))
                st.markdown(md_table(["Model", "Lesion", "K", "Lesion logit", "Reference logits",
                                      "REM", "Top-1"], rows))
                st.caption("Logit units of each model's shared head (not probabilities). Self-checks "
                           "passed for every model: a whole-image box reproduces the image logit, and "
                           "K ≥ 1 scores equal the training path `rel_adapter.pair_scores_from_frozen`.")

# ---------------------------------------------------------------------------
# E. About BG-REL
# ---------------------------------------------------------------------------
st.header("E. About BG-REL")
c1, c2 = st.columns(2, gap="large")
with c1:
    st.subheader("Image-level BCE (B0)")
    st.markdown("All three models share one backbone: MobileNetV2 with output stride 16, global "
                "average pooling and a single linear head producing one logit. B0 is trained only "
                "with binary cross-entropy against the RSNA image label (*Lung Opacity* vs. *Normal* "
                "or *No Lung Opacity / Not Normal*). It learns *whether* an image is positive, but "
                "nothing tells it *where* the evidence should come from.")
    st.subheader("Positive-box supervision (B1)")
    st.markdown("B1 keeps the image BCE and adds a second BCE term on *regions*: each expert lesion "
                "box of a positive TRAIN image is scored by pooling the 14×14 feature grid inside "
                "the box (fractional, area-weighted) and passing it through the **same** linear "
                "head, with a positive target. The box term is weighted by λ (0.02 → 0.06 → 0.10 "
                "over the first three epochs). B1 learns that lesions should score high, but never "
                "compares them with other lung tissue.")
with c2:
    st.subheader("Lung-aware relative evidence learning (A3)")
    st.markdown("BG-REL keeps the image BCE and adds a *listwise relative* term. For a lesion box L "
                "and up to two matched reference regions C₁…C_K from the same image, the loss is "
                "`logsumexp(s_L, s_C1, …, s_CK) − s_L`: it is small only when the lesion region "
                "scores clearly above its references. Lesions without a valid reference (K = 0) "
                "contribute only to the image BCE. The term uses the same λ schedule as B1. "
                "At inference A3 is the same single-logit network as B0 and B1 - no lung mask, box "
                "or extra input is used.")
    st.subheader("Lesion regions and reference regions")
    st.markdown("A **lesion region** is an expert-drawn RSNA Lung Opacity box. A **reference region** "
                "(control) is an unannotated area chosen during training from a PSPNet lung mask: "
                "at least 80 % lung occupancy, vertically aligned with the lesion (≤ 0.10 normalised "
                "offset), not overlapping annotated lesions, and with IoU ≤ 0.20 between controls. "
                "Controls are selected by geometry only - never by the classifier's score - and are "
                "*not* verified healthy tissue. The aim is evidence that is specific to the lesion "
                "relative to comparable lung, rather than spread across the whole image.")
st.caption("Section D scores frozen TRAIN regions only. The research regional evaluation "
           "(VAL, with PSPNet and interventions), lung-mask generation, control sampling and "
           "test-set evaluation are part of the research pipeline and are not run here.")

# ---------------------------------------------------------------------------
# F. Grad-CAM Explainability (post-hoc, image-level; separate from section D)
# ---------------------------------------------------------------------------
st.header("F. Grad-CAM Explainability")
st.markdown(
    "Grad-CAM highlights where each model's **image-level Lung Opacity logit** (before sigmoid) "
    "draws its positive evidence. It is computed on the final 1280×14×14 feature map "
    "(`features.18`, the map whose average feeds the logit), with the same preprocessed input "
    "tensor used for the predictions above.\n\n"
    "- It is a **post-hoc visualisation of prediction evidence**, not attention, and it does not "
    "show that a model attends only to a lesion or has learned medically meaningful features.\n"
    "- **Red boxes** (section A) are expert-annotated lesion regions and **blue dashed boxes** "
    "(section D) are frozen reference regions - both only when available for a TRAIN image.\n"
    "- **REM and Top-1** (section D) are BG-REL regional evidence metrics computed from region "
    "logits. Grad-CAM is **not** a replacement for them, and no REM or Top-1 is derived from "
    "these heatmaps.")
gc_x = x if decoded is not None else None
if gc_x is None:
    st.info("Select a TRAIN sample or upload an image in section A to generate Grad-CAM.")
else:
    import hashlib
    gc_key = hashlib.sha256(gc_x.detach().cpu().numpy().tobytes()).hexdigest()
    store = st.session_state.get("gradcam")
    if store is not None and store.get("key") != gc_key:   # input changed: never show stale maps
        st.session_state.pop("gradcam", None)
        store = None
    if is_raster:
        st.warning(f"**{bi.RASTER_LABEL} - out-of-distribution input.** Grad-CAM below describes the "
                   "models' response to a converted external image. It carries no RSNA label or "
                   "ground truth and must not be read as a finding.")
    if st.button("Generate Grad-CAM", key="gradcam_button", type="primary"):
        try:
            with st.spinner("Computing Grad-CAM for B0, B1 and A3…"):
                st.session_state["gradcam"] = {"key": gc_key, "results": gc.gradcam_all(bundle, gc_x)}
            store = st.session_state["gradcam"]
        except Exception as e:
            st.error(f"**Grad-CAM could not be computed.** {type(e).__name__}: {e}")
            store = None
    if store is None:
        st.caption("Grad-CAM is computed on request for the current image only.")
    else:
        results = store["results"]
        c_scale, c_op = st.columns([2, 1])
        with c_scale:
            common = st.radio("Heatmap scale", ["Common scale across B0 / B1 / A3", "Per-model scale"],
                              horizontal=True, key="gc_scale") == "Common scale across B0 / B1 / A3"
        with c_op:
            opacity = st.slider("Overlay opacity", 0.2, 0.9, 0.55, 0.05, key="gc_opacity")
        # --- Visual-only outlines (never passed to Grad-CAM; results above are already computed) ---
        gt_ok = sample is not None                     # verified TRAIN sample / matched TRAIN DICOM
        gc_gt_boxes = get_catalog().boxes(sample.image_id) if gt_ok else ()
        gc_lesions = get_regions().for_image(sample.image_id) if gt_ok and gc_gt_boxes else ()
        has_ref = any(l.k > 0 for l in gc_lesions)
        b_gt, b_ref = st.columns(2)
        with b_gt:
            show_gc_gt = st.checkbox("Show Ground Truth lesion boxes", key="gc_show_gt",
                                     disabled=not gc_gt_boxes)
        with b_ref:
            show_gc_ref = st.checkbox("Show frozen reference regions", key="gc_show_ref",
                                      disabled=not has_ref)
        st.caption("Red boxes: expert-annotated lesion regions. Blue dashed boxes: frozen reference "
                   "regions. Boxes are shown for visual comparison only and are not used to "
                   "calculate Grad-CAM.")
        if not gt_ok:
            st.info("No ground-truth lesion boxes or reference regions are shown: this image is not a "
                    "verified TRAIN sample or a DICOM matched to a verified TRAIN record. Boxes are "
                    "never inferred from file names or image content.")
        elif not gc_gt_boxes:
            st.info(f"No expert lesion box exists for this image ({sample.final_label}, negative "
                    "TRAIN image), so there is nothing to outline.")
        elif not has_ref:
            reasons = sorted({l.skip_reason for l in gc_lesions if l.k == 0} - {""})
            st.info("No valid frozen reference region exists for this image (K = 0 for every "
                    f"lesion{'; ' + ', '.join(reasons) if reasons else ''}), so only lesion boxes "
                    "can be shown.")
        gc_layers = []
        if show_gc_ref and has_ref:
            gc_layers.append(gco.reference_layer(gc_lesions))     # drawn first, like section D
        if show_gc_gt and gc_gt_boxes:
            gc_layers.append(gco.gt_layer(gc_gt_boxes))           # lesions on top
        vmax_common = max(float(r.cam.max()) for r in results)
        cols = st.columns(4, gap="small")
        with cols[0]:
            if gc_layers:
                st.image(gco.apply_layers(gco.grey_rgb(decoded.pixel01), gc_layers), width="stretch",
                         caption="Original X-ray (1024×1024 model view)")
            else:
                st.image((decoded.pixel01 * 255).round().astype(np.uint8), clamp=True, width="stretch",
                         caption="Original X-ray (1024×1024 model view)")
        for col, r in zip(cols[1:], results):
            vmax = vmax_common if common else float(r.cam.max())
            heat, is_zero = gc.normalise(gc.upsample(r.cam, decoded.pixel01.shape), vmax)
            with col:
                st.image(gco.apply_layers(gc.overlay(decoded.pixel01, heat, opacity), gc_layers),
                         width="stretch",
                         caption=f"{r.label} Grad-CAM · {bi.MODEL_INFO[r.run_id][1]}")
                st.markdown(f"**{r.label}** · logit {r.logit:+.3f} · peak {float(r.cam.max()):.4f}")
                if is_zero:
                    st.caption("No positive evidence: the weighted map is ≤ 0 everywhere, so the "
                               "heatmap is empty.")
        stops = ", ".join(gc.colormap_stops(8))
        top = vmax_common if common else None
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;max-width:520px;margin-top:4px">'
            f'<span style="font-size:13px">0</span>'
            f'<div style="flex:1;height:14px;border-radius:3px;background:linear-gradient(90deg,{stops})">'
            f'</div><span style="font-size:13px">'
            f'{f"{top:.4f}" if top is not None else "each model&#39;s own peak"}</span></div>',
            unsafe_allow_html=True)
        st.caption(
            "Colour = Grad-CAM value after ReLU: the positive contribution of each feature-map cell "
            "(≈ 73×73 px of the 1024 image) to the image logit, in logit units. "
            + ("All three panels share one scale (0 → the largest peak of the three), so their "
               "intensities are comparable." if common else
               "Each panel is scaled to its own peak, so intensities are NOT comparable across "
               "models.")
            + " Transparency rises with the value, so low-evidence areas show the radiograph "
            "unchanged. For this architecture (global average pooling + one linear layer) Grad-CAM "
            "equals the classic class activation map.")
