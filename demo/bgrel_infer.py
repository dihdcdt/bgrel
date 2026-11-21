"""BG-REL demo: verified image-level inference for the frozen B0 / B1 / A3 checkpoints.

Read-only with respect to the research project. Checkpoint loading and integrity
checks are taken from the verified smoke test (scratchpad smoke_infer.py), which
mirrors evaluate_p8a.verify_run / evaluate_b1_val.verify_b1 with one documented
exception: the dataset.py hash, replaced by a proof that the R5 file differs from
the training-time file only in the two DICOM path-lookup lines.

This module never imports torchxrayvision (it shadows the research `model` module).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # never write __pycache__ into the research code/ folder

import copy
import hashlib
import io
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Repository root = the parent of demo/. Override with the BGREL_ROOT environment variable.
RESEARCH_ROOT = Path(os.environ.get("BGREL_ROOT", Path(__file__).resolve().parents[1])).resolve()
_CODE = RESEARCH_ROOT / "code"
if not (_CODE / "model.py").is_file():
    raise RuntimeError(f"BG-REL research code not found at {_CODE}. "
                       "Set BGREL_ROOT to the repository root (the folder that contains code/).")
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import pydicom  # noqa: E402
from pydicom.errors import InvalidDicomError  # noqa: E402
import torch  # noqa: E402

# Research modules (unchanged; imported, never copied, so preprocessing cannot drift).
from dataset import prepare_image, sha256  # noqa: E402
from model import MobileNetV2OS16  # noqa: E402

assert "torchxrayvision" not in sys.modules, "torchxrayvision must not be loaded in this process"

# ---------------------------------------------------------------------------
# Constants (identical to the verified smoke test)
# ---------------------------------------------------------------------------
RECORDED_DATASET_SHA = "7e1ecaa18d1cd03f27157942645d19ca21364e975bda38d70af281755c45740c"
TRAINING_DATASET = RESEARCH_ROOT / "reproducibility/pre_r3_source/dataset.py"
SPLIT_SHA = "ccabcf432b0d266264e4ea96ca018b05a67d944af65fca4ad159f5130da7f9a7"

RUN_IDS = ("b0_dev_s42", "b1_boxpos_s42", "a3_l100_s42")
MODEL_INFO = {
    "b0_dev_s42": ("B0", "Image-level BCE baseline"),
    "b1_boxpos_s42": ("B1", "BCE + positive-box supervision"),
    "a3_l100_s42": ("A3", "BCE + lung-aware relative evidence (BG-REL)"),
}
EXPECTED_STATUS = {"b0_dev_s42": "FINISHED", "b1_boxpos_s42": "FINISHED",
                   "a3_l100_s42": "FINISHED_P6_DEVELOPMENT_PILOT"}
SOURCES = {"b0_dev_s42": ("model.py", "metrics.py"),
           "b1_boxpos_s42": ("model.py", "metrics.py", "regions.py"),
           "a3_l100_s42": ("model.py", "metrics.py")}

# TF32 policy: identical to the successful GPU smoke test and to the archived
# research evaluators, which all ran with PyTorch defaults on Ampere GPUs:
# cuDNN convolutions may use TF32, plain matmul stays full FP32. Set explicitly
# here so nothing else in the process can change it silently.
TF32_POLICY = {"cudnn.allow_tf32": True, "cuda.matmul.allow_tf32": False}
torch.backends.cudnn.allow_tf32 = TF32_POLICY["cudnn.allow_tf32"]
torch.backends.cuda.matmul.allow_tf32 = TF32_POLICY["cuda.matmul.allow_tf32"]


class IntegrityError(RuntimeError):
    """A frozen artifact or source file does not match its recorded provenance."""


class DicomInputError(ValueError):
    """The uploaded file is not a DICOM image this research pipeline accepts."""


def require(cond, msg):
    if not cond:
        raise IntegrityError(msg)


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Integrity checks (verbatim logic from the verified smoke test)
# ---------------------------------------------------------------------------
def dataset_diff_is_path_only() -> None:
    """Undo exactly the 2 R5 path-lookup edits; the result must hash to the training-time file."""
    cur = (RESEARCH_ROOT / "code/dataset.py").read_bytes()
    nl = b"\r\n" if b"\r\n" in cur else b"\n"
    imp = b"from paths import dicom_from_manifest" + nl
    new_line = b"dicom_path=dicom_from_manifest(row['dicom_path'], root=root)"
    old_line = b"dicom_path=root/row['dicom_path'].replace('\\\\','/')"
    require(cur.count(imp) == 1 and cur.count(new_line) == 1, "dataset.py: unexpected R5 edits")
    reverted = cur.replace(imp, b"", 1).replace(new_line, old_line, 1)
    require(hashlib.sha256(reverted).hexdigest() == RECORDED_DATASET_SHA,
            "dataset.py differs from training version beyond the path lookup")
    require(sha256(TRAINING_DATASET) == RECORDED_DATASET_SHA, "archived training dataset.py hash changed")


def load_checked(run_id: str) -> dict:
    """Same checks as evaluate_p8a.verify_run / evaluate_b1_val.verify_b1, except the
    dataset.py hash, which is replaced by dataset_diff_is_path_only()."""
    root = RESEARCH_ROOT
    folder = root / "results/runs" / run_id
    config, summary = read_json(folder / "config.json"), read_json(folder / "summary.json")
    require(summary.get("status") == EXPECTED_STATUS[run_id], f"{run_id}: status")
    if run_id == "b1_boxpos_s42":
        require(summary.get("variant") == "b1", "B1 variant")
        require(config.get("run_id") == run_id and config.get("seed") == 42, "B1 config")
        require(config.get("test_images_opened") is False, "B1 config TEST seal")
        require(config.get("split_sha256") == sha256(root / "data/processed/split.csv"), "B1 split")
    require(summary.get("best_epoch") == 3, f"{run_id}: best epoch")
    require(summary.get("test_images_opened") is False, f"{run_id}: TEST seal")
    require(config.get("split_sha256") == SPLIT_SHA, f"{run_id}: split sha")
    require(sha256(folder / "best.pt") == summary["checkpoint_sha256"], f"{run_id}: checkpoint sha")
    require(sha256(folder / "metrics.csv") == summary["metrics_sha256"], f"{run_id}: metrics sha")
    rec = summary["source_sha256"]
    require(rec.get("dataset.py") == RECORDED_DATASET_SHA, f"{run_id}: recorded dataset.py")
    for name in SOURCES[run_id]:
        require(sha256(root / "code" / name) == rec.get(name), f"{run_id}: source changed: {name}")
    cp = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
    require(isinstance(cp, dict) and isinstance(cp.get("model"), dict), f"{run_id}: schema")
    require(cp.get("epoch") == summary["best_epoch"], f"{run_id}: epoch")
    require(cp.get("config") == config, f"{run_id}: checkpoint/config")
    require(math.isclose(float(cp["val_auc"]), float(summary["best_val_auroc"]),
                         abs_tol=1e-8, rel_tol=0.0), f"{run_id}: val_auc")
    model = MobileNetV2OS16(pretrained=False)
    model.load_state_dict(cp["model"], strict=True)  # strict: every key and shape must match
    return {"model": model, "checkpoint_sha256": summary["checkpoint_sha256"],
            "epoch": int(cp["epoch"]), "val_auc": float(cp["val_auc"])}


# ---------------------------------------------------------------------------
# DICOM decoding for uploads: a step-for-step mirror of research decode_dicom()
# ---------------------------------------------------------------------------
@dataclass
class DecodedImage:
    pixel01: np.ndarray            # 1024x1024 float32 in [0,1], MONOCHROME1 already inverted
    metadata: dict = field(default_factory=dict)
    # Identifiers used ONLY to match an upload to a verified TRAIN record. Never displayed.
    _identity: dict = field(default_factory=dict, repr=False)


# Displayed metadata is limited to acquisition / image attributes. Identifiers
# (PatientName, PatientID, dates, UIDs) are deliberately never shown.
_DISPLAY_TAGS = ("Modality", "ViewPosition", "BodyPartExamined", "PhotometricInterpretation",
                 "Rows", "Columns", "BitsAllocated", "BitsStored", "PixelRepresentation",
                 "SamplesPerPixel", "PixelSpacing")


def decode_dicom_bytes(data: bytes) -> DecodedImage:
    """Decode an uploaded DICOM exactly like research dataset.decode_dicom().

    decode_dicom() also cross-checks SOPInstanceUID / PatientID / photometric against
    the frozen manifest row; an upload has no manifest row, so those identity checks
    cannot apply. Every other rule is enforced unchanged: 1024x1024, one sample per
    pixel, MONOCHROME1/2 (the only values the P0 audit admits), finite, non-constant.
    """
    try:
        ds = pydicom.dcmread(io.BytesIO(data))
    except InvalidDicomError as e:
        raise DicomInputError("Not a valid DICOM file (missing DICM preamble/header).") from e
    except Exception as e:  # truncated / corrupt files
        raise DicomInputError(f"Could not parse DICOM: {type(e).__name__}: {e}") from e
    if "PixelData" not in ds:
        raise DicomInputError("DICOM file contains no image (PixelData missing).")
    photometric = str(getattr(ds, "PhotometricInterpretation", ""))
    if photometric not in ("MONOCHROME1", "MONOCHROME2"):
        raise DicomInputError(f"Unsupported PhotometricInterpretation {photometric!r}; "
                              "expected MONOCHROME1 or MONOCHROME2.")
    try:
        rows, cols, spp = int(ds.Rows), int(ds.Columns), int(ds.SamplesPerPixel)
    except Exception as e:
        raise DicomInputError("DICOM is missing Rows/Columns/SamplesPerPixel.") from e
    if (rows, cols) != (1024, 1024) or spp != 1:
        raise DicomInputError(f"Unsupported image geometry {rows}x{cols}, {spp} sample(s)/pixel. "
                              "The research pipeline accepts only 1024x1024 single-channel "
                              "images (RSNA format); other sizes are rejected, not resized.")
    try:
        a = np.asarray(ds.pixel_array, dtype=np.float32)
    except Exception as e:
        ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "unknown")
        raise DicomInputError(f"Pixel data could not be decoded (transfer syntax {ts}): {e}") from e
    if a.shape != (1024, 1024) or not np.isfinite(a).all():
        raise DicomInputError("Invalid pixel array (shape or non-finite values).")
    lo = float(a.min()); hi = float(a.max())
    if hi <= lo:
        raise DicomInputError("Image has constant pixel values.")
    a = (a - lo) / (hi - lo)
    if photometric == "MONOCHROME1":
        a = 1.0 - a
    meta = {}
    for tag in _DISPLAY_TAGS:
        if tag in ds:
            v = ds.data_element(tag).value
            meta[tag] = ", ".join(str(x) for x in v) if isinstance(v, (list, tuple, pydicom.multival.MultiValue)) else str(v)
    ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
    meta["TransferSyntax"] = (getattr(ts, "name", "") or str(ts)) if ts else "n/a"
    identity = {k: str(getattr(ds, k, "")) for k in
                ("SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID", "PatientID")}
    meta["RawPixelRange"] = f"{lo:g} – {hi:g}"
    return DecodedImage(pixel01=a, metadata=meta, _identity=identity)


def preprocess(pixel01: np.ndarray) -> torch.Tensor:
    """Research prepare_image(train=False): resize 224 bilinear, 3 channels, ImageNet norm."""
    return prepare_image(pixel01, train=False)


# ---------------------------------------------------------------------------
# Model bundle: load once, reuse
# ---------------------------------------------------------------------------
@dataclass
class Prediction:
    run_id: str
    label: str
    logit: float
    score: float          # sigmoid(logit): research-model Lung Opacity score
    ms: float             # forward-pass time incl. host->device copy, CUDA-synchronised
    device: str
    note: str = ""


class ModelBundle:
    def __init__(self, device: str | None = None):
        dataset_diff_is_path_only()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
        self.records = {}
        self.models = {}
        self._cpu_models = {}
        self._lock = threading.Lock()
        for run_id in RUN_IDS:
            rec = load_checked(run_id)
            self.records[run_id] = {k: v for k, v in rec.items() if k != "model"}
            self.models[run_id] = rec["model"].to(self.device).float().eval()
        with torch.inference_mode():  # warm-up on zeros (no image data), as in the smoke test
            for m in self.models.values():
                for _ in range(3):
                    m(torch.zeros(1, 3, 224, 224, device=self.device))
        self._sync()

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _cpu_model(self, run_id):
        if run_id not in self._cpu_models:
            self._cpu_models[run_id] = copy.deepcopy(self.models[run_id]).to("cpu").eval()
        return self._cpu_models[run_id]

    def predict(self, x: torch.Tensor) -> list[Prediction]:
        """Batch size 1, eval(), inference_mode, FP32 weights/inputs, no augmentation."""
        if x.shape != (3, 224, 224) or x.dtype != torch.float32:
            raise ValueError(f"unexpected model input {tuple(x.shape)} {x.dtype}")
        out = []
        with self._lock, torch.inference_mode():
            for run_id, m in self.models.items():
                note, dev = "", self.device.type
                try:
                    self._sync(); t = time.perf_counter()
                    logit = m(x.unsqueeze(0).to(self.device)).float()
                    self._sync(); ms = (time.perf_counter() - t) * 1000
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    cm = self._cpu_model(run_id)
                    t = time.perf_counter()
                    logit = cm(x.unsqueeze(0)).float()
                    ms = (time.perf_counter() - t) * 1000
                    dev, note = "cpu", "GPU out of memory: this prediction fell back to CPU (FP32)."
                if logit.shape != (1,) or not bool(torch.isfinite(logit).all()):
                    raise RuntimeError(f"{run_id}: non-finite model output")
                v = float(logit.item())
                out.append(Prediction(run_id, MODEL_INFO[run_id][0], v,
                                      float(torch.sigmoid(torch.tensor(v)).item()), ms, dev, note))
        return out

    def region_evidence(self, x: torch.Tensor, image_id: str, lesions, region_index) -> dict:
        """Shared-head region logits for the frozen lesion/control boxes of ONE TRAIN image.

        Uses the research functions unchanged: regions.scale_boxes_xyxy (1024 -> 224, continuous)
        and regions.shared_region_scores (fractional area-weighted pooling on the 14x14 feature map,
        then the SAME linear head as the image logit). REM / Top-1 follow evaluate_regions.py:
        REM = lesion logit - mean(control logits); Top-1 = lesion logit > max(control logit).

        Two self-checks run on every call and raise IntegrityError on failure:
          * a whole-image box (0,0,1024,1024) must score the model's image logit (|d| < 1e-4);
          * for K >= 1 lesions, scores must equal the training path rel_adapter.pair_scores_from_frozen.
        """
        from regions import scale_boxes_xyxy, shared_region_scores
        from rel_adapter import pair_scores_from_frozen
        boxes = []
        for les in lesions:
            boxes.append(les.lesion_xyxy)
            boxes.extend(c.xyxy for c in les.controls)
        boxes.append((0.0, 0.0, 1024.0, 1024.0))               # whole-image consistency check
        out = {}
        with self._lock, torch.inference_mode():
            xb = x.unsqueeze(0).to(self.device)
            src = torch.tensor(boxes, dtype=torch.float32, device=self.device)
            scaled = scale_boxes_xyxy(src, source_size=(1024, 1024), input_size=(224, 224))
            owners = torch.zeros(len(boxes), dtype=torch.int64, device=self.device)
            target = torch.tensor([region_index.train_target(image_id)], dtype=torch.float32,
                                  device=self.device)
            for run_id, m in self.models.items():
                feats = m.forward_features(xb)
                image_logit = float(m.classifier(feats.mean(dim=(2, 3))).flatten()[0])
                scores = shared_region_scores(feats, m.classifier, scaled, owners,
                                              input_size=(224, 224)).float().cpu()
                if not bool(torch.isfinite(scores).all()):
                    raise IntegrityError(f"{run_id}: non-finite region score")
                whole = float(scores[-1])
                if abs(whole - image_logit) > 1e-4:
                    raise IntegrityError(f"{run_id}: whole-image region score {whole} != image logit {image_logit}")
                # Training-path parity (frozen K>=1 lesions only, exactly as used for REL).
                pairs = pair_scores_from_frozen(region_index.frozen, [image_id], target, feats, m.classifier)[0]
                groups = region_index.frozen.for_image(image_id)
                pair_by_ann = {g.annotation_id: p for g, p in zip(groups, pairs)}
                per_lesion, pos, k_pos = [], 0, 0
                for les in lesions:
                    lesion_logit = float(scores[pos])
                    ctrl = [float(v) for v in scores[pos + 1: pos + 1 + len(les.controls)]]
                    pos += 1 + len(les.controls)
                    rec = {"box_number": les.box_number, "k": les.k, "lesion_logit": lesion_logit,
                           "control_logits": ctrl, "rem": None, "top1": None}
                    if ctrl:
                        ref_l, ref_c = pair_by_ann[les.annotation_id]; k_pos += 1
                        if abs(float(ref_l) - lesion_logit) > 1e-5 or \
                                any(abs(float(a) - b) > 1e-5 for a, b in zip(ref_c.float().cpu(), ctrl)):
                            raise IntegrityError(f"{run_id}: region scores differ from training path")
                        rec["rem"] = lesion_logit - sum(ctrl) / len(ctrl)
                        rec["top1"] = lesion_logit > max(ctrl)
                    per_lesion.append(rec)
                if k_pos != len(pairs):
                    raise IntegrityError(f"{run_id}: frozen K>=1 lesion count mismatch")
                out[run_id] = {"image_logit": image_logit, "whole_image_region_logit": whole,
                               "lesions": per_lesion}
        return out


# ---------------------------------------------------------------------------
# TRAIN-only sample catalog (read-only view of the frozen split + manifest)
# ---------------------------------------------------------------------------
LABEL_CATEGORY = {"Lung Opacity": "Positive",
                  "Normal": "Negative",
                  "No Lung Opacity / Not Normal": "Negative"}
VERIFIED_EXAMPLES = ("00000032_007.png", "00000003_003.png", "00000003_000.png")
EXPECTED_TRAIN = {"images": 23748, "Positive": 5685, "Negative": 18063, "boxes": 9161}
BOXES_SHA = "6372a1facc3e1b78fbcd97a5bd103d2fc33ed257c92f8076b1d36e9168078e45"
GT_LEGEND = "Ground-truth RSNA expert annotation — not model-generated localization."


@dataclass(frozen=True)
class LesionBox:
    """One adjudicated RSNA lesion box in original 1024x1024 DICOM pixel coordinates
    (continuous; x = column, y = row; [x, x2) x [y, y2))."""
    number: int
    x: float
    y: float
    width: float
    height: float
    x2: float
    y2: float
    annotation_id: str = ""   # RSNA/MD.ai annotation key; internal join key, not displayed

    def pixel_bounds(self) -> tuple[int, int, int, int]:
        """Inclusive integer pixel bounds of the pixels the continuous box touches."""
        return (math.floor(self.x), math.floor(self.y), math.ceil(self.x2) - 1, math.ceil(self.y2) - 1)


GT_RED = (230, 25, 25)


def draw_ground_truth(pixel01: np.ndarray, boxes, line: int = 4) -> np.ndarray:
    """Overlay boxes on the SAME decoded 1024x1024 array the models receive (no resizing):
    returns uint8 RGB. The outline is drawn inward from each box's pixel bounds."""
    from PIL import Image, ImageDraw, ImageFont
    if pixel01.shape != (1024, 1024):
        raise ValueError("expected the decoded 1024x1024 radiograph")
    grey = (np.clip(pixel01, 0, 1) * 255).round().astype(np.uint8)
    img = Image.fromarray(grey).convert("RGB")   # 2-D uint8 -> greyscale "L"
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=30)
    except TypeError:  # very old Pillow
        font = ImageFont.load_default()
    for b in boxes:
        x0, y0, x1, y1 = b.pixel_bounds()
        draw.rectangle([x0, y0, x1, y1], outline=GT_RED, width=line)
        tag = str(b.number) if b.number else "?"
        tx, ty = x0 + line + 4, y0 + line + 2
        tb = draw.textbbox((tx, ty), tag, font=font)
        draw.rectangle([tb[0] - 4, tb[1] - 3, tb[2] + 4, tb[3] + 3], fill=GT_RED)
        draw.text((tx, ty), tag, fill=(255, 255, 255), font=font)
    return np.asarray(img)


@dataclass(frozen=True)
class TrainSample:
    """Display-safe fields only. No patient_id, DICOM PatientID, Study/Series/SOP UIDs or path."""
    image_id: str        # manifest image ID, e.g. "00000032_007.png" (not the DICOM filename)
    final_label: str
    category: str        # "Positive" | "Negative"
    view_position: str   # AP / PA

    @property
    def display_id(self) -> str:
        return self.image_id[:-4] if self.image_id.endswith(".png") else self.image_id


class TrainCatalog:
    """Joins split.csv and master_manifest.csv on image_id and keeps ONLY split == 'train'.

    Both frozen CSVs are SHA-256-verified and opened read-only. VAL/TEST rows are dropped
    while reading and never stored; no VAL/TEST pixel is ever read. Manifest rows (which
    contain identifiers and the UID-based DICOM path) are kept private and never displayed.
    """

    def __init__(self, root: Path = RESEARCH_ROOT):
        import csv
        from dataset import MASTER_SHA, SPLIT_SHA as DATASET_SPLIT_SHA
        self.root = root
        base = root / "data/processed"
        require(DATASET_SPLIT_SHA == SPLIT_SHA, "split SHA constant mismatch")
        require(sha256(base / "split.csv") == SPLIT_SHA, "split.csv SHA-256 mismatch")
        require(sha256(base / "master_manifest.csv") == MASTER_SHA, "master_manifest.csv SHA-256 mismatch")
        with (base / "split.csv").open("r", encoding="utf-8-sig", newline="") as f:
            train_ids = {r["image_id"]: r["patient_id"] for r in csv.DictReader(f) if r["split"] == "train"}
        rows: dict[str, dict[str, str]] = {}
        with (base / "master_manifest.csv").open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r["image_id"] in train_ids:      # join key: image_id
                    require(r["patient_id"] == train_ids[r["image_id"]], "split/manifest patient mismatch")
                    require(r["final_label"] in LABEL_CATEGORY, f"unexpected label {r['final_label']!r}")
                    require((r["final_label"] == "Lung Opacity") == (r["target"] == "1"), "target/label mismatch")
                    rows[r["image_id"]] = r
        require(set(rows) == set(train_ids), "TRAIN IDs missing from manifest")
        self._rows = rows
        samples = [TrainSample(i, r["final_label"], LABEL_CATEGORY[r["final_label"]], r["view_position"])
                   for i, r in sorted(rows.items())]
        self.by_category = {c: [s for s in samples if s.category == c] for c in ("Positive", "Negative")}
        self.by_id = {s.image_id: s for s in samples}
        require(len(samples) == EXPECTED_TRAIN["images"], "TRAIN image count changed")
        for c in ("Positive", "Negative"):
            require(len(self.by_category[c]) == EXPECTED_TRAIN[c], f"TRAIN {c} count changed")
        for v in VERIFIED_EXAMPLES:
            require(v in self.by_id, f"verified example {v} not in TRAIN")

        # Ground-truth lesion boxes (adjudicated RSNA "Calculated" group, via P0 boxes.csv).
        require(sha256(base / "boxes.csv") == BOXES_SHA, "boxes.csv SHA-256 mismatch")
        boxes: dict[str, list[LesionBox]] = {}
        with (base / "boxes.csv").open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r["image_id"] not in rows:       # TRAIN only; VAL/TEST boxes are not kept
                    continue
                m = rows[r["image_id"]]
                require(r["sop_uid"] == m["sop_uid"] and r["patient_id"] == m["patient_id"],
                        "boxes.csv / manifest identity mismatch")
                require((r["image_width"], r["image_height"]) == (m["columns"], m["rows"]) == ("1024", "1024"),
                        "box canvas is not the 1024x1024 DICOM frame")
                x, y, w, h = (float(r[k]) for k in ("x", "y", "width", "height"))
                x2, y2 = float(r["x2"]), float(r["y2"])
                require(abs(x + w - x2) < 1e-3 and abs(y + h - y2) < 1e-3, "box x2/y2 inconsistent")
                require(0 <= x < x2 <= 1024 and 0 <= y < y2 <= 1024, "box outside the 1024x1024 frame")
                boxes.setdefault(r["image_id"], []).append(
                    LesionBox(int(r["box_number"] or 0), x, y, w, h, x2, y2, r["annotation_id"]))
        for image_id, m in rows.items():
            got = boxes.get(image_id, [])
            require(len(got) == int(m["num_boxes"]), f"{image_id}: box count != manifest num_boxes")
            require((len(got) > 0) == (m["target"] == "1"), f"{image_id}: boxes inconsistent with label")
        self._boxes = {k: tuple(sorted(v, key=lambda b: b.number)) for k, v in boxes.items()}
        require(sum(len(v) for v in self._boxes.values()) == EXPECTED_TRAIN["boxes"], "TRAIN box count changed")
        self._by_sop = {r["sop_uid"]: i for i, r in rows.items()}

    def boxes(self, image_id: str) -> tuple["LesionBox", ...]:
        """Ground-truth lesion boxes for a TRAIN image (empty for negatives)."""
        if image_id not in self._rows:
            raise PermissionError(f"{image_id!r} is not a TRAIN sample; refusing access")
        return self._boxes.get(image_id, ())

    def match_upload(self, decoded: DecodedImage) -> tuple[TrainSample | None, str]:
        """Match an uploaded DICOM to a verified TRAIN record, or return (None, reason).

        A match requires ALL of: the SOP Instance UID is a TRAIN record; Study UID, Series UID
        and DICOM PatientID equal that manifest row; and the decoded pixels are bit-identical to
        the archived TRAIN DICOM. Anything else is "no confident match" - no boxes are inferred.
        Non-TRAIN records are never looked up, so the reason never reveals VAL/TEST membership.
        """
        ident = decoded._identity
        image_id = self._by_sop.get(ident.get("SOPInstanceUID", ""))
        if image_id is None:
            return None, "not a record of the TRAIN partition"
        m = self._rows[image_id]
        if (ident.get("StudyInstanceUID"), ident.get("SeriesInstanceUID"), ident.get("PatientID")) != \
                (m["study_uid"], m["series_uid"], m["dicom_patient_id"]):
            return None, "DICOM identifiers do not all agree with the TRAIN record"
        if not np.array_equal(decoded.pixel01, self.load(image_id).pixel01):
            return None, "pixel data differs from the archived TRAIN image"
        return self.by_id[image_id], "matched"

    def options(self, category: str) -> list[TrainSample]:
        """Verified examples of this category first, then the rest by image ID."""
        verified = [self.by_id[v] for v in VERIFIED_EXAMPLES if self.by_id[v].category == category]
        return verified + [s for s in self.by_category[category] if s.image_id not in VERIFIED_EXAMPLES]

    def dicom_path(self, image_id: str) -> Path:
        if image_id not in self._rows:
            raise PermissionError(f"{image_id!r} is not a TRAIN sample; refusing access")
        from paths import dicom_from_manifest
        return dicom_from_manifest(self._rows[image_id]["dicom_path"], root=self.root)

    def load(self, image_id: str) -> DecodedImage:
        """Read the DICOM resolved from the manifest and decode it with the SAME function the
        upload path uses. Additionally runs the research decode_dicom() on the manifest row,
        which verifies SOPInstanceUID / PatientID / photometric against the frozen manifest,
        and requires both decodes to be bit-identical."""
        from dataset import decode_dicom
        path = self.dicom_path(image_id)       # raises for non-TRAIN IDs before any file access
        decoded = decode_dicom_bytes(path.read_bytes())
        reference = decode_dicom(self.root, self._rows[image_id])
        if not np.array_equal(decoded.pixel01, reference):
            raise IntegrityError(f"{image_id}: upload decoder and research decoder disagree")
        return decoded


_CATALOG: TrainCatalog | None = None


def train_catalog() -> TrainCatalog:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = TrainCatalog()
    return _CATALOG


def train_reference_path(image_id: str):
    """(path, manifest row) for a TRAIN image; refuses VAL/TEST IDs. Used by tests/parity."""
    cat = train_catalog()
    path = cat.dicom_path(image_id)
    return path, cat._rows[image_id]


# ---------------------------------------------------------------------------
# Frozen P3 TRAIN reference (control) regions
# ---------------------------------------------------------------------------
P3_CONTROLS_SHA = "59ac71c8c2268d55e448bc9b674999060bb2f828d0a3c1cf541bb856367f2e55"
P3_TRACE_SHA = "2932a24672f4c5db2fe8daf13fd2a54474732f4ab72f0ecc3b13d1d56b91654f"
REGION_REFERENCE_ID = "00000032_058.png"   # TRAIN: one lesion with K=2 controls, one with K=0
SKIP_REASON_TEXT = {
    "IMAGE_SKIP_REL": "the image's P2 lung mask was flagged by QA (SKIP_REL), so P3 did not search "
                      "for reference regions",
    "SEARCH_EXHAUSTED": "no candidate passed the P3 geometry rules (lung occupancy ≥ 0.80, vertical "
                        "alignment, no lesion overlap) within the search budget",
    "SIDE_AMBIGUOUS": "the lesion's lung side could not be determined from the P2 mask",
    "NO_LUNG_OVERLAP": "the lesion does not overlap the P2 lung mask",
}


@dataclass(frozen=True)
class ControlRegion:
    label: str            # "1a", "1b": lesion box number + control letter (display only)
    tier: str
    side: str
    occupancy: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class LesionRegions:
    box_number: int
    annotation_id: str    # internal join key, not displayed
    lesion_xyxy: tuple[float, float, float, float]
    k: int
    skip_reason: str
    controls: tuple[ControlRegion, ...]


class RegionIndex:
    """TRAIN-only lesion + frozen P3 control geometry.

    Built from the research loader rel_adapter.load_frozen_controls() (SHA-pinned qa_controls.csv,
    sampler_trace.csv, split, manifest, boxes; full integrity checks) plus the same two CSVs read
    again under the same SHA pins for per-lesion K, skip reason, tier, side and occupancy. Every
    coordinate is cross-checked against the research loader and against boxes.csv. Nothing is
    sampled, generated or recomputed; PSPNet is not used.
    """

    def __init__(self, catalog: "TrainCatalog"):
        import csv
        from rel_adapter import P3_CONTROLS_SHA as RA_C, P3_TRACE_SHA as RA_T, load_frozen_controls
        require((RA_C, RA_T) == (P3_CONTROLS_SHA, P3_TRACE_SHA), "rel_adapter P3 SHA pins changed")
        self.catalog = catalog
        self.frozen = load_frozen_controls(catalog.root)
        qa = catalog.root / "results/qa"
        require(sha256(qa / "qa_controls.csv") == P3_CONTROLS_SHA, "qa_controls.csv SHA-256 mismatch")
        require(sha256(qa / "sampler_trace.csv") == P3_TRACE_SHA, "sampler_trace.csv SHA-256 mismatch")
        controls_by_ann: dict[str, list[dict]] = {}
        with (qa / "qa_controls.csv").open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                controls_by_ann.setdefault(r["annotation_id"], []).append(r)
        lesions: dict[str, list[LesionRegions]] = {}
        with (qa / "sampler_trace.csv").open("r", encoding="utf-8-sig", newline="") as f:
            for t in csv.DictReader(f):
                image_id = t["image_id"]
                require(image_id in catalog._rows, "sampler_trace contains a non-TRAIN image")
                box = next((b for b in catalog.boxes(image_id) if b.annotation_id == t["annotation_id"]), None)
                require(box is not None, "trace lesion missing from boxes.csv")
                lesion = tuple(float(t[k]) for k in ("lesion_x0", "lesion_y0", "lesion_x1", "lesion_y1"))
                require(all(abs(a - b) < 2e-4 for a, b in zip(lesion, (box.x, box.y, box.x2, box.y2))),
                        "trace lesion box differs from boxes.csv")
                rows = sorted(controls_by_ann.get(t["annotation_id"], []), key=lambda r: r["control_id"])
                k = int(t["k"])
                require(k == len(rows), "trace K != number of frozen controls")
                ctrls = tuple(ControlRegion(f"{box.number}{chr(97 + i)}", r["tier"], r["side"],
                                            float(r["occupancy"]),
                                            tuple(float(r[c]) for c in ("x0", "y0", "x1", "y1")))
                              for i, r in enumerate(rows))
                lesions.setdefault(image_id, []).append(
                    LesionRegions(box.number, t["annotation_id"], lesion, k, t["skip_reason"], ctrls))
        # Cross-check against the research loader: same K>=1 lesions, same control coordinates.
        for image_id, items in lesions.items():
            ref = {g.annotation_id: g for g in self.frozen.for_image(image_id)}
            mine = {l.annotation_id: l for l in items if l.k > 0}
            require(set(ref) == set(mine), f"{image_id}: K>=1 lesions differ from rel_adapter")
            for ann, g in ref.items():
                require(len(g.control_xyxy) == len(mine[ann].controls), "control count differs")
                for a, c in zip(g.control_xyxy, mine[ann].controls):
                    require(all(abs(p - q) < 1e-9 for p, q in zip(a, c.xyxy)), "control coords differ")
        self._lesions = {k: tuple(sorted(v, key=lambda l: l.box_number)) for k, v in lesions.items()}
        self.images_with_controls = frozenset(i for i, v in self._lesions.items() if any(l.k for l in v))
        require(len(self.images_with_controls) == self.frozen.eligible_images == 2421,
                "unexpected number of TRAIN images with frozen controls")

    def for_image(self, image_id: str) -> tuple[LesionRegions, ...]:
        if image_id not in self.catalog._rows:
            raise PermissionError(f"{image_id!r} is not a TRAIN sample; refusing access")
        return self._lesions.get(image_id, ())

    def train_target(self, image_id: str) -> int:
        return self.frozen.train_targets[image_id]


_REGIONS: RegionIndex | None = None


def region_index() -> RegionIndex:
    global _REGIONS
    if _REGIONS is None:
        _REGIONS = RegionIndex(train_catalog())
    return _REGIONS


CONTROL_BLUE = (37, 99, 235)
REGION_LEGEND = ("Red: expert lesion box (ground truth). Blue dashed: frozen P3 reference (control) "
                 "region — a geometry-matched, unannotated lung area chosen during training data "
                 "preparation; not verified healthy tissue (P3 status: pending independent review).")


def _dashed_rect(draw, x0, y0, x1, y1, color, width, dash=18, gap=10):
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x0, y1, x1, y1), (x0, y0, x0, y1), (x1, y0, x1, y1)):
        length = max(abs(bx - ax), abs(by - ay))
        pos = 0
        while pos <= length:
            end = min(pos + dash, length)
            if ax == bx:   # vertical edge: thicken inward
                xs = (ax, ax + width - 1) if ax == x0 else (ax - width + 1, ax)
                draw.rectangle([xs[0], ay + pos, xs[1], ay + end], fill=color)
            else:
                ys = (ay, ay + width - 1) if ay == y0 else (ay - width + 1, ay)
                draw.rectangle([ax + pos, ys[0], ax + end, ys[1]], fill=color)
            pos += dash + gap


def draw_regions(pixel01: np.ndarray, lesions: tuple[LesionRegions, ...], line: int = 4) -> np.ndarray:
    """Lesions (red, solid) and their frozen controls (blue, dashed) on the decoded 1024x1024 frame."""
    from PIL import Image, ImageDraw, ImageFont
    if pixel01.shape != (1024, 1024):
        raise ValueError("expected the decoded 1024x1024 radiograph")
    grey = (np.clip(pixel01, 0, 1) * 255).round().astype(np.uint8)
    img = Image.fromarray(grey).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:
        font = ImageFont.load_default()

    def bounds(xyxy):
        return (math.floor(xyxy[0]), math.floor(xyxy[1]), math.ceil(xyxy[2]) - 1, math.ceil(xyxy[3]) - 1)

    def tag(x0, y0, text, color):
        tb = draw.textbbox((x0 + line + 4, y0 + line + 2), text, font=font)
        draw.rectangle([tb[0] - 4, tb[1] - 3, tb[2] + 4, tb[3] + 3], fill=color)
        draw.text((x0 + line + 4, y0 + line + 2), text, fill=(255, 255, 255), font=font)

    for les in lesions:
        for c in les.controls:
            b = bounds(c.xyxy)
            _dashed_rect(draw, *b, CONTROL_BLUE, line)
            tag(b[0], b[1], c.label, CONTROL_BLUE)
    for les in lesions:
        b = bounds(les.lesion_xyxy)
        draw.rectangle(list(b), outline=GT_RED, width=line)
        tag(b[0], b[1], str(les.box_number), GT_RED)
    return np.asarray(img)


PARITY_REFERENCE_ID = "00000003_000.png"  # TRAIN image, already used in the smoke test


def preprocessing_parity_check(image_id: str = PARITY_REFERENCE_ID) -> dict:
    """Prove the upload path equals the research path bit-for-bit on a TRAIN image:
    decode_dicom_bytes()+preprocess() vs dataset.RSNAImageDataset(train=False)."""
    from dataset import RSNAImageDataset, decode_dicom
    path, row = train_reference_path(image_id)
    mine = decode_dicom_bytes(path.read_bytes())
    ref_pixels = decode_dicom(RESEARCH_ROOT, row)
    ref_tensor, _ = RSNAImageDataset(RESEARCH_ROOT, [row], train=False, seed=42, epoch=0)[0]
    pix_equal = bool(np.array_equal(mine.pixel01, ref_pixels))
    tensor_equal = bool(torch.equal(preprocess(mine.pixel01), ref_tensor))
    if not (pix_equal and tensor_equal):
        raise IntegrityError("Upload preprocessing differs from research RSNAImageDataset")
    return {"image_id": image_id, "decode_bitwise_equal": pix_equal, "tensor_bitwise_equal": tensor_equal}


# ===========================================================================
# External PNG/JPG rasters - SEPARATE, non-standard preprocessing path.
# Nothing above this line is used differently: the DICOM path is untouched.
# ===========================================================================
RASTER_LABEL = "External raster – non-standard preprocessing"
RASTER_FORMATS = ("PNG", "JPEG")
CANVAS = 1024


class RasterInputError(ValueError):
    """The uploaded file is not a PNG/JPG this demo can convert."""


@dataclass
class RasterImage(DecodedImage):
    """A PNG/JPG converted to the 1024x1024 canvas. `ops` documents every step taken."""
    ops: dict = field(default_factory=dict)


def letterbox_geometry(width: int, height: int, canvas: int = CANVAS) -> dict:
    """Aspect-preserving fit inside `canvas` x `canvas`, centred, no cropping.

    The limiting side becomes exactly `canvas`; the other side is round(side * scale) (so the
    aspect ratio is kept to within half a pixel). Padding is split evenly, extra pixel bottom/right.
    """
    if width <= 0 or height <= 0:
        raise RasterInputError("Image has no pixels.")
    scale = min(canvas / width, canvas / height)
    if width >= height:      # landscape or square: width is the limiting side
        new_w, new_h = canvas, max(1, min(canvas, round(height * canvas / width)))
    else:                    # portrait: height is the limiting side
        new_w, new_h = max(1, min(canvas, round(width * canvas / height))), canvas
    left, top = (canvas - new_w) // 2, (canvas - new_h) // 2
    return {"scale": scale, "resized": (new_w, new_h),
            "pad": {"left": left, "right": canvas - new_w - left,
                    "top": top, "bottom": canvas - new_h - top}}


def decode_raster_bytes(data: bytes) -> RasterImage:
    """PNG/JPG -> greyscale -> aspect-preserving resize into 1024x1024 -> pad -> min-max to [0,1].

    Steps (each recorded in `ops`):
      1. Pillow decode; only PNG and JPEG; images with transparency are rejected.
      2. Greyscale: 8/16/32-bit and float greyscale are kept as-is (lossless); colour (RGB, palette,
         CMYK) uses Pillow's standard luminance conversion convert("L") (ITU-R 601-2).
      3. Resize with Pillow BILINEAR (antialiased when shrinking; non-negative weights, so no
         overshoot) to fit inside 1024x1024 with the original aspect ratio - never stretched,
         never cropped. Skipped when the image is already 1024x1024.
      4. Pad (centred) to 1024x1024 with the MINIMUM grey value of the resized image, so the
         canvas introduces no new intensity minimum.
      5. The same per-image min-max normalisation as the DICOM path. No MONOCHROME1 inversion
         (PNG/JPG carry no photometric tag). The result goes to the unchanged preprocess().
    EXIF orientation is NOT applied: the stored pixel grid is used as-is.
    """
    from PIL import Image, UnidentifiedImageError
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img.load()
    except UnidentifiedImageError as e:
        raise RasterInputError("Not a readable PNG or JPEG image.") from e
    except Exception as e:
        raise RasterInputError(f"Could not read image: {type(e).__name__}: {e}") from e
    if fmt not in RASTER_FORMATS:
        raise RasterInputError(f"Unsupported format {fmt!r}; only PNG and JPEG are accepted here "
                               "(use the DICOM upload for .dcm files).")
    if getattr(img, "n_frames", 1) != 1:
        raise RasterInputError("Animated / multi-frame images are not supported.")
    mode = img.mode
    alpha_note = ""
    if mode in ("RGBA", "LA"):
        # Many tools save opaque images with an alpha channel. A FULLY opaque alpha carries no
        # information and is dropped losslessly; any real transparency is rejected.
        if not bool((np.asarray(img.getchannel("A")) == 255).all()):
            raise RasterInputError(f"Images with transparent pixels (mode {mode}) are not supported.")
        img = img.convert("RGB" if mode == "RGBA" else "L")
        alpha_note, mode = f"; fully opaque alpha channel dropped ({mode})", ("RGB" if mode == "RGBA" else "L")
    elif mode in ("PA", "La", "RGBa") or "transparency" in img.info:
        raise RasterInputError(f"Images with transparency (mode {mode}) are not supported.")
    orig_w, orig_h = img.size
    if mode in ("L", "I", "F") or mode.startswith("I;16"):
        grey = np.asarray(img, dtype=np.float32)
        conversion = f"none (already greyscale, mode {mode})"
    elif mode in ("RGB", "P", "CMYK", "YCbCr", "1"):
        grey = np.asarray(img.convert("L"), dtype=np.float32)
        conversion = f"{mode} → L (Pillow luminance: 0.299 R + 0.587 G + 0.114 B)" if mode != "1" \
            else "1-bit → L"
    else:
        raise RasterInputError(f"Unsupported image mode {mode!r}.")
    conversion += alpha_note
    if grey.shape != (orig_h, orig_w) or not np.isfinite(grey).all():
        raise RasterInputError("Invalid pixel array.")
    if float(grey.max()) <= float(grey.min()):
        raise RasterInputError("Image has constant pixel values.")

    geo = letterbox_geometry(orig_w, orig_h)
    new_w, new_h = geo["resized"]
    if (new_w, new_h) == (orig_w, orig_h):
        resized, interp = grey, "none (already fits exactly)"
    else:
        resized = np.asarray(Image.fromarray(grey)          # float32 2-D -> Pillow mode "F"
                             .resize((new_w, new_h), resample=Image.Resampling.BILINEAR),
                             dtype=np.float32)
        interp = "Pillow BILINEAR (antialiased when shrinking)"
    pad_value = float(resized.min())
    canvas = np.full((CANVAS, CANVAS), pad_value, dtype=np.float32)
    p = geo["pad"]
    canvas[p["top"]:p["top"] + new_h, p["left"]:p["left"] + new_w] = resized

    lo, hi = float(canvas.min()), float(canvas.max())
    if hi <= lo or not np.isfinite(canvas).all():
        raise RasterInputError("Image has constant pixel values after resizing.")
    pixel01 = (canvas - lo) / (hi - lo)

    meta = {"Source": RASTER_LABEL, "Format": fmt, "Mode": mode,
            "Original size": f"{orig_w} × {orig_h} px",
            "Greyscale conversion": conversion,
            "Grey range (original)": f"{float(grey.min()):g} – {float(grey.max()):g}"}
    ops = {"original_size": (orig_w, orig_h), "scale": geo["scale"], "resized": (new_w, new_h),
           "pad": p, "pad_value": pad_value, "interpolation": interp, "canvas": (CANVAS, CANVAS),
           "exif_orientation_applied": False, "photometric_inversion": False}
    return RasterImage(pixel01=pixel01.astype(np.float32), metadata=meta, ops=ops)
