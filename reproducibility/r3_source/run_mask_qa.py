"""Generate visual-review evidence for the frozen 200-image TRAIN PSPNet QA set.

No training masks are produced. Output remains UNREVIEWED until a separate
manual anatomical review is completed. This script never alters P0/P1 files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from paths import dicom_from_manifest
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from generate_lung_masks import (
    CANDIDATE_THRESHOLD,
    EXPECTED_WEIGHTS_SHA256,
    LEFT_INDEX,
    RIGHT_INDEX,
    WEIGHTS_NAME,
    create_overlay,
    normalize_dicom_pixels,
    read_csv,
    resize_binary_nearest,
    sha256,
    verified_split,
)
from settings import PROJECT_ROOT


EXPECTED_MANIFEST_SHA = "4c690ddf7918d0890b6715dba290be85a16959adc0dcaa70b5b4c5bd387d6c9b"
EXPECTED_BOXES_SHA = "6372a1facc3e1b78fbcd97a5bd103d2fc33ed257c92f8076b1d36e9168078e45"
EXPECTED_SELECTION_SIZE = 200
EXPECTED_QUOTAS = {
    ("Lung Opacity", "AP"): 34,
    ("Lung Opacity", "PA"): 34,
    ("Normal", "AP"): 33,
    ("Normal", "PA"): 33,
    ("No Lung Opacity / Not Normal", "AP"): 33,
    ("No Lung Opacity / Not Normal", "PA"): 33,
}
QA_FIELDS = (
    "image_id", "patient_id_sha256", "final_label", "view_position",
    "num_boxes", "box_count_group", "lesion_size_quartile", "overlay",
    "sheet_file", "left_fraction", "right_fraction", "overlap_fraction",
    "empty_left", "empty_right", "input_min", "input_max",
    "logits_min", "logits_max", "image_height", "image_width",
    "auto_flags", "review_status", "reviewer", "review_reason",
)


def validated_qa_inputs(root: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, list[dict[str, str]]], dict[str, Any]]:
    """Recheck frozen lineage and every QA image against the full TRAIN manifest."""
    processed = root / "data/processed"
    qa = root / "results/qa"
    manifest_file = processed / "master_manifest.csv"
    boxes_file = processed / "boxes.csv"
    selection_file = qa / "mask_qa_selection.csv"
    selection_audit_file = qa / "mask_qa_selection.json"
    smoke_file = qa / "pspnet_smoke.json"

    if sha256(manifest_file) != EXPECTED_MANIFEST_SHA or sha256(boxes_file) != EXPECTED_BOXES_SHA:
        raise ValueError("P0 manifest or boxes hash changed")
    split, split_hash = verified_split(root)
    split_by_id = {row["image_id"]: row for row in split}
    manifest = read_csv(manifest_file)
    manifest_by_id = {row["image_id"]: row for row in manifest}
    if len(manifest_by_id) != len(manifest) or set(manifest_by_id) != set(split_by_id):
        raise ValueError("P0/P1 image ID sets disagree")
    selections = read_csv(selection_file)
    audit = json.loads(selection_audit_file.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_file.read_text(encoding="utf-8"))
    if audit.get("status") != "SELECTION_REVIEW_REQUIRED" or audit.get("scope") != "TRAIN_QA_SELECTION_ONLY":
        raise ValueError("QA selection has an unexpected status or scope")
    for label, expected in (
        ("split_sha256", split_hash),
        ("manifest_sha256", EXPECTED_MANIFEST_SHA),
        ("boxes_sha256", EXPECTED_BOXES_SHA),
        ("selection_csv_sha256", sha256(selection_file)),
        ("smoke_json_sha256", sha256(smoke_file)),
    ):
        if audit.get(label) != expected:
            raise ValueError(f"QA selection provenance mismatch: {label}")
    if audit.get("seed") != 42 or audit.get("sample_size") != EXPECTED_SELECTION_SIZE:
        raise ValueError("Unexpected QA sample size or random seed")
    if audit.get("unique_patient_count") != EXPECTED_SELECTION_SIZE or audit.get("excluded_smoke_image_count") != 8:
        raise ValueError("QA selection audit count mismatch")
    if audit.get("source_sha256") != sha256(root / "code/prepare_mask_qa.py"):
        raise ValueError("QA sampling source changed since sample was frozen")
    audited_strata = {(item["label"], item["view"]): item["n"] for item in audit.get("strata", [])}
    if audited_strata != EXPECTED_QUOTAS:
        raise ValueError("QA selection audit strata mismatch")
    if smoke.get("scope") != "TRAIN_ONLY_SMOKE" or smoke.get("split_sha256") != split_hash:
        raise ValueError("Smoke and frozen split disagree")
    if smoke.get("checkpoint_sha256") != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("Smoke checkpoint differs from current generator contract")
    if smoke.get("generator_source_sha256") != sha256(root / "code/generate_lung_masks.py"):
        raise ValueError("Shared PSPNet preprocessing code changed since smoke run")
    if len(selections) != EXPECTED_SELECTION_SIZE:
        raise ValueError("QA selection must contain exactly 200 images")
    ids = [row["image_id"] for row in selections]
    if len(set(ids)) != EXPECTED_SELECTION_SIZE or ids != sorted(ids):
        raise ValueError("QA selection contains duplicate or unordered IDs")
    smoke_ids = {row["image_id"] for row in smoke.get("images", [])}
    smoke_patients = {manifest_by_id[image_id]["patient_id"] for image_id in smoke_ids}
    patients: set[str] = set()
    strata: Counter[tuple[str, str]] = Counter()
    boxes_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(boxes_file):
        if row["image_id"] not in manifest_by_id:
            raise ValueError("Expert box references unknown image")
        boxes_by_id[row["image_id"]].append(row)
    for selected in selections:
        image_id = selected["image_id"]
        if not re.fullmatch(r"\d{8}_\d{3}\.png", image_id) or image_id not in manifest_by_id:
            raise ValueError("Unsafe or unknown image ID in QA selection")
        item = manifest_by_id[image_id]
        frozen = split_by_id[image_id]
        patient = item["patient_id"]
        if frozen["split"] != "train" or frozen["patient_id"] != patient:
            raise ValueError("QA selection contains non-TRAIN image or incorrect patient")
        if patient in patients or patient in smoke_patients or image_id in smoke_ids:
            raise ValueError("Duplicate patient or smoke challenge contamination")
        if selected["patient_id_sha256"] != hashlib.sha256(patient.encode("utf-8")).hexdigest():
            raise ValueError("QA patient hash mismatch")
        if (item["final_label"], item["view_position"]) != (selected["final_label"], selected["view_position"]):
            raise ValueError("QA label or projection differs from manifest")
        if int(item["num_boxes"]) != int(selected["num_boxes"]) or int(item["num_boxes"]) != len(boxes_by_id[image_id]):
            raise ValueError("QA box count inconsistent with P0")
        if selected["review_status"] != "UNREVIEWED" or selected["review_reason"].strip():
            raise ValueError("Selection was edited after freeze; do not mix manual review with frozen selection")
        if item["final_label"] == "Lung Opacity":
            if selected["box_count_group"] not in {"SINGLE", "MULTI"} or selected["lesion_size_quartile"] not in {"Q1", "Q2", "Q3", "Q4"}:
                raise ValueError("Invalid positive QA subgroup metadata")
        else:
            if selected["box_count_group"] != "NONE" or selected["lesion_size_quartile"]:
                raise ValueError("Invalid negative QA subgroup metadata")
        patients.add(patient)
        strata[(selected["final_label"], selected["view_position"])] += 1
    if strata != Counter(EXPECTED_QUOTAS):
        raise ValueError("QA selection class-by-projection quota mismatch")
    return selections, manifest_by_id, boxes_by_id, audit


def output_locations(root: Path, selected: list[dict[str, str]]) -> dict[str, Path]:
    """Fail closed rather than overwrite reviewed evidence or previous attempts."""
    qa = root / "results/qa"
    paths = {
        "stage": qa / ".qa_200_stage",
        "overlays": qa / "overlays/qa_200",
        "sheets": qa / "qa_sheets",
        "csv": qa / "qa_images.csv",
        "manifest": qa / "pspnet_manifest.json",
    }
    for name, path in paths.items():
        if path.exists():
            raise FileExistsError(f"Existing QA artifact ({name}): {path}. Do not overwrite")
    if not selected:
        raise ValueError("QA selection is empty")
    return paths


def validate_dicom_record(root: Path, record: dict[str, str], ds: Any) -> tuple[np.ndarray, np.ndarray]:
    """Decode unchanged source pixels and enforce P0 image geometry/UID."""
    if str(ds.SOPInstanceUID) != record["sop_uid"]:
        raise ValueError(f"SOP UID mismatch for {record['image_id']}")
    if str(ds.PhotometricInterpretation) != record["photometric_interpretation"]:
        raise ValueError(f"Photometric interpretation mismatch for {record['image_id']}")
    display, normalized = normalize_dicom_pixels(ds)
    if display.shape != (int(record["rows"]), int(record["columns"])):
        raise ValueError(f"DICOM/manifest geometry mismatch for {record['image_id']}")
    return display, normalized


def dicom_path(root: Path, relative: str) -> Path:
    """Map immutable manifest path to the proposed data/raw location."""
    return dicom_from_manifest(relative, root=root)


def make_contact_sheets(records: list[dict[str, str]], source_dir: Path, target_dir: Path) -> list[str]:
    """Create overview sheets for all 200 overlays; full-size images stay available."""
    target_dir.mkdir(parents=True, exist_ok=False)
    sorted_records = sorted(records, key=lambda row: (row["final_label"], row["view_position"], row["image_id"]))
    per_sheet, cols, cell_w, cell_h = 20, 4, 268, 290
    output_names: list[str] = []
    for start in range(0, len(sorted_records), per_sheet):
        index = start // per_sheet + 1
        page = sorted_records[start:start + per_sheet]
        canvas = Image.new("RGB", (cols * cell_w, 5 * cell_h + 36), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), f"BG-REL P2 | QA overview {index:02d} | NOT ANATOMICALLY REVIEWED", fill="black")
        for j, item in enumerate(page):
            col, row = j % cols, j // cols
            x, y = col * cell_w + 6, row * cell_h + 40
            with Image.open(source_dir / f"qa_{Path(item['image_id']).stem}.png") as src:
                thumb = src.convert("RGB").resize((256, 256), resample=Image.Resampling.BILINEAR)
            canvas.paste(thumb, (x, y))
            label = "POS" if item["final_label"] == "Lung Opacity" else ("NOR" if item["final_label"] == "Normal" else "OTH")
            draw.text((x, y + 258), f"{item['image_id']} | {label} / {item['view_position']}", fill="black")
        filename = f"qa_sheet_{index:02d}.png"
        canvas.save(target_dir / filename, format="PNG")
        output_names.append(filename)
    return output_names


def write_csv(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QA_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def make_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bgrel.p2.qa200")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def run(root: Path, device_name: str, logger: logging.Logger) -> None:
    """Run frozen PSPNet on QA images only, retaining raw masks out of training."""
    import pydicom
    import torch
    import torchxrayvision as xrv

    selection, manifest_by_id, boxes_by_id, audit = validated_qa_inputs(root)
    paths = output_locations(root, selection)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    weights = root / "models/pretrained" / WEIGHTS_NAME
    if sha256(weights) != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("PSPNet checkpoint hash differs from verified download")
    if importlib.metadata.version("torchxrayvision") != "1.5.7":
        raise ValueError("TorchXRayVision version changed")
    smoke = json.loads((root / "results/qa/pspnet_smoke.json").read_text(encoding="utf-8"))
    wrapper_hash = sha256(Path(inspect.getfile(xrv.baseline_models.chestx_det.PSPNet)))
    if wrapper_hash != smoke["pspnet_wrapper_source_sha256"]:
        raise ValueError("PSPNet wrapper source changed since smoke run")
    if torch.__version__ != smoke["torch_version"]:
        raise ValueError("PyTorch version changed since smoke run")
    if LEFT_INDEX != 4 or RIGHT_INDEX != 5 or CANDIDATE_THRESHOLD != 0.5:
        raise ValueError("Frozen PSPNet preprocessing/threshold contract changed")
    device = torch.device(device_name)
    model = xrv.baseline_models.chestx_det.PSPNet(cache_dir=str(weights.parent)).to(device).eval()
    if len(model.targets) != 14 or model.targets[LEFT_INDEX] != "Left Lung" or model.targets[RIGHT_INDEX] != "Right Lung":
        raise ValueError("PSPNet targets changed")

    stage = paths["stage"]
    stage.mkdir(parents=True, exist_ok=False)
    stage_overlays = stage / "overlays"
    stage_overlays.mkdir()
    logger.info("P2 QA200 | seed=42 | sample_sha256=%s | split_sha256=%s | weights_sha256=%s | images=%d | device=%s",
                audit["selection_csv_sha256"], audit["split_sha256"], EXPECTED_WEIGHTS_SHA256, len(selection), device_name)
    records: list[dict[str, str]] = []
    for index, selected in enumerate(selection, start=1):
        image_id = selected["image_id"]
        record = manifest_by_id[image_id]
        ds = pydicom.dcmread(dicom_path(root, record["dicom_path"]))
        display, normalized = validate_dicom_record(root, record, ds)
        tensor = torch.from_numpy(normalized.copy())[None, None].to(device)
        with torch.inference_mode():
            logits = model(tensor)
            if tuple(logits.shape) != (1, 14, 512, 512) or not bool(torch.isfinite(logits).all().item()):
                raise ValueError(f"Unexpected or non-finite PSPNet logits: {image_id}")
            probabilities = torch.sigmoid(logits[0, [LEFT_INDEX, RIGHT_INDEX]]).cpu().numpy()
            logits_min, logits_max = float(logits.min().item()), float(logits.max().item())
        h, w = display.shape
        left = resize_binary_nearest(probabilities[0] > CANDIDATE_THRESHOLD, (w, h))
        right = resize_binary_nearest(probabilities[1] > CANDIDATE_THRESHOLD, (w, h))
        filename = f"qa_{Path(image_id).stem}.png"
        create_overlay(display, left, right, boxes_by_id[image_id], stage_overlays / filename)
        area = float(display.size)
        left_frac, right_frac = float(left.sum()) / area, float(right.sum()) / area
        overlap_frac = float(np.logical_and(left, right).sum()) / area
        flags = []
        if not left.any():
            flags.append("EMPTY_LEFT")
        if not right.any():
            flags.append("EMPTY_RIGHT")
        records.append({
            "image_id": image_id,
            "patient_id_sha256": selected["patient_id_sha256"],
            "final_label": selected["final_label"],
            "view_position": selected["view_position"],
            "num_boxes": selected["num_boxes"],
            "box_count_group": selected["box_count_group"],
            "lesion_size_quartile": selected["lesion_size_quartile"],
            "overlay": f"results/qa/overlays/qa_200/{filename}",
            "sheet_file": "",  # Filled after overview sheets are created.
            "left_fraction": f"{left_frac:.6f}",
            "right_fraction": f"{right_frac:.6f}",
            "overlap_fraction": f"{overlap_frac:.6f}",
            "empty_left": str(not bool(left.any())),
            "empty_right": str(not bool(right.any())),
            "input_min": f"{float(normalized.min()):.6f}",
            "input_max": f"{float(normalized.max()):.6f}",
            "logits_min": f"{logits_min:.6f}",
            "logits_max": f"{logits_max:.6f}",
            "image_height": str(h),
            "image_width": str(w),
            "auto_flags": "|".join(flags),
            "review_status": "UNREVIEWED",
            "reviewer": "",
            "review_reason": "",
        })
        if index % 10 == 0 or index == len(selection):
            logger.info("P2 QA200 | progress=%d/%d", index, len(selection))
            print(f"P2 QA200: {index}/{len(selection)}")
        del logits, probabilities, tensor

    sheet_names = make_contact_sheets(records, stage_overlays, stage / "sheets")
    sheet_assignments: dict[str, str] = {}
    ordered = sorted(records, key=lambda row: (row["final_label"], row["view_position"], row["image_id"]))
    for offset, row in enumerate(ordered):
        sheet_assignments[row["image_id"]] = f"results/qa/qa_sheets/qa_sheet_{offset // 20 + 1:02d}.png"
    for row in records:
        row["sheet_file"] = sheet_assignments[row["image_id"]]
    stage_csv = stage / "qa_images.csv"
    write_csv(stage_csv, records)
    output_hashes = {
        "overlay_sha256": {f"qa_{Path(r['image_id']).stem}.png": sha256(stage_overlays / f"qa_{Path(r['image_id']).stem}.png") for r in records},
        "sheet_sha256": {name: sha256(stage / "sheets" / name) for name in sheet_names},
    }
    output_manifest = {
        "phase": "P2", "scope": "TRAIN_200_QA_INFERENCE_ONLY", "status": "VISUAL_REVIEW_REQUIRED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_count": len(records), "unique_patient_count": len(set(row["patient_id_sha256"] for row in records)),
        "review_status_counts": dict(Counter(row["review_status"] for row in records)),
        "split_sha256": audit["split_sha256"],
        "master_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "boxes_sha256": EXPECTED_BOXES_SHA,
        "qa_selection_sha256": audit["selection_csv_sha256"],
        "qa_selection_audit_sha256": sha256(root / "results/qa/mask_qa_selection.json"),
        "smoke_sha256": sha256(root / "results/qa/pspnet_smoke.json"),
        "checkpoint_file": WEIGHTS_NAME,
        "checkpoint_sha256": EXPECTED_WEIGHTS_SHA256,
        "checkpoint_source": "https://github.com/mlmed/torchxrayvision/releases/download/v1/pspnet_chestxray_best_model_4.pth",
        "model": "torchxrayvision.baseline_models.chestx_det.PSPNet",
        "torchxrayvision_version": importlib.metadata.version("torchxrayvision"),
        "torch_version": torch.__version__, "torchvision_version": importlib.metadata.version("torchvision"),
        "numpy_version": np.__version__, "device": device_name,
        "wrapper_sha256": wrapper_hash,
        "generator_sha256": sha256(root / "code/generate_lung_masks.py"),
        "qa_script_sha256": sha256(Path(__file__)),
        "preprocessing": "DICOM raw 1-channel unsigned grayscale -> MONOCHROME1 invert if needed -> [-1024,1024] -> PSPNet internal resize 512 and ImageNet standardization",
        "output_shape": [1, 14, 512, 512],
        "target_channels": {"left_lung": LEFT_INDEX, "right_lung": RIGHT_INDEX},
        "probability_transform": "sigmoid", "threshold": CANDIDATE_THRESHOLD,
        "mask_mapping": "binary logits frame 512x512 -> original image nearest-neighbor",
        "qa_images_csv_sha256": sha256(stage_csv),
        "overlays": {"directory": "results/qa/overlays/qa_200", "count": len(records)},
        "contact_sheets": {"directory": "results/qa/qa_sheets", "count": len(sheet_names)},
        "output_hashes": output_hashes,
        "limitations": [
            "No anatomical QA has been performed; all 200 rows are UNREVIEWED.",
            "Quantitative area and overlap metrics are visual-review cues, not segmentation accuracy.",
            "A stratified enriched sample does not estimate population failure prevalence.",
            "No ground-truth lung masks; Dice or IoU cannot be calculated.",
            "No data/lung_masks/train outputs were created.",
        ],
    }
    (stage / "pspnet_manifest.json").write_text(
        json.dumps(output_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Publish directory-based outputs first; the manifest is the final completion marker.
    stage_overlays.rename(paths["overlays"])
    (stage / "sheets").rename(paths["sheets"])
    stage_csv.rename(paths["csv"])
    (stage / "pspnet_manifest.json").rename(paths["manifest"])
    stage.rmdir()
    logger.info("P2 QA200 | RUN END: COMPUTE PASS | VISUAL REVIEW REQUIRED | csv_sha256=%s", output_manifest["qa_images_csv_sha256"])
    print("P2 QA200 COMPUTE PASS; 200 ANATOMICAL REVIEWS STILL REQUIRED")
    print("Report: results/qa/pspnet_manifest.json")
    print("Review table: results/qa/qa_images.csv")
    print("Contact sheets: results/qa/qa_sheets/qa_sheet_*.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", action="store_true", help="run frozen TRAIN-only 200-image QA inference")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if not args.qa:
        parser.error("Only --qa is supported")
    log = make_logger(PROJECT_ROOT / "results/logs/run_mask_qa.log")
    try:
        log.info("RUN START | P2 | QA200 | device=%s", args.device)
        run(PROJECT_ROOT, args.device, log)
        return 0
    except Exception:
        log.exception("RUN END: FAIL | P2 QA200")
        raise
    finally:
        for handler in list(log.handlers):
            handler.close()
            log.removeHandler(handler)


if __name__ == "__main__":
    raise SystemExit(main())
