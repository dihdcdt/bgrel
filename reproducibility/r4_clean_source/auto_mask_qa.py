"""Conservative automated P2 mask screening for the frozen 200-image TRAIN QA cohort.

This is a technical plausibility screen, NOT anatomical validation. Images with
uncertain masks are excluded from future REL control proposals, not from BCE.
The script never writes TRAIN masks and never modifies frozen P0/P1/QA200 files.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import inspect
import json
import logging
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from generate_lung_masks import (
    CANDIDATE_THRESHOLD, EXPECTED_WEIGHTS_SHA256, LEFT_INDEX, RIGHT_INDEX,
    WEIGHTS_NAME, read_csv, resize_binary_nearest, sha256,
)
from run_mask_qa import dicom_path, validate_dicom_record, validated_qa_inputs
from paths import PROJECT_ROOT

OUTPUT_FIELDS = (
    "image_id", "final_label", "view_position", "left_fraction", "right_fraction",
    "left_largest_component_share", "right_largest_component_share",
    "left_vertical_span", "right_vertical_span", "left_centroid_x", "right_centroid_x",
    "side_area_ratio", "side_height_ratio", "overlap_fraction", "stratum_outlier",
    "auto_status", "rel_policy", "auto_reasons",
)


def largest_component(mask: np.ndarray) -> dict[str, float]:
    """Compute connected geometry in the native 512x512 binary frame."""
    if mask.dtype != np.bool_ or mask.ndim != 2 or not mask.size:
        raise ValueError("Expected a nonempty 2D boolean mask array")
    area = int(mask.sum())
    if area == 0:
        return {"share": 0.0, "height": 0.0, "centroid_x": math.nan}
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    yy, xx = np.nonzero(labels == largest_label)
    return {
        "share": float(sizes[largest_label] / area),
        "height": float((int(yy.max()) - int(yy.min()) + 1) / mask.shape[0]),
        "centroid_x": float(xx.mean() / mask.shape[1]),
    }


def shape_features(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.shape != right.shape:
        raise ValueError("Left and right masks have different shapes")
    left_info, right_info = largest_component(left), largest_component(right)
    left_area = float(left.mean())
    right_area = float(right.mean())
    min_area, max_area = min(left_area, right_area), max(left_area, right_area)
    lh, rh = left_info["height"], right_info["height"]
    return {
        "left_fraction": left_area,
        "right_fraction": right_area,
        "left_largest_component_share": left_info["share"],
        "right_largest_component_share": right_info["share"],
        "left_vertical_span": lh,
        "right_vertical_span": rh,
        "left_centroid_x": left_info["centroid_x"],
        "right_centroid_x": right_info["centroid_x"],
        "side_area_ratio": min_area / max_area if max_area else 0.0,
        "side_height_ratio": min(lh, rh) / max(lh, rh) if max(lh, rh) else 0.0,
        "overlap_fraction": float(np.logical_and(left, right).mean()),
    }


def preliminary_decision(f: dict[str, float]) -> tuple[str, list[str]]:
    """Conservative engineering guards, not clinically validated cutoffs."""
    hard: list[str] = []
    uncertain: list[str] = []
    for side in ("left", "right"):
        frac = f[f"{side}_fraction"]
        comp = f[f"{side}_largest_component_share"]
        span = f[f"{side}_vertical_span"]
        if frac == 0:
            hard.append(f"EMPTY_{side.upper()}")
        elif frac < 0.02 or frac > 0.45:
            hard.append(f"EXTREME_{side.upper()}_AREA")
        elif frac < 0.045:
            uncertain.append(f"SMALL_{side.upper()}_AREA")
        if frac > 0 and comp < 0.75:
            hard.append(f"FRAGMENTED_{side.upper()}")
        elif frac > 0 and comp < 0.90:
            uncertain.append(f"POSSIBLE_FRAGMENTATION_{side.upper()}")
        if frac > 0 and span < 0.18:
            hard.append(f"SHORT_{side.upper()}_SPAN")
    if f["side_area_ratio"] < 0.25 and f["left_fraction"] > 0 and f["right_fraction"] > 0:
        hard.append("EXTREME_SIDE_AREA_ASYMMETRY")
    elif f["side_area_ratio"] < 0.50:
        uncertain.append("SIDE_AREA_ASYMMETRY")
    if f["side_height_ratio"] < 0.45 and f["left_fraction"] > 0 and f["right_fraction"] > 0:
        hard.append("EXTREME_SIDE_HEIGHT_ASYMMETRY")
    elif f["side_height_ratio"] < 0.65:
        uncertain.append("SIDE_HEIGHT_ASYMMETRY")
    if f["overlap_fraction"] > 0.01:
        hard.append("SUBSTANTIAL_OVERLAPPING_LUNGS")
    elif f["overlap_fraction"] > 0.001:
        uncertain.append("OVERLAPPING_LUNG_LABELS")
    if math.isfinite(f["left_centroid_x"]) and math.isfinite(f["right_centroid_x"]):
        if f["left_centroid_x"] <= f["right_centroid_x"]:
            uncertain.append("UNUSUAL_SIDE_ORIENTATION")
    if hard:
        return "AUTO_REJECT", sorted(set(hard + uncertain))
    if uncertain:
        return "AUTO_FLAG", sorted(set(uncertain))
    return "AUTO_SCREENED", []


def flag_stratum_outliers(records: list[dict[str, Any]]) -> None:
    """Outliers within the six fixed label x projection groups are review cues."""
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_stratum[(record["final_label"], record["view_position"])].append(record)
    for stratum in by_stratum.values():
        if len(stratum) < 10:
            raise ValueError("Too few observations for stratum-calibrated outlier screening")
        # The outlier rule flags only; it never turns a rejected image into screened.
        for metric in ("left_fraction", "right_fraction", "left_vertical_span", "right_vertical_span"):
            values = [float(r[metric]) for r in stratum]
            median = statistics.median(values)
            mad = statistics.median(abs(x - median) for x in values)
            if mad <= 1e-9:
                continue
            for record in stratum:
                robust_z = 0.67448975 * abs(float(record[metric]) - median) / mad
                if robust_z > 3.5:
                    record["_outliers"].add(metric.upper() + "_ROBUST_OUTLIER")
    for record in records:
        flags = sorted(record.pop("_outliers"))
        record["stratum_outlier"] = "|".join(flags)
        reasons = record["auto_reasons"].split("|") if record["auto_reasons"] else []
        record["auto_reasons"] = "|".join(sorted(set(reasons + flags)))
        if flags and record["auto_status"] == "AUTO_SCREENED":
            record["auto_status"] = "AUTO_FLAG"
        record["rel_policy"] = "REL_CANDIDATE_ONLY" if record["auto_status"] == "AUTO_SCREENED" else "SKIP_REL"


def validate_prior_qa(root: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict[str, Any]]:
    selected, manifest_by_id, _, selection_audit = validated_qa_inputs(root)
    qa_dir = root / "results/qa"
    prior = json.loads((qa_dir / "pspnet_manifest.json").read_text(encoding="utf-8"))
    prior_csv_file = qa_dir / "qa_images.csv"
    if prior["status"] != "VISUAL_REVIEW_REQUIRED" or prior["scope"] != "TRAIN_200_QA_INFERENCE_ONLY":
        raise ValueError("QA200 manifest scope or status unexpected")
    if prior["image_count"] != 200 or prior["qa_images_csv_sha256"] != sha256(prior_csv_file):
        raise ValueError("QA200 CSV is missing, incomplete or modified")
    if prior["qa_selection_sha256"] != selection_audit["selection_csv_sha256"]:
        raise ValueError("QA200 selection differs from frozen QA selection")
    if prior["checkpoint_sha256"] != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("Checkpoint differs from previous QA run")
    if prior["qa_script_sha256"] != sha256(root / "code/run_mask_qa.py"):
        raise ValueError("QA200 script changed since original inference")
    if prior["generator_sha256"] != sha256(root / "code/generate_lung_masks.py"):
        raise ValueError("PSPNet preprocessing script changed since QA200")
    observed = read_csv(prior_csv_file)
    if len(observed) != len(selected) or [r["image_id"] for r in observed] != [r["image_id"] for r in selected]:
        raise ValueError("QA200 image order or IDs changed")
    for prior_row, selection_row in zip(observed, selected):
        for field in ("image_id", "final_label", "view_position", "patient_id_sha256"):
            if prior_row[field] != selection_row[field]:
                raise ValueError(f"QA200 record metadata changed: {field}")
        if prior_row["review_status"] != "UNREVIEWED":
            raise ValueError("QA200 source was manually changed; stop rather than overwrite")
    return observed, manifest_by_id, prior


def write_output(root: Path, records: list[dict[str, Any]], prior: dict[str, Any]) -> None:
    qa_dir = root / "results/qa"
    csv_file = qa_dir / "qa_auto_assessment.csv"
    json_file = qa_dir / "qa_auto_audit.json"
    stage = qa_dir / ".qa_auto_stage"
    for dest in (csv_file, json_file, stage):
        if dest.exists():
            raise FileExistsError(f"Existing QA artifact: {dest}. Do not overwrite")
    stage.mkdir(parents=True, exist_ok=False)
    staged_csv = stage / csv_file.name
    with staged_csv.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in OUTPUT_FIELDS}
            for key in (
                "left_fraction", "right_fraction", "left_largest_component_share",
                "right_largest_component_share", "left_vertical_span", "right_vertical_span",
                "left_centroid_x", "right_centroid_x", "side_area_ratio", "side_height_ratio",
                "overlap_fraction",
            ):
                row[key] = f"{float(row[key]):.6f}"
            writer.writerow(row)
    breakdown: dict[str, dict[str, int]] = {}
    for group in sorted({(r["final_label"], r["view_position"]) for r in records}):
        selected = [r for r in records if (r["final_label"], r["view_position"]) == group]
        breakdown[group[0] + " | " + group[1]] = dict(sorted(Counter(r["auto_status"] for r in selected).items()))
    audit = {
        "phase": "P2", "scope": "TRAIN_200_AUTOMATED_TECHNICAL_SCREEN_ONLY",
        "status": "AUTOMATED_SCREEN_COMPLETE_NOT_ANATOMICAL_VALIDATION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_count": len(records),
        "counts": dict(sorted(Counter(r["auto_status"] for r in records).items())),
        "rel_policy_counts": dict(sorted(Counter(r["rel_policy"] for r in records).items())),
        "reason_counts": dict(sorted(Counter(flag for r in records for flag in r["auto_reasons"].split("|") if flag).items())),
        "strata": breakdown,
        "prior_qa_images_csv_sha256": prior["qa_images_csv_sha256"],
        "prior_qa_manifest_sha256": sha256(qa_dir / "pspnet_manifest.json"),
        "split_sha256": prior["split_sha256"],
        "selection_sha256": prior["qa_selection_sha256"],
        "checkpoint_sha256": prior["checkpoint_sha256"],
        "script_sha256": sha256(Path(__file__)),
        "qa_auto_csv_sha256": sha256(staged_csv),
        "rules": {
            "empty_mask": "AUTO_REJECT",
            "area_fraction": "reject if <0.02 or >0.45; flag if <0.045",
            "largest_component_share": "reject if <0.75; flag if <0.90",
            "individual_vertical_span": "reject if <0.18 of image height",
            "min_max_area_ratio": "reject if <0.25; flag if <0.50",
            "min_max_vertical_span_ratio": "reject if <0.45; flag if <0.65",
            "overlap_fraction": "reject if >0.01; flag if >0.001",
            "side_centroid_order": "flag only; orientation must not be inferred solely from x",
            "robust_stratum_outlier": "flag if 0.67448975 * |x - median| / MAD >3.5 within label x AP/PA",
        },
        "limitations": [
            "Rules are conservative engineering heuristics, not medically validated segmentation thresholds.",
            "AUTO_SCREENED is not anatomical PASS, Dice, IoU, or proof of valid regional controls.",
            "Both AUTO_FLAG and AUTO_REJECT skip REL as a safety-first default; BCE retains all images.",
            "Screening covers 200 deliberately stratified TRAIN samples, not all TRAIN images.",
            "No manual assessment was conducted; proposal's visual audit was replaced at user request.",
            "P3 control geometry and lesion/control overlap still require independent tests.",
        ],
    }
    staged_json = stage / json_file.name
    staged_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    staged_csv.rename(csv_file)
    staged_json.rename(json_file)
    stage.rmdir()


def run(root: Path, device_name: str, logger: logging.Logger) -> None:
    import pydicom
    import torch
    import torchxrayvision as xrv

    prior_rows, manifest_by_id, prior = validate_prior_qa(root)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but not available")
    weight = root / "models/pretrained" / WEIGHTS_NAME
    if sha256(weight) != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("PSPNet checkpoint hash changed")
    if importlib.metadata.version("torchxrayvision") != "1.5.7":
        raise ValueError("TorchXRayVision version changed")
    if CANDIDATE_THRESHOLD != 0.5 or LEFT_INDEX != 4 or RIGHT_INDEX != 5:
        raise ValueError("PSPNet segmentation configuration changed")
    if sha256(Path(inspect.getfile(xrv.baseline_models.chestx_det.PSPNet))) != prior["wrapper_sha256"]:
        raise ValueError("PSPNet wrapper changed since original QA200")
    if torch.__version__ != prior["torch_version"]:
        raise ValueError("PyTorch version changed since original QA200")
    for dest in (root / "results/qa/qa_auto_assessment.csv", root / "results/qa/qa_auto_audit.json", root / "results/qa/.qa_auto_stage"):
        if dest.exists():
            raise FileExistsError(f"Automated QA output exists: {dest}. Do not overwrite")
    device = torch.device(device_name)
    model = xrv.baseline_models.chestx_det.PSPNet(cache_dir=str(weight.parent)).to(device).eval()
    if len(model.targets) != 14 or model.targets[LEFT_INDEX] != "Left Lung" or model.targets[RIGHT_INDEX] != "Right Lung":
        raise ValueError("PSPNet target index mismatch")
    logger.info("P2 AUTO QA START | images=%s | device=%s | split=%s | weights=%s",
                len(prior_rows), device_name, prior["split_sha256"], EXPECTED_WEIGHTS_SHA256)
    records: list[dict[str, Any]] = []
    for index, existing in enumerate(prior_rows, start=1):
        item = manifest_by_id[existing["image_id"]]
        ds = pydicom.dcmread(dicom_path(root, item["dicom_path"]))
        display, pixels = validate_dicom_record(root, item, ds)
        tensor = torch.from_numpy(pixels.copy())[None, None].to(device)
        with torch.inference_mode():
            logits = model(tensor)
            if tuple(logits.shape) != (1, 14, 512, 512) or not bool(torch.isfinite(logits).all().item()):
                raise ValueError(f"Invalid logits for {existing['image_id']}")
            probs = torch.sigmoid(logits[0, [LEFT_INDEX, RIGHT_INDEX]]).cpu().numpy()
        mask_left = probs[0] > CANDIDATE_THRESHOLD
        mask_right = probs[1] > CANDIDATE_THRESHOLD
        features = shape_features(mask_left, mask_right)
        # Independently reproduce stored pixel occupancy in the original frame.
        left_original = resize_binary_nearest(mask_left, (display.shape[1], display.shape[0]))
        right_original = resize_binary_nearest(mask_right, (display.shape[1], display.shape[0]))
        for label, mask in (("left_fraction", left_original), ("right_fraction", right_original)):
            if abs(float(mask.mean()) - float(existing[label])) > 2e-6:
                raise ValueError(f"Re-inference differs from frozen QA200: {existing['image_id']} {label}")
        status, reasons = preliminary_decision(features)
        row: dict[str, Any] = {
            "image_id": existing["image_id"], "final_label": existing["final_label"],
            "view_position": existing["view_position"], **features,
            "stratum_outlier": "", "auto_status": status,
            "rel_policy": "REL_CANDIDATE_ONLY" if status == "AUTO_SCREENED" else "SKIP_REL",
            "auto_reasons": "|".join(reasons), "_outliers": set(),
        }
        records.append(row)
        if index % 10 == 0 or index == len(prior_rows):
            logger.info("P2 AUTO QA | progress=%s/%s", index, len(prior_rows))
            print(f"P2 auto QA: {index}/{len(prior_rows)}")
        del tensor, logits, probs
    flag_stratum_outliers(records)
    write_output(root, records, prior)
    logger.info("P2 AUTO QA RUN END | TECHNICAL SCREEN COMPLETE | statuses=%s",
                dict(Counter(r["auto_status"] for r in records)))
    print("P2 AUTO QA TECHNICAL SCREEN COMPLETE; NOT ANATOMICAL VALIDATION")
    print("Report: results/qa/qa_auto_audit.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", action="store_true", help="screen the existing frozen 200-image TRAIN QA set")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if not args.qa:
        parser.error("Only --qa is supported")
    logfile = PROJECT_ROOT / "results/logs/auto_mask_qa.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bgrel.p2.auto_mask_qa")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    try:
        logger.info("RUN START | P2 | automated QA | %s", args.device)
        run(PROJECT_ROOT, args.device, logger)
        return 0
    except Exception:
        logger.exception("RUN END: FAIL | P2 automated QA")
        raise
    finally:
        logger.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    raise SystemExit(main())
