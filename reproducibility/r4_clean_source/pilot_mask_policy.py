"""Run a small out-of-QA200 P2 PSPNet policy pilot on frozen TRAIN only.

This diagnostic does not save masks for REL, change training decisions, or
validate segmentation accuracy. Its output requires an independent QA review.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import inspect
import json
import logging
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from auto_mask_qa import shape_features
from generate_lung_masks import (
    CANDIDATE_THRESHOLD, EXPECTED_WEIGHTS_SHA256, LEFT_INDEX, RIGHT_INDEX,
    WEIGHTS_NAME, create_overlay, read_csv, resize_binary_nearest, sha256,
    verified_split,
)
from mask_policy import POLICY_ID, decide_geometry
from pilot_lung_masks import verified_evidence
from run_mask_qa import dicom_path, make_contact_sheets, validate_dicom_record
from paths import PROJECT_ROOT

SEED = 42
SAMPLE_PER_STRATUM = 6
SAMPLE_SIZE = 6 * SAMPLE_PER_STRATUM
STRATA = (
    ("Lung Opacity", "AP"), ("Lung Opacity", "PA"),
    ("Normal", "AP"), ("Normal", "PA"),
    ("No Lung Opacity / Not Normal", "AP"),
    ("No Lung Opacity / Not Normal", "PA"),
)
CSV_FIELDS = (
    "image_id", "final_label", "view_position", "geometry_status",
    "policy_for_diagnostic_only", "geometry_reasons", "left_fraction",
    "right_fraction", "left_largest_component_share",
    "right_largest_component_share", "left_vertical_span",
    "right_vertical_span", "side_area_ratio", "side_height_ratio",
    "overlap_fraction", "overlay",
)


def choose_fresh_train(
    manifest_by_id: dict[str, dict[str, str]],
    split_rows: list[dict[str, str]],
    excluded_patient_ids: set[str],
    seed: int = SEED,
) -> list[dict[str, str]]:
    """Sample six class/view strata, one image per new TRAIN patient."""
    split_by_id = {r["image_id"]: r for r in split_rows}
    if len(split_by_id) != len(split_rows) or set(split_by_id) != set(manifest_by_id):
        raise ValueError("Split and manifest image IDs do not match")
    by_stratum: dict[tuple[str, str], dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for image_id, row in manifest_by_id.items():
        spl = split_by_id[image_id]
        if row["patient_id"] != spl["patient_id"]:
            raise ValueError("Patient identity mismatch")
        if spl["split"] != "train" or row["patient_id"] in excluded_patient_ids:
            continue
        key = (row["final_label"], row["view_position"])
        if key not in STRATA:
            raise ValueError("Unexpected final label or projection")
        by_stratum[key][row["patient_id"]].append(row)
    rng = random.Random(seed)
    selected: list[dict[str, str]] = []
    used = set(excluded_patient_ids)
    for key in STRATA:
        available = sorted(p for p in by_stratum[key] if p not in used)
        if len(available) < SAMPLE_PER_STRATUM:
            raise ValueError(f"Not enough disjoint TRAIN patients for {key}")
        for patient in rng.sample(available, SAMPLE_PER_STRATUM):
            records = sorted(by_stratum[key][patient], key=lambda x: x["image_id"])
            row = records[rng.randrange(len(records))]
            selected.append(row)
            used.add(patient)
    if len(selected) != SAMPLE_SIZE or len({r["patient_id"] for r in selected}) != SAMPLE_SIZE:
        raise AssertionError("Fresh cohort is not patient-disjoint")
    return selected


def validate_policy_source(root: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict, dict]:
    """Verify frozen QA lineage; do not pretend this policy is approved for bulk."""
    reviewed, manifest_by_id, qa_manifest, auto_audit = verified_evidence(root)
    qa = root / "results/qa"
    policy_file = qa / "mask_policy_audit.json"
    policy = json.loads(policy_file.read_text(encoding="utf-8"))
    if policy.get("status") != "POLICY_AUDIT_COMPLETE_NOT_APPROVED_FOR_FULL_TRAIN":
        raise ValueError("Expected the unapproved full-TRAIN policy audit")
    if policy.get("policy_id") != POLICY_ID:
        raise ValueError("Mask policy ID changed")
    for key, p in (
        ("source_sha256", root / "code/mask_policy.py"),
        ("audit_csv_sha256", qa / "mask_policy_audit.csv"),
        ("pilot_manifest_sha256", qa / "pilot_mask_manifest.json"),
    ):
        if policy.get(key) != sha256(p):
            raise ValueError(f"Mask policy provenance mismatch: {key}")
    if policy.get("split_sha256") != auto_audit["split_sha256"]:
        raise ValueError("Policy references different frozen split")
    return reviewed, manifest_by_id, qa_manifest, policy


def run(root: Path, device_name: str, logger: logging.Logger) -> None:
    import pydicom
    import torch
    import torchxrayvision as xrv

    qa = root / "results/qa"
    outputs = {
        "csv": qa / "mask_policy_fresh_pilot.csv",
        "json": qa / "mask_policy_fresh_pilot.json",
        "overlays": qa / "overlays/policy_fresh",
        "sheets": qa / "qa_policy_fresh_sheets",
        "stage": qa / ".mask_policy_fresh_stage",
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("Fresh pilot artifacts already exist; do not overwrite")
    reviewed, manifest, qa_manifest, policy = validate_policy_source(root)
    split_rows, split_hash = verified_split(root)
    if split_hash != policy["split_sha256"]:
        raise ValueError("Split hash drift")
    smoke = json.loads((qa / "pspnet_smoke.json").read_text(encoding="utf-8"))
    excluded = {manifest[r["image_id"]]["patient_id"] for r in reviewed}
    excluded.update(manifest[r["image_id"]]["patient_id"] for r in smoke["images"])
    chosen = choose_fresh_train(manifest, split_rows, excluded)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    weights = root / "models/pretrained" / WEIGHTS_NAME
    if sha256(weights) != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("PSPNet checkpoint hash changed")
    if importlib.metadata.version("torchxrayvision") != "1.5.7":
        raise ValueError("TorchXRayVision version changed")
    if torch.__version__ != qa_manifest["torch_version"]:
        raise ValueError("PyTorch version drift")
    if sha256(Path(inspect.getfile(xrv.baseline_models.chestx_det.PSPNet))) != qa_manifest["wrapper_sha256"]:
        raise ValueError("PSPNet wrapper source drift")
    if (CANDIDATE_THRESHOLD, LEFT_INDEX, RIGHT_INDEX) != (0.5, 4, 5):
        raise ValueError("PSPNet preprocessing contract drift")
    model = xrv.baseline_models.chestx_det.PSPNet(cache_dir=str(weights.parent)).to(device_name).eval()
    if (len(model.targets) != 14 or model.targets[LEFT_INDEX] != "Left Lung" or
            model.targets[RIGHT_INDEX] != "Right Lung"):
        raise ValueError("Unexpected target mapping")
    stage = outputs["stage"]
    stage.mkdir(parents=True, exist_ok=False)
    ov = stage / "overlays"; ov.mkdir()
    sheets = stage / "sheets"  # make_contact_sheets creates this directory
    boxes_by_id = defaultdict(list)
    for box in read_csv(root / "data/processed/boxes.csv"):
        boxes_by_id[box["image_id"]].append(box)
    result = []
    logger.info("RUN START P2 fresh policy diagnostic | %s | split=%s | n=%d",
                POLICY_ID, split_hash, SAMPLE_SIZE)
    for index, row in enumerate(chosen, 1):
        image_id = row["image_id"]
        ds = pydicom.dcmread(dicom_path(root, row["dicom_path"]))
        display, normalized = validate_dicom_record(root, row, ds)
        if display.shape != (1024, 1024):
            raise ValueError(f"Unrecognized DICOM geometry: {image_id}")
        tensor = torch.from_numpy(normalized.copy())[None, None].to(device_name)
        with torch.inference_mode():
            logits = model(tensor)
            if tuple(logits.shape) != (1, 14, 512, 512) or not bool(torch.isfinite(logits).all()):
                raise ValueError(f"Invalid PSPNet output: {image_id}")
            probs = torch.sigmoid(logits[0, [LEFT_INDEX, RIGHT_INDEX]]).cpu().numpy()
        native_left = probs[0] > CANDIDATE_THRESHOLD
        native_right = probs[1] > CANDIDATE_THRESHOLD
        geometry = shape_features(native_left, native_right)
        decision = decide_geometry(geometry)
        left = resize_binary_nearest(native_left, display.shape[::-1])
        right = resize_binary_nearest(native_right, display.shape[::-1])
        create_overlay(display, left, right, boxes_by_id[image_id], ov / f"qa_{Path(image_id).stem}.png")
        result.append({
            "image_id": image_id,
            "final_label": row["final_label"],
            "view_position": row["view_position"],
            "geometry_status": decision.status,
            "policy_for_diagnostic_only": decision.rel_policy,
            "geometry_reasons": "|".join(decision.reasons),
            "left_fraction": f"{geometry['left_fraction']:.6f}",
            "right_fraction": f"{geometry['right_fraction']:.6f}",
            "left_largest_component_share": f"{geometry['left_largest_component_share']:.6f}",
            "right_largest_component_share": f"{geometry['right_largest_component_share']:.6f}",
            "left_vertical_span": f"{geometry['left_vertical_span']:.6f}",
            "right_vertical_span": f"{geometry['right_vertical_span']:.6f}",
            "side_area_ratio": f"{geometry['side_area_ratio']:.6f}",
            "side_height_ratio": f"{geometry['side_height_ratio']:.6f}",
            "overlap_fraction": f"{geometry['overlap_fraction']:.6f}",
            "overlay": f"results/qa/overlays/policy_fresh/qa_{Path(image_id).stem}.png",
        })
        logger.info("FRESH POLICY | %d/%d | %s | %s", index, SAMPLE_SIZE, image_id, decision.status)
        if index % 6 == 0:
            print(f"P2 fresh pilot: {index}/{SAMPLE_SIZE}")
        del tensor, logits, probs
    names = make_contact_sheets(result, ov, sheets)
    csv_stage = stage / "mask_policy_fresh_pilot.csv"
    with csv_stage.open("x", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=CSV_FIELDS)
        writer.writeheader(); writer.writerows(result)
    summary = {
        "phase": "P2", "scope": "FRESH_TRAIN_36_IMAGE_DIAGNOSTIC_ONLY",
        "status": "COMPUTE_COMPLETE_NOT_ANATOMICAL_VALIDATION",
        "full_train_authorized": False,
        "seed": SEED, "image_count": len(result),
        "unique_patient_count": len({r["patient_id"] for r in chosen}),
        "strata": {f"{a} | {b}": SAMPLE_PER_STRATUM for a, b in STRATA},
        "counts": dict(sorted(Counter(r["geometry_status"] for r in result).items())),
        "policy_id": POLICY_ID, "source_sha256": sha256(Path(__file__).resolve()),
        "mask_policy_source_sha256": sha256(root / "code/mask_policy.py"),
        "prior_policy_audit_sha256": sha256(qa / "mask_policy_audit.json"),
        "qa_review_manifest_sha256": sha256(qa / "qa_review_manifest.json"),
        "split_sha256": split_hash, "checkpoint_sha256": EXPECTED_WEIGHTS_SHA256,
        "torch_version": torch.__version__, "torchxrayvision_version": importlib.metadata.version("torchxrayvision"),
        "csv_sha256": sha256(csv_stage),
        "overlays_sha256": {r["image_id"]: sha256(ov / f"qa_{Path(r['image_id']).stem}.png") for r in result},
        "sheets_sha256": {name: sha256(sheets / name) for name in names},
        "image_ids": [r["image_id"] for r in result],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Label-blind geometry screen can miss anatomical errors.",
            "No lung ground truth: this is not segmentation validation.",
            "36 quota-stratified TRAIN images do not establish full-cohort prevalence.",
            "No masks are saved for training and no full-TRAIN policy is approved.",
        ],
    }
    (stage / "mask_policy_fresh_pilot.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.rename(ov, outputs["overlays"])
    os.rename(sheets, outputs["sheets"])
    os.rename(csv_stage, outputs["csv"])
    os.rename(stage / "mask_policy_fresh_pilot.json", outputs["json"])
    stage.rmdir()
    logger.info("RUN END: COMPUTE COMPLETE | count=%d | status=%s", len(result), summary["status"])
    print("P2 fresh-TRAIN policy pilot complete; full-TRAIN policy NOT approved")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if not args.pilot:
        parser.error("Only --pilot is supported")
    logfile = PROJECT_ROOT / "results/logs/pilot_mask_policy.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("bgrel.p2.pilot_mask_policy")
    log.setLevel(logging.INFO); log.propagate = False
    handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    log.addHandler(handler)
    try:
        run(PROJECT_ROOT, args.device, log)
        return 0
    except Exception:
        log.exception("RUN END: FAIL | P2 fresh policy pilot")
        raise
    finally:
        log.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    raise SystemExit(main())
