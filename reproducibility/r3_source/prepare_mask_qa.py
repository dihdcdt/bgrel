"""Freeze a stratified, patient-unique TRAIN-only 200-image PSPNet QA sample.

This step selects images and verifies provenance only. It never loads DICOMs,
instantiates PSPNet, writes overlays, or creates lung masks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from generate_lung_masks import read_csv, sha256, verified_split
from settings import PROJECT_ROOT


SEED = 42
SAMPLE_SIZE = 200
TARGETS = ("Lung Opacity", "Normal", "No Lung Opacity / Not Normal")
VIEWS = ("AP", "PA")
QUOTAS = {
    ("Lung Opacity", "AP"): 34,
    ("Lung Opacity", "PA"): 34,
    ("Normal", "AP"): 33,
    ("Normal", "PA"): 33,
    ("No Lung Opacity / Not Normal", "AP"): 33,
    ("No Lung Opacity / Not Normal", "PA"): 33,
}
EXPECTED_MANIFEST_SHA = "4c690ddf7918d0890b6715dba290be85a16959adc0dcaa70b5b4c5bd387d6c9b"
EXPECTED_BOXES_SHA = "6372a1facc3e1b78fbcd97a5bd103d2fc33ed257c92f8076b1d36e9168078e45"
CSV_FIELDS = (
    "image_id", "patient_id_sha256", "final_label", "view_position", "num_boxes",
    "box_count_group", "lesion_size_quartile", "review_status", "review_reason",
)


def validate_metadata(
    manifest: list[dict[str, str]], split: list[dict[str, str]],
    boxes: list[dict[str, str]], excluded_smoke_ids: set[str],
) -> list[dict[str, object]]:
    """Validate frozen dataset lineage and build candidate metadata from TRAIN."""
    split_by_id = {row["image_id"]: row for row in split}
    manifest_by_id = {row["image_id"]: row for row in manifest}
    if len(split_by_id) != len(split) or len(manifest_by_id) != len(manifest):
        raise ValueError("Duplicate image IDs in manifest or split")
    if set(split_by_id) != set(manifest_by_id):
        raise ValueError("Split and manifest image ID sets differ")
    if not excluded_smoke_ids.issubset(manifest_by_id):
        raise ValueError("Smoke references unknown image IDs")
    smoke_patients = {manifest_by_id[image_id]["patient_id"] for image_id in excluded_smoke_ids}
    if any(split_by_id[image_id]["split"] != "train" for image_id in excluded_smoke_ids):
        raise ValueError("Smoke contains non-TRAIN images")

    counts: Counter[str] = Counter()
    max_areas: dict[str, float] = defaultdict(float)
    for box in boxes:
        image_id = box["image_id"]
        if image_id not in manifest_by_id:
            raise ValueError("Box references image missing from manifest")
        width = float(box["image_width"])
        height = float(box["image_height"])
        w = float(box["width"])
        h = float(box["height"])
        if not (width > 0 and height > 0 and w > 0 and h > 0):
            raise ValueError("Invalid box size")
        counts[image_id] += 1
        max_areas[image_id] = max(max_areas[image_id], w * h / (width * height))

    candidates: list[dict[str, object]] = []
    positive_all: list[tuple[float, str]] = []
    for row in manifest:
        image_id = row["image_id"]
        srow = split_by_id[image_id]
        if row["patient_id"] != srow["patient_id"]:
            raise ValueError("Manifest/split patient mismatch")
        label = row["final_label"]
        view = row["view_position"]
        if label not in TARGETS or view not in VIEWS:
            raise ValueError("Unexpected label or view")
        num = counts[image_id]
        if num != int(row["num_boxes"]):
            raise ValueError("Box counts disagree with manifest")
        if (label == "Lung Opacity") != (num > 0):
            raise ValueError("Final label and expert box availability disagree")
        if srow["split"] != "train":
            continue
        if label == "Lung Opacity":
            positive_all.append((max_areas[image_id], image_id))
        if image_id in excluded_smoke_ids or row["patient_id"] in smoke_patients:
            continue
        candidates.append({
            "image_id": image_id,
            "patient_id": row["patient_id"],
            "final_label": label,
            "view_position": view,
            "num_boxes": num,
            "box_count_group": "MULTI" if num >= 2 else ("SINGLE" if num == 1 else "NONE"),
            "lesion_size_quartile": "",
        })

    # Deterministic rank-based quartiles on *all* positive TRAIN images.
    positive_all.sort(key=lambda item: (item[0], item[1]))
    if len(positive_all) < 4:
        raise ValueError("Too few positive TRAIN images for lesion-area quartiles")
    quartiles = {
        image_id: f"Q{min(4, 1 + 4 * index // len(positive_all))}"
        for index, (_, image_id) in enumerate(positive_all)
    }
    for row in candidates:
        if row["final_label"] == "Lung Opacity":
            row["lesion_size_quartile"] = quartiles[str(row["image_id"])]
    return candidates


def select_sample(candidates: list[dict[str, object]], quotas: dict[tuple[str, str], int], seed: int) -> list[dict[str, object]]:
    """Select without replacement or patient duplication, balancing positive subgroups."""
    rng = random.Random(seed)
    seen_patients: set[str] = set()
    selected: list[dict[str, object]] = []
    by_stratum: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        by_stratum[(str(row["final_label"]), str(row["view_position"]))].append(row)

    # Process scarce strata first to protect their chance of reaching quota.
    strata = sorted(quotas, key=lambda key: (len(by_stratum.get(key, [])), key))
    for key in strata:
        quota = quotas[key]
        if quota < 0:
            raise ValueError("Negative quota")
        pool = sorted(by_stratum.get(key, []), key=lambda row: str(row["image_id"]))
        rng.shuffle(pool)
        if key[0] == "Lung Opacity":
            buckets: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
            for row in pool:
                buckets[(str(row["box_count_group"]), str(row["lesion_size_quartile"]))].append(row)
            bucket_keys = [(group, f"Q{q}") for group in ("SINGLE", "MULTI") for q in range(1, 5)]
            if quota >= len(bucket_keys) and any(not buckets[k] for k in bucket_keys):
                raise ValueError(f"Insufficient positive subgroup coverage for {key}")
            picked_for_stratum: list[dict[str, object]] = []
            # Round-robin balances single/multi and lesion-size quartiles.
            while len(picked_for_stratum) < quota:
                progressed = False
                for subgroup in bucket_keys:
                    queue = buckets[subgroup]
                    while queue and str(queue[-1]["patient_id"]) in seen_patients:
                        queue.pop()
                    if not queue:
                        continue
                    row = queue.pop()
                    selected.append(row)
                    picked_for_stratum.append(row)
                    seen_patients.add(str(row["patient_id"]))
                    progressed = True
                    if len(picked_for_stratum) == quota:
                        break
                if not progressed:
                    raise ValueError(f"Insufficient patient-distinct images for {key}")
        else:
            picked = 0
            for row in pool:
                patient = str(row["patient_id"])
                if patient in seen_patients:
                    continue
                selected.append(row)
                seen_patients.add(patient)
                picked += 1
                if picked == quota:
                    break
            if picked != quota:
                raise ValueError(f"Insufficient patient-distinct images for {key}")
    if len(selected) != sum(quotas.values()):
        raise AssertionError("Wrong total sample count")
    if len({row["image_id"] for row in selected}) != len(selected):
        raise AssertionError("Duplicate image IDs selected")
    return sorted(selected, key=lambda row: str(row["image_id"]))


def build_audit(selected: list[dict[str, object]], excluded_count: int) -> dict[str, object]:
    """Report actual selection composition without implying anatomical QA approval."""
    strata = Counter((str(row["final_label"]), str(row["view_position"])) for row in selected)
    positive_subgroups = Counter(
        (str(row["view_position"]), str(row["box_count_group"]), str(row["lesion_size_quartile"]))
        for row in selected if row["final_label"] == "Lung Opacity"
    )
    return {
        "phase": "P2",
        "scope": "TRAIN_QA_SELECTION_ONLY",
        "status": "SELECTION_REVIEW_REQUIRED",
        "seed": SEED,
        "sample_size": len(selected),
        "unique_patient_count": len({row["patient_id"] for row in selected}),
        "excluded_smoke_image_count": excluded_count,
        "quartile_definition": "rank of largest annotated bbox area / image area among all positive TRAIN images",
        "strata": [{"label": label, "view": view, "n": strata[(label, view)]} for label in TARGETS for view in VIEWS],
        "positive_subgroups": [
            {"view": view, "box_count_group": group, "size_quartile": f"Q{q}",
             "n": positive_subgroups[(view, group, f"Q{q}")]}
            for view in VIEWS for group in ("SINGLE", "MULTI") for q in range(1, 5)
        ],
        "notes": [
            "The sample is deliberately stratified/enriched, not population representative.",
            "No DICOM was read; no masks or overlays were generated.",
            "No image has an anatomical QA verdict at selection stage.",
        ],
    }


def run(root: Path, logger: logging.Logger) -> None:
    """Read frozen metadata and create exactly two new reviewed-selection artifacts."""
    qa_dir = root / "results/qa"
    selection_file = qa_dir / "mask_qa_selection.csv"
    audit_file = qa_dir / "mask_qa_selection.json"
    if selection_file.exists() or audit_file.exists():
        raise FileExistsError("QA selection already exists. Do not overwrite")

    manifest_path = root / "data/processed/master_manifest.csv"
    boxes_path = root / "data/processed/boxes.csv"
    smoke_path = qa_dir / "pspnet_smoke.json"
    if sha256(manifest_path) != EXPECTED_MANIFEST_SHA:
        raise ValueError("Unexpected P0 master manifest hash")
    if sha256(boxes_path) != EXPECTED_BOXES_SHA:
        raise ValueError("Unexpected P0 expert boxes hash")
    split, split_hash = verified_split(root)
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("scope") != "TRAIN_ONLY_SMOKE" or smoke.get("split_sha256") != split_hash:
        raise ValueError("Prior smoke report disagrees with frozen split")
    excluded = {row["image_id"] for row in smoke.get("images", [])}
    if len(excluded) != 8:
        raise ValueError("Expected the eight-image smoke challenge set")

    manifest = read_csv(manifest_path)
    boxes = read_csv(boxes_path)
    candidates = validate_metadata(manifest, split, boxes, excluded)
    selected = select_sample(candidates, QUOTAS, SEED)
    if len(selected) != SAMPLE_SIZE:
        raise AssertionError("QA cohort must have 200 images")
    audit = build_audit(selected, len(excluded))
    if any(row["n"] != QUOTAS[(row["label"], row["view"])] for row in audit["strata"]):
        raise AssertionError("QA quotas are not met")
    audit.update({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": split_hash,
        "manifest_sha256": EXPECTED_MANIFEST_SHA,
        "boxes_sha256": EXPECTED_BOXES_SHA,
        "smoke_json_sha256": sha256(smoke_path),
        "source_sha256": sha256(Path(__file__)),
    })
    qa_dir.mkdir(parents=True, exist_ok=True)
    with selection_file.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in selected:
            writer.writerow({
                "image_id": row["image_id"],
                "patient_id_sha256": hashlib.sha256(str(row["patient_id"]).encode()).hexdigest(),
                "final_label": row["final_label"],
                "view_position": row["view_position"],
                "num_boxes": row["num_boxes"],
                "box_count_group": row["box_count_group"],
                "lesion_size_quartile": row["lesion_size_quartile"],
                "review_status": "UNREVIEWED",
                "review_reason": "",
            })
    audit["selection_csv_sha256"] = sha256(selection_file)
    audit_file.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", errors="strict")
    logger.info("P2 QA SELECT | count=%d | patients=%d | sha256=%s", SAMPLE_SIZE,
                audit["unique_patient_count"], audit["selection_csv_sha256"])
    print("P2 QA SELECTION COMPLETE; REVIEW REQUIRED")
    print(f"Selection: {selection_file.relative_to(root).as_posix()}")
    print(f"Audit: {audit_file.relative_to(root).as_posix()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if not args.prepare:
        parser.error("Only --prepare is supported")
    log_path = PROJECT_ROOT / "results/logs/prepare_mask_qa.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bgrel.p2.qa.selection")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    try:
        logger.info("RUN START | P2 QA SELECTION | seed=%d | requested=%d", SEED, SAMPLE_SIZE)
        run(PROJECT_ROOT, logger)
        logger.info("RUN END: PASS | SELECTION_ONLY | REVIEW_REQUIRED")
        return 0
    except Exception:
        logger.exception("RUN END: FAIL | P2 QA SELECTION")
        raise
    finally:
        for active in list(logger.handlers):
            active.close()
            logger.removeHandler(active)


if __name__ == "__main__":
    raise SystemExit(main())
