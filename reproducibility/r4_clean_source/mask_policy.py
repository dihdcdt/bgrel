"""Audit a label-blind P2 mask screening policy against frozen TRAIN QA evidence.

The existing QA200 results are read-only. This is NOT a full-TRAIN mask
producer and does not claim anatomical correctness or validated sensitivity.
The geometric decision is independent of class labels and AP/PA; pre-existing
visual warnings are a separate override only for images already reviewed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from auto_mask_qa import preliminary_decision
from generate_lung_masks import read_csv, sha256
from pilot_lung_masks import verified_evidence
from paths import PROJECT_ROOT

POLICY_ID = "geometry_v1_unchanged_thresholds"
GEOMETRY_FIELDS = (
    "left_fraction", "right_fraction", "left_largest_component_share",
    "right_largest_component_share", "left_vertical_span", "right_vertical_span",
    "left_centroid_x", "right_centroid_x", "side_area_ratio",
    "side_height_ratio", "overlap_fraction",
)
AUDIT_FIELDS = (
    "image_id", "final_label", "view_position", "visual_review",
    "old_auto_status", "geometry_status", "geometry_reasons", "geometry_policy",
    "reviewed_policy", "review_override_reason", "screening_disagreement",
)
VISUAL_STATES = frozenset({"PASS", "WARN", "FAIL"})
STATUS = "POLICY_AUDIT_COMPLETE_NOT_APPROVED_FOR_FULL_TRAIN"


@dataclass(frozen=True)
class Decision:
    status: str
    rel_policy: str
    reasons: tuple[str, ...]


def decide_geometry(features: dict[str, float]) -> Decision:
    """Screen mask geometry without any class, view or reviewer information.

    The thresholds are owned by the already-run P2 auto_mask_qa module.
    No refit to labels, image IDs, or QA visual outcomes takes place here.
    """
    if set(features) != set(GEOMETRY_FIELDS):
        raise ValueError("Mask features must contain exactly the known geometry fields")
    for name, value in features.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"Non-numeric feature: {name}")
        if not math.isfinite(float(value)):
            if name in ("left_centroid_x", "right_centroid_x") and math.isnan(float(value)):
                side = name.split("_")[0]
                if float(features[side + "_fraction"]) == 0.0:
                    continue
            raise ValueError(f"Non-finite or inconsistent geometry feature: {name}")
    state, reasons = preliminary_decision(features)
    if state not in {"AUTO_SCREENED", "AUTO_FLAG", "AUTO_REJECT"}:
        raise ValueError("Unexpected geometry status")
    return Decision(state,
                    "REL_CANDIDATE_ONLY" if state == "AUTO_SCREENED" else "SKIP_REL",
                    tuple(reasons))


def apply_known_review(geometry: Decision, review: str | None) -> tuple[str, str]:
    """Block explicitly flagged QA200 masks; absence of review is not a PASS.

    This function is a *safeguard on previously reviewed samples*, not a
    substitute for checking unreviewed mask anatomy or future P3 controls.
    """
    if review is None:
        return geometry.rel_policy, "NOT_VISUALLY_REVIEWED"
    if review not in VISUAL_STATES:
        raise ValueError(f"Unexpected visual review: {review}")
    if review != "PASS":
        return "SKIP_REL", "VISUAL_" + review
    return geometry.rel_policy, ""


def parse_features(row: dict[str, str]) -> dict[str, float]:
    try:
        return {name: float(row[name]) for name in GEOMETRY_FIELDS}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Missing or invalid QA geometry feature") from exc


def audit(root: Path) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Perform read-only provenance validation and compare geometric decisions.

    Labels and projection are copied to the QA output only for subgroup analysis.
    No label or view is passed into the decision function.
    """
    reviewed, _, _, auto_audit = verified_evidence(root)
    qa_dir = root / "results/qa"
    auto_rows = read_csv(qa_dir / "qa_auto_assessment.csv")
    auto_by_id = {row["image_id"]: row for row in auto_rows}
    if len(auto_rows) != 200 or len(auto_by_id) != 200:
        raise ValueError("Automated QA records are not unique and complete")
    if len(reviewed) != 200 or {r["image_id"] for r in reviewed} != set(auto_by_id):
        raise ValueError("QA review and automated screening do not match")

    pilot_file = qa_dir / "pilot_mask_manifest.json"
    pilot = json.loads(pilot_file.read_text(encoding="utf-8"))
    if (pilot.get("status") != "PILOT_COMPUTE_COMPLETE_REVIEW_REQUIRED" or
            pilot.get("scope") != "TRAIN_QA200_20_IMAGE_PILOT_ONLY" or
            pilot.get("pilot_csv_sha256") != sha256(qa_dir / "pilot_mask_audit.csv") or
            pilot.get("pilot_source_sha256") != sha256(root / "code/pilot_lung_masks.py") or
            pilot.get("split_sha256") != auto_audit["split_sha256"]):
        raise ValueError("P2 pilot lineage invalid; stop")
    rows: list[dict[str, str]] = []
    for source in sorted(reviewed, key=lambda item: item["image_id"]):
        image_id = source["image_id"]
        original = auto_by_id[image_id]
        if source["review_status"] not in VISUAL_STATES:
            raise ValueError("Missing or unexpected visual evidence")
        geom = decide_geometry(parse_features(original))
        reviewed_policy, override_reason = apply_known_review(geom, source["review_status"])
        rows.append({
            "image_id": image_id,
            "final_label": source["final_label"],
            "view_position": source["view_position"],
            "visual_review": source["review_status"],
            "old_auto_status": original["auto_status"],
            "geometry_status": geom.status,
            "geometry_reasons": "|".join(geom.reasons),
            "geometry_policy": geom.rel_policy,
            "reviewed_policy": reviewed_policy,
            "review_override_reason": override_reason,
            "screening_disagreement": "1" if (geom.status != original["auto_status"]) else "0",
        })
    misses = [r["image_id"] for r in rows if r["visual_review"] == "FAIL" and
              r["geometry_policy"] == "REL_CANDIDATE_ONLY"]
    review_nonpass = [r for r in rows if r["visual_review"] != "PASS" and
                      r["geometry_policy"] == "REL_CANDIDATE_ONLY"]
    summary: dict[str, object] = {
        "phase": "P2", "scope": "TRAIN_QA200_READ_ONLY_POLICY_AUDIT",
        "status": STATUS, "policy_id": POLICY_ID,
        "source_sha256": sha256(Path(__file__).resolve()),
        "split_sha256": auto_audit["split_sha256"],
        "checkpoint_sha256": auto_audit["checkpoint_sha256"],
        "review_csv_sha256": sha256(qa_dir / "qa_review_completed.csv"),
        "auto_csv_sha256": sha256(qa_dir / "qa_auto_assessment.csv"),
        "pilot_manifest_sha256": sha256(pilot_file),
        "qa200_manifest_sha256": sha256(qa_dir / "pspnet_manifest.json"),
        "records": len(rows),
        "geometry_status_counts": dict(sorted(Counter(r["geometry_status"] for r in rows).items())),
        "geometry_policy_counts": dict(sorted(Counter(r["geometry_policy"] for r in rows).items())),
        "reviewed_policy_counts": dict(sorted(Counter(r["reviewed_policy"] for r in rows).items())),
        "visual_status_counts": dict(sorted(Counter(r["visual_review"] for r in rows).items())),
        "geometry_screened_but_visual_not_pass_count": len(review_nonpass),
        "geometry_screened_but_visual_fail_ids": misses,
        "geometry_vs_old_auto_disagreement_ids": [r["image_id"] for r in rows if r["screening_disagreement"] == "1"],
        "model_changes": "NONE; thresholds reused from existing preliminary_decision",
        "limitations": [
            "Contact-sheet review is qualitative AI visual screening, not pixel-wise ground truth.",
            "This is a quota-stratified QA sample, not an unbiased estimate of all TRAIN images.",
            "Unreviewed images have no visual PASS label; geometry-only status remains provisional.",
            "P3 must check each lesion/control candidate for geometry, lung occupancy and all boxes.",
            "Do not enable full-TRAIN generation solely on this policy audit.",
        ],
    }
    return rows, summary


def write_audit(root: Path, records: list[dict[str, str]], summary: dict[str, object]) -> None:
    """Write new audit atomically; never modify or overwrite frozen QA files."""
    qa_dir = root / "results/qa"
    csv_out = qa_dir / "mask_policy_audit.csv"
    json_out = qa_dir / "mask_policy_audit.json"
    stage = qa_dir / ".mask_policy_stage"
    for p in (csv_out, json_out, stage):
        if p.exists():
            raise FileExistsError(f"Existing mask policy output: {p}")
    stage.mkdir(parents=True, exist_ok=False)
    csv_stage = stage / csv_out.name
    json_stage = stage / json_out.name
    with csv_stage.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    summary["audit_csv_sha256"] = sha256(csv_stage)
    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    with json_stage.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.rename(csv_stage, csv_out)
    os.rename(json_stage, json_out)
    stage.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-qa200", action="store_true")
    args = parser.parse_args()
    if not args.audit_qa200:
        parser.error("Only --audit-qa200 is supported")
    logfile = PROJECT_ROOT / "results/logs/mask_policy.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("bgrel.p2.mask_policy")
    log.setLevel(logging.INFO)
    log.propagate = False
    handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    log.addHandler(handler)
    try:
        log.info("RUN START | P2 label-blind geometry policy audit | policy=%s", POLICY_ID)
        records, summary = audit(PROJECT_ROOT)
        write_audit(PROJECT_ROOT, records, summary)
        log.info("RUN END: AUDIT COMPLETE NOT FULL-TRAIN PASS | records=%d missed_visual_fail=%d nonpass_screened=%d",
                 len(records), len(summary["geometry_screened_but_visual_fail_ids"]),
                 summary["geometry_screened_but_visual_not_pass_count"])
        print("P2 POLICY AUDIT COMPLETE; FULL TRAIN NOT AUTHORIZED")
        print("Audit: results/qa/mask_policy_audit.json")
        return 0
    except Exception:
        log.exception("RUN END: FAIL | P2 mask policy audit")
        raise
    finally:
        log.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    raise SystemExit(main())
