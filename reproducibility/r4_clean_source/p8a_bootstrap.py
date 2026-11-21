
"""Paired patient-cluster bootstrap for P8a validation predictions.

Read-only inputs. No DICOM access, training, TEST access or overwrite.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from dataset import sha256
from metrics import binary_metrics

BASE = ROOT / "results" / "evaluation" / "p8a_val"
RUNS = ("b0_dev_s42", "a3_l100_s42")
OUTPUT = BASE / "paired_patient_bootstrap.json"


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_predictions():
    manifest = json.loads(
        (BASE / "manifest.json").read_text(encoding="utf-8")
    )
    check(
        manifest.get("status") == "VAL_EVALUATION_COMPLETE"
        and manifest.get("scope") == "VALIDATION_ONLY",
        "P8a evaluation manifest invalid",
    )

    records = []
    metrics = []

    for run_id in RUNS:
        relative = f"{run_id}/predictions.csv"
        path = BASE / relative
        expected_hash = manifest["file_sha256"].get(relative)

        check(
            expected_hash and sha256(path) == expected_hash,
            f"Prediction integrity failed: {run_id}",
        )

        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        metric_path = BASE / run_id / "metrics.json"
        metric_hash = manifest["file_sha256"].get(
            f"{run_id}/metrics.json"
        )

        check(
            metric_hash and sha256(metric_path) == metric_hash,
            f"Metrics integrity failed: {run_id}",
        )

        metric = json.loads(metric_path.read_text(encoding="utf-8"))

        check(len(rows) == 2969, "Incomplete prediction cohort")
        check(metric["n_images"] == 2969, "Metrics count mismatch")
        check(
            len({r["image_id"] for r in rows}) == 2969,
            "Duplicate image ID",
        )

        records.append(rows)
        metrics.append(metric)

    b0, a3 = records

    for x, y in zip(b0, a3):
        check(
            (
                x["image_id"],
                x["patient_id"],
                x["target"],
                x["view_position"],
            ) == (
                y["image_id"],
                y["patient_id"],
                y["target"],
                y["view_position"],
            ),
            "Paired image identity mismatch",
        )

    targets = np.asarray(
        [int(row["target"]) for row in b0], dtype=np.int64
    )
    check(set(targets.tolist()) == {0, 1}, "Only one class")

    z0 = np.asarray(
        [float(row["logit"]) for row in b0], dtype=np.float64
    )
    z1 = np.asarray(
        [float(row["logit"]) for row in a3], dtype=np.float64
    )

    check(
        np.isfinite(z0).all() and np.isfinite(z1).all(),
        "Invalid logit",
    )

    for z, metric, run_id in zip(
        (z0, z1), metrics, RUNS
    ):
        auc, ap = binary_metrics(targets, z)
        check(
            math.isclose(
                auc, metric["auroc"], abs_tol=1e-10, rel_tol=0
            ),
            f"{run_id}: AUROC mismatch",
        )
        check(
            math.isclose(
                ap, metric["average_precision"],
                abs_tol=1e-10, rel_tol=0
            ),
            f"{run_id}: AP mismatch",
        )

    patients = defaultdict(list)

    for i, row in enumerate(b0):
        patients[row["patient_id"]].append(i)

    groups = [
        np.asarray(patients[pid], dtype=np.int64)
        for pid in sorted(patients)
    ]

    return targets, z0, z1, groups, manifest


def interval(values):
    a = np.asarray(values, dtype=np.float64)
    return {
        "lower_95": float(np.percentile(a, 2.5)),
        "upper_95": float(np.percentile(a, 97.5)),
    }


def run(n_bootstrap, seed):
    check(
        not OUTPUT.exists(),
        f"Refusing to overwrite existing result: {OUTPUT}",
    )

    y, z0, z1, groups, manifest = load_predictions()
    point0 = binary_metrics(y, z0)
    point1 = binary_metrics(y, z1)

    rng = np.random.default_rng(seed)
    n_patients = len(groups)

    samples = {
        "b0_auroc": [],
        "a3_auroc": [],
        "b0_ap": [],
        "a3_ap": [],
        "delta_auroc": [],
        "delta_ap": [],
    }

    invalid = 0
    attempts = 0

    while len(samples["delta_auroc"]) < n_bootstrap:
        attempts += 1
        check(
            attempts <= n_bootstrap * 2,
            "Too many invalid resamples",
        )

        chosen = rng.integers(
            low=0, high=n_patients, size=n_patients
        )

        indices = np.concatenate(
            [groups[int(i)] for i in chosen]
        )

        labels = y[indices]
        if len(np.unique(labels)) != 2:
            invalid += 1
            continue

        b0_auc, b0_ap = binary_metrics(
            labels, z0[indices]
        )
        a3_auc, a3_ap = binary_metrics(
            labels, z1[indices]
        )

        samples["b0_auroc"].append(b0_auc)
        samples["a3_auroc"].append(a3_auc)
        samples["b0_ap"].append(b0_ap)
        samples["a3_ap"].append(a3_ap)
        samples["delta_auroc"].append(a3_auc - b0_auc)
        samples["delta_ap"].append(a3_ap - b0_ap)

        done = len(samples["delta_auroc"])
        if done % 250 == 0:
            print(f"Bootstrap {done}/{n_bootstrap}", flush=True)

    result = {
        "phase": "P8a",
        "scope": "VALIDATION_ONLY",
        "status": "PAIRED_PATIENT_BOOTSTRAP_COMPLETE",
        "method": "paired_patient_cluster_percentile_bootstrap",
        "confidence_level": 0.95,
        "n_images": int(len(y)),
        "n_patients": n_patients,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "invalid_resamples": invalid,
        "auroc": {
            "b0": {
                "point": float(point0[0]),
                **interval(samples["b0_auroc"]),
            },
            "a3": {
                "point": float(point1[0]),
                **interval(samples["a3_auroc"]),
            },
            "delta_a3_minus_b0": {
                "point": float(point1[0] - point0[0]),
                **interval(samples["delta_auroc"]),
            },
        },
        "average_precision": {
            "b0": {
                "point": float(point0[1]),
                **interval(samples["b0_ap"]),
            },
            "a3": {
                "point": float(point1[1]),
                **interval(samples["a3_ap"]),
            },
            "delta_a3_minus_b0": {
                "point": float(point1[1] - point0[1]),
                **interval(samples["delta_ap"]),
            },
        },
        "input_manifest_sha256": sha256(
            BASE / "manifest.json"
        ),
        "limitations": [
            "Validation-only exploratory inference",
            "Percentile CI is not a non-inferiority test",
            "No inference about regional lesion reliance",
            "Paired seed-42 models only; no between-seed uncertainty",
        ],
        "test_pixels_opened": False,
    }

    with OUTPUT.open("x", encoding="utf-8") as f:
        json.dump(
            result, f, indent=2,
            ensure_ascii=False, allow_nan=False
        )
        f.write("\n")

    print("P8A_BOOTSTRAP_PASS")
    print(json.dumps({
        "n_patients": n_patients,
        "delta_auroc": result["auroc"]["delta_a3_minus_b0"],
        "delta_ap": result["average_precision"]["delta_a3_minus_b0"],
        "output": str(OUTPUT),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    check(args.bootstrap >= 100, "Bootstrap count too small")
    check(args.seed >= 0, "Invalid random seed")

    if args.preflight:
        y, _, _, groups, _ = load_predictions()
        print("P8A_BOOTSTRAP_PREFLIGHT_PASS")
        print("Images:", len(y))
        print("Patients:", len(groups))
        print("Output exists:", OUTPUT.exists())
    else:
        run(args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
