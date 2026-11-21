"""Stateless validation, logging and schema-aware atomic CSV helpers."""
from __future__ import annotations

import json
import logging
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def write_metrics_atomic(path: Path, rows: list[dict], *, fields, replace) -> None:
    """Share CSV serialization; retain each trainer's atomic/error policy.

    Passing the original replacement callback preserves its treatment of
    interrupted .tmp files and failures. Column order and bytes stay fixed.
    """
    import csv

    def writer(temporary: Path) -> None:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            output = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            output.writeheader()
            output.writerows(rows)

    replace(path, writer)


BASELINE_CURRENT_SHA256 = {'train_baseline.py': 'ea4b348c311d7e0fa564d44ed1e78858965b1b8ce20ec998f605c5771734b34b', 'train_t2a_noaug.py': 'a39b295dd28be55379dffe2207c361cd16cac69f9ca52d94d6d69becdb1a66ff'}


def verify_current_baseline_layout(root: Path) -> dict:
    """Verify the two current diagnostic runners, not historical experiment SHA.

    Historical config/summary hashes remain subject to their original verifier.
    Passing this identity check does not certify the R3 dataset migration.
    """
    import ast
    import hashlib
    expected_runs = {"train_baseline.py": "b0_dev_s42", "train_t2a_noaug.py": "b0_t2_noaug_s42"}
    identities = {}
    for filename, expected_sha in BASELINE_CURRENT_SHA256.items():
        path = root / "code" / filename
        require(path.is_file(), f"Current baseline source missing: {filename}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha,
                f"Current baseline version drift: {filename}")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = {n.targets[0].id: ast.literal_eval(n.value)
                     for n in tree.body if isinstance(n, ast.Assign)
                     and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                     and isinstance(n.value, (ast.Constant, ast.List))}
        require(constants.get("RUN_ID") == expected_runs[filename], "Baseline run ID identity mismatch")
        require(filename in constants.get("CODE_FILES", ()), "Runner self-hash does not name its own entrypoint")
        if filename == "train_t2a_noaug.py":
            require(constants.get("AUGMENT_TRAIN") is False, "T2-A unexpectedly augments TRAIN")
        identities[filename] = {"sha256": expected_sha, "run_id": expected_runs[filename]}
    return {"current": identities, "historical_runtime_replay": "PENDING",
            "historical_hashes_rewritten": False}
