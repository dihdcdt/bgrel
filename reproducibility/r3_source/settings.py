"""Centralized project paths for BG-REL.

Paths are resolved from this file, not from the terminal working directory.
The verified R3 cutover layout is explicit. Data is never moved by this module.
"""

from dataclasses import dataclass
from pathlib import Path

from paths import DICOM_ROOT, MAPPINGS_JSON, ANNOTATIONS_JSON

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AuditPaths:
    root: Path
    dicom_root: Path
    mappings: Path
    annotations: Path
    processed: Path
    qa: Path
    logs: Path


def get_audit_paths(
    dicom_root: Path | None = None,
    mappings: Path | None = None,
    annotations: Path | None = None,
) -> AuditPaths:
    """Choose the single verified data source; never silently prefer a duplicate."""
    root = PROJECT_ROOT
    return AuditPaths(
        root=root,
        dicom_root=Path(dicom_root).resolve() if dicom_root else DICOM_ROOT,
        mappings=Path(mappings).resolve() if mappings else MAPPINGS_JSON,
        annotations=Path(annotations).resolve() if annotations else ANNOTATIONS_JSON,
        processed=root / "data/processed",
        qa=root / "results/qa",
        logs=root / "results/logs",
    )
