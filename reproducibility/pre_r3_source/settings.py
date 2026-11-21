"""Centralized project paths for BG-REL.

Paths are resolved from this file, not from the terminal working directory.
The legacy raw-data layout is supported only because its existence was verified
in the user's P0 pre-flight ZIP. Data is never moved by this module.
"""

from dataclasses import dataclass
from pathlib import Path

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


def _choose_existing(description: str, candidates: tuple[Path, ...]) -> Path:
    existing = [path for path in candidates if path.exists()]
    if len(existing) == 1:
        return existing[0]
    if not existing:
        raise FileNotFoundError(
            f"{description} not found. Checked: " + ", ".join(str(p) for p in candidates)
        )
    raise RuntimeError(
        f"Ambiguous {description}; both old and new locations exist: "
        + ", ".join(str(p) for p in existing)
        + ". Keep only one active source or specify the path explicitly."
    )


def get_audit_paths(
    dicom_root: Path | None = None,
    mappings: Path | None = None,
    annotations: Path | None = None,
) -> AuditPaths:
    """Choose the single verified data source; never silently prefer a duplicate."""
    root = PROJECT_ROOT
    return AuditPaths(
        root=root,
        dicom_root=Path(dicom_root).resolve() if dicom_root else _choose_existing(
            "DICOM root",
            (root / "data/raw/RSNA Pneumonia/raw", root / "RSNA Pneumonia/raw"),
        ),
        mappings=Path(mappings).resolve() if mappings else _choose_existing(
            "mappings JSON",
            (root / "data/raw/mappings.json", root / "pneumonia-challenge-dataset-mappings_2018.json"),
        ),
        annotations=Path(annotations).resolve() if annotations else _choose_existing(
            "annotations JSON",
            (root / "data/raw/annotations.json", root / "pneumonia-challenge-annotations-adjudicated-kaggle_2018.json"),
        ),
        processed=root / "data/processed",
        qa=root / "results/qa",
        logs=root / "results/logs",
    )
