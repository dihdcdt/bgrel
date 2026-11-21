"""Shared stateless helpers extracted from the frozen R1 candidate source.

R2 staging only. This module MUST NOT be installed into the live BG-REL
code directory until historical source provenance and path migration gates pass.
"""
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
