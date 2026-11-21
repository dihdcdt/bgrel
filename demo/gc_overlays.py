"""Visual-only box outlines for section F (Grad-CAM). Never used by any computation.

Exact reuse of sections A and D: instead of re-implementing any box geometry, the ALREADY
VERIFIED drawers bgrel_infer.draw_ground_truth (section A) and bgrel_infer.draw_regions
(section D) are rendered on an all-black sentinel canvas. Every pixel those functions draw
(outline, label badge, label text) differs from black, so the rendering yields an exact mask
plus colours of the outlines - same coordinates, same floor/ceil pixel bounds, same line
widths and labels - which is then pasted over a Grad-CAM (or plain) panel. Pixels outside
the outlines are left untouched, so the heatmap stays visible underneath.

These functions receive only boxes from the verified TRAIN catalog / frozen P3 index and an
RGB panel; they never touch the model, the input tensor or the Grad-CAM values.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import dataclasses

import numpy as np

import bgrel_infer as bi

_BLACK = np.zeros((1024, 1024), dtype=np.float32)


def gt_layer(gt_boxes) -> tuple[np.ndarray, np.ndarray]:
    """(mask, rgb) of the section-A red lesion outlines + labels."""
    rgb = bi.draw_ground_truth(_BLACK, tuple(gt_boxes))
    return rgb.any(axis=-1), rgb


def reference_layer(lesions) -> tuple[np.ndarray, np.ndarray]:
    """(mask, rgb) of ONLY the section-D blue dashed reference regions + their labels.

    draw_regions draws controls first and lesions on top. Rendering it once with and once without
    the controls and keeping the differing pixels isolates exactly the visible control drawing.
    """
    lesions = tuple(lesions)
    full = bi.draw_regions(_BLACK, lesions)
    no_ctrl = bi.draw_regions(_BLACK, tuple(dataclasses.replace(l, controls=()) for l in lesions))
    return (full != no_ctrl).any(axis=-1), full


def apply_layers(panel_rgb: np.ndarray, layers) -> np.ndarray:
    """Paste (mask, rgb) layers in order onto a copy of a 1024x1024x3 uint8 panel."""
    if panel_rgb.shape != (1024, 1024, 3) or panel_rgb.dtype != np.uint8:
        raise ValueError("expected a 1024x1024 RGB uint8 panel")
    out = panel_rgb.copy()
    for mask, rgb in layers:
        out[mask] = rgb[mask]
    return out


def grey_rgb(pixel01: np.ndarray) -> np.ndarray:
    """The plain radiograph as RGB, exactly as sections A/D/F display it."""
    g = (np.clip(pixel01, 0, 1) * 255).round().astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=-1)
