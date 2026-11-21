"""Grad-CAM for the frozen BG-REL image-level classifiers (demo-only, post-hoc visualisation).

Target layer (verified by inspecting research code/model.py, not assumed):
    model.features[18] - torchvision Conv2dNormActivation (1x1 conv 320->1280, BatchNorm, ReLU6).
It is the LAST module of `features`; MobileNetV2OS16.forward() is
    classifier(features(x).mean(dim=(2, 3))).flatten()
so its 1280x14x14 output is exactly the feature map whose global average feeds the single
Lung Opacity logit. The same layer is used for B0, B1 and A3 (identical architecture).

Target: the image-level Lung Opacity LOGIT (before sigmoid).

Note on this architecture: because the head is GAP + one linear layer, dlogit/dA_c(i,j) = w_c/196
at every position, so Grad-CAM equals the classic CAM  ReLU(sum_c w_c * A_c) / 196. The tests use
this identity as an independent check. Grad-CAM shows image-level prediction evidence only; it
is not attention, not a localisation of pathology, and not the BG-REL regional metrics.

Nothing here modifies weights, train/eval mode, preprocessing or the research code. Gradients are
taken with torch.autograd.grad w.r.t. the activation only, so no parameter .grad is accumulated.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

TARGET_LAYER = "features.18"
FEATURE_SHAPE = (1280, 14, 14)
EPS = 1e-8


class GradCAMError(RuntimeError):
    """Grad-CAM could not be computed (or produced non-finite values)."""


def target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """The final convolutional block feeding the image logit; fails loudly if the model differs."""
    layer = model.get_submodule(TARGET_LAYER)
    if layer is not model.features[-1]:
        raise GradCAMError("features.18 is not the last feature block - unexpected architecture")
    if not isinstance(layer[0], torch.nn.Conv2d) or layer[0].out_channels != FEATURE_SHAPE[0]:
        raise GradCAMError("unexpected target layer structure")
    return layer


@dataclass
class GradCAMResult:
    run_id: str
    label: str
    logit: float           # image-level logit from the same forward pass
    cam: np.ndarray        # 14x14 float32, after ReLU, in logit-contribution units (not normalised)
    raw_min: float         # min of the weighted sum BEFORE ReLU (diagnostic)
    raw_max: float         # max of the weighted sum BEFORE ReLU (diagnostic)
    device: str


def gradcam(model: torch.nn.Module, x: torch.Tensor, device: torch.device | str,
            run_id: str = "", label: str = "") -> GradCAMResult:
    """Standard Grad-CAM for one preprocessed input x of shape (3, 224, 224).

    Works even if called inside torch.no_grad()/torch.inference_mode(): both are explicitly
    disabled locally. The forward hook is always removed (try/finally), the model's eval/train
    state is checked unchanged, and no parameter gradient is accumulated.
    """
    if x.shape != (3, 224, 224) or x.dtype != torch.float32:
        raise GradCAMError(f"expected the preprocessed (3, 224, 224) float32 tensor, got {tuple(x.shape)}")
    was_training = model.training
    layer = target_layer(model)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        captured["A"] = output

    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode(False), torch.enable_grad():
            xb = x.detach().clone().unsqueeze(0).to(device)      # same tensor values, normal tensor
            xb.requires_grad_(True)       # guarantees a graph even if parameters were frozen
            logit = model(xb).flatten()
            if logit.shape != (1,):
                raise GradCAMError(f"expected one image logit, got shape {tuple(logit.shape)}")
            act = captured.get("A")
            if act is None or tuple(act.shape[1:]) != FEATURE_SHAPE:
                raise GradCAMError("target-layer activation was not captured with the expected shape")
            (grad,) = torch.autograd.grad(logit[0], act, retain_graph=False, create_graph=False)
            act, grad = act.detach().float(), grad.detach().float()
    finally:
        handle.remove()
    if model.training != was_training:
        raise GradCAMError("model train/eval state changed")
    if not (torch.isfinite(act).all() and torch.isfinite(grad).all()):
        raise GradCAMError("non-finite activations or gradients")
    weights = grad.mean(dim=(2, 3), keepdim=True)                  # [1, 1280, 1, 1]
    weighted = (weights * act).sum(dim=1)[0]                       # [14, 14], before ReLU
    cam = torch.relu(weighted)
    if not torch.isfinite(cam).all():
        raise GradCAMError("non-finite Grad-CAM map")
    return GradCAMResult(run_id, label, float(logit[0].detach()), cam.cpu().numpy().astype(np.float32),
                         float(weighted.min()), float(weighted.max()), str(torch.device(device).type))


def gradcam_all(bundle, x: torch.Tensor) -> list[GradCAMResult]:
    """Grad-CAM for B0, B1 and A3 on the same input, using the already-loaded verified models.
    Falls back to a CPU copy of a model on CUDA out-of-memory (FP32), like predict()."""
    from bgrel_infer import MODEL_INFO
    out = []
    with bundle._lock:                      # serialise GPU use with predict()/region_evidence()
        for run_id, m in bundle.models.items():
            try:
                out.append(gradcam(m, x, bundle.device, run_id, MODEL_INFO[run_id][0]))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                out.append(gradcam(bundle._cpu_model(run_id), x, "cpu", run_id, MODEL_INFO[run_id][0]))
    return out


# ---------------------------------------------------------------------------
# Upsampling, normalisation and overlay (no extra dependencies)
# ---------------------------------------------------------------------------
def upsample(cam: np.ndarray, size: tuple[int, int] = (1024, 1024)) -> np.ndarray:
    """Bilinear (align_corners=False: 14x14 cell centres map onto the image grid). Non-negative
    input stays non-negative."""
    t = torch.from_numpy(np.ascontiguousarray(cam, dtype=np.float32))[None, None]
    up = F.interpolate(t, size=size, mode="bilinear", align_corners=False)[0, 0].numpy()
    return np.clip(up, 0.0, None).astype(np.float32)


def normalise(cam: np.ndarray, vmax: float | None = None) -> tuple[np.ndarray, bool]:
    """Scale to [0, 1] by `vmax` (default: the map's own max). Returns (map, is_zero).
    A zero / near-zero map (vmax <= 1e-8) returns all zeros instead of dividing by ~0."""
    cam = np.nan_to_num(np.asarray(cam, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    cam = np.clip(cam, 0.0, None)
    scale = float(cam.max()) if vmax is None else float(vmax)
    if not np.isfinite(scale) or scale <= EPS:
        return np.zeros_like(cam, dtype=np.float32), True
    return np.clip(cam / scale, 0.0, 1.0).astype(np.float32), bool(cam.max() <= EPS)


# "inferno"-style sequential map (dark -> bright), interpolated from fixed anchor colours.
_ANCHORS = np.array([(0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99), (212, 72, 66),
                     (245, 125, 21), (250, 193, 39), (252, 255, 164)], dtype=np.float32)
_LUT = np.stack([np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(_ANCHORS)), _ANCHORS[:, c])
                 for c in range(3)], axis=1).astype(np.uint8)


def colormap_stops(n: int = 8) -> list[str]:
    return ["#%02x%02x%02x" % tuple(_LUT[int(round(i * 255 / (n - 1)))]) for i in range(n)]


def overlay(pixel01: np.ndarray, heat01: np.ndarray, opacity: float = 0.55) -> np.ndarray:
    """Transparent overlay: per-pixel alpha = opacity * heat, so zero-evidence areas show the
    radiograph unchanged. Returns uint8 RGB with the shape of pixel01."""
    if pixel01.shape != heat01.shape:
        raise ValueError("heatmap and image sizes differ")
    grey = np.clip(pixel01, 0, 1)[..., None] * 255.0
    colour = _LUT[np.clip(np.round(heat01 * 255), 0, 255).astype(np.uint8)].astype(np.float32)
    alpha = (np.clip(opacity, 0, 1) * np.clip(heat01, 0, 1))[..., None]
    return np.clip(grey * (1 - alpha) + colour * alpha, 0, 255).round().astype(np.uint8)
