"""Tests for the demo-only Grad-CAM module (TRAIN images only; no VAL/TEST access)."""
import copy
import hashlib
import io
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import bgrel_infer as bi  # noqa: E402
import gradcam as gc  # noqa: E402
from test_regression import SCORE_TOL, SMOKE_REFERENCE  # noqa: E402

_B = None


def bundle():
    global _B
    if _B is None:
        _B = bi.ModelBundle()
    return _B


def state_hash(model) -> str:
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def x_for(image_id):
    return bi.preprocess(bi.train_catalog().load(image_id).pixel01)


class GradCAMCore(unittest.TestCase):
    def test_target_layer_is_final_feature_map_feeding_the_logit(self):
        b = bundle()
        x = x_for("00000032_007.png")
        for run_id, m in b.models.items():
            layer = gc.target_layer(m)
            self.assertIs(layer, m.features[-1])
            self.assertIs(layer, m.get_submodule("features.18"))
            seen = {}
            h = layer.register_forward_hook(lambda mod, i, o: seen.__setitem__("A", o))
            try:
                with torch.inference_mode():
                    logit = m(x[None].to(b.device))
            finally:
                h.remove()
            self.assertEqual(tuple(seen["A"].shape), (1, 1280, 14, 14))
            # forward() == classifier(GAP(features.18 output)): the map feeds the logit directly.
            with torch.inference_mode():
                again = m.classifier(seen["A"].mean(dim=(2, 3))).flatten()
            self.assertAlmostEqual(float(logit), float(again), delta=1e-6)

    def test_all_models_shapes_finite_and_logits_unchanged(self):
        b = bundle()
        for image_id, (_, expected) in SMOKE_REFERENCE.items():
            x = x_for(image_id)
            res = gc.gradcam_all(b, x)
            self.assertEqual([r.run_id for r in res], list(bi.RUN_IDS))
            preds = {p.run_id: p for p in b.predict(x)}
            for r in res:
                with self.subTest(image_id=image_id, run=r.run_id):
                    self.assertEqual(r.cam.shape, (14, 14))
                    self.assertTrue(np.isfinite(r.cam).all())
                    self.assertGreaterEqual(float(r.cam.min()), 0.0)        # after ReLU
                    self.assertAlmostEqual(r.logit, preds[r.run_id].logit, delta=1e-4)
                    self.assertLessEqual(abs(preds[r.run_id].score - expected[r.run_id][1]), SCORE_TOL)
                    up = gc.upsample(r.cam, (1024, 1024))
                    self.assertEqual(up.shape, (1024, 1024))
                    self.assertTrue(np.isfinite(up).all())
                    for vmax in (None, max(float(q.cam.max()) for q in res)):
                        heat, _ = gc.normalise(up, vmax)
                        self.assertEqual(heat.shape, (1024, 1024))
                        self.assertGreaterEqual(float(heat.min()), 0.0)
                        self.assertLessEqual(float(heat.max()), 1.0)

    def test_equals_independent_cam_and_sums_to_logit(self):
        """GAP + linear head: Grad-CAM == ReLU(sum_c w_c A_c)/196, and the pre-ReLU map sums to
        logit - bias. Computed independently from the activation and the classifier weights."""
        b = bundle()
        x = x_for("00000032_007.png")
        for run_id, m in b.models.items():
            r = gc.gradcam(m, x, b.device, run_id)
            seen = {}
            h = m.features[18].register_forward_hook(lambda mod, i, o: seen.__setitem__("A", o))
            try:
                with torch.inference_mode():
                    m(x[None].to(b.device))
            finally:
                h.remove()
            A = seen["A"][0].float().cpu()
            w = m.classifier.weight.detach().float().cpu()[0]
            bias = float(m.classifier.bias.detach().cpu()[0])
            weighted = torch.einsum("c,chw->hw", w, A) / 196.0
            np.testing.assert_allclose(r.cam, torch.relu(weighted).numpy(), rtol=1e-4, atol=1e-6)
            self.assertAlmostEqual(float(weighted.sum()) + bias, r.logit, delta=1e-3)
            self.assertAlmostEqual(float(weighted.min()), r.raw_min, delta=1e-5)

    def test_hooks_removed_params_unchanged_eval_kept(self):
        b = bundle()
        x = x_for("00000003_003.png")
        before = {k: state_hash(m) for k, m in b.models.items()}
        gc.gradcam_all(b, x)
        for run_id, m in b.models.items():
            layer = gc.target_layer(m)
            self.assertEqual(len(layer._forward_hooks), 0)
            self.assertEqual(len(layer._backward_hooks), 0)
            self.assertFalse(m.training)
            self.assertTrue(all(p.grad is None for p in m.parameters()))
            self.assertEqual(state_hash(m), before[run_id])

    def test_hook_removed_on_exception(self):
        b = bundle()
        m = b.models["a3_l100_s42"]
        bad = torch.zeros(3, 224, 224, dtype=torch.float32)
        orig_forward = m.forward
        m.forward = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))   # fail mid-call
        try:
            with self.assertRaises(RuntimeError):
                gc.gradcam(m, bad, b.device, "a3_l100_s42")
        finally:
            del m.forward
        self.assertIs(m.forward.__func__, orig_forward.__func__)
        self.assertEqual(len(gc.target_layer(m)._forward_hooks), 0)
        with self.assertRaises(gc.GradCAMError):                           # wrong input shape
            gc.gradcam(m, torch.zeros(1, 224, 224), b.device)
        self.assertEqual(len(gc.target_layer(m)._forward_hooks), 0)

    def test_works_inside_no_grad_and_inference_mode_and_on_cpu(self):
        b = bundle()
        x = x_for("00000032_007.png")
        ref = gc.gradcam(b.models["b0_dev_s42"], x, b.device)
        for ctx in (torch.no_grad, torch.inference_mode):
            with self.subTest(ctx=ctx.__name__), ctx():
                r = gc.gradcam(b.models["b0_dev_s42"], x, b.device)
            np.testing.assert_allclose(r.cam, ref.cam, rtol=1e-5, atol=1e-7)
        cpu_model = copy.deepcopy(b.models["b0_dev_s42"]).to("cpu").eval()
        r_cpu = gc.gradcam(cpu_model, x, "cpu")
        self.assertEqual(r_cpu.device, "cpu")
        np.testing.assert_allclose(r_cpu.cam, ref.cam, rtol=0, atol=5e-4)     # TF32 on GPU vs CPU

    def test_zero_and_near_zero_heatmaps(self):
        for arr in (np.zeros((14, 14), np.float32), np.full((14, 14), 1e-12, np.float32),
                    np.full((14, 14), np.nan, np.float32), -np.ones((14, 14), np.float32)):
            heat, is_zero = gc.normalise(gc.upsample(np.nan_to_num(arr, nan=0.0).clip(0)), None)
            self.assertTrue(is_zero)
            self.assertTrue(np.isfinite(heat).all())
            self.assertEqual(float(heat.max()), 0.0)
            heat2, is_zero2 = gc.normalise(arr, 1.0)                        # common-scale path
            self.assertTrue(np.isfinite(heat2).all())
            self.assertGreaterEqual(float(heat2.min()), 0.0)
        heat, is_zero = gc.normalise(np.ones((4, 4), np.float32), 0.0)    # vmax 0 -> no division
        self.assertTrue(is_zero and float(heat.max()) == 0.0)

    def test_overlay_keeps_anatomy_where_heat_is_zero(self):
        pix = bi.train_catalog().load("00000003_000.png").pixel01
        heat = np.zeros_like(pix); heat[100:200, 300:400] = 1.0
        ov = gc.overlay(pix, heat, 0.55)
        grey = (pix * 255).round().astype(np.uint8)
        self.assertEqual(ov.shape, (1024, 1024, 3))
        self.assertTrue((ov[heat == 0] == grey[heat == 0][:, None]).all())   # untouched
        self.assertFalse((ov[150, 350] == grey[150, 350]).all())             # coloured

    def test_raster_input_works(self):
        from test_raster import png_bytes, synthetic
        r = bi.decode_raster_bytes(png_bytes(synthetic(2048, 1536)))
        res = gc.gradcam_all(bundle(), bi.preprocess(r.pixel01))
        self.assertEqual(len(res), 3)
        self.assertTrue(all(np.isfinite(q.cam).all() for q in res))


if __name__ == "__main__":
    unittest.main(verbosity=2)
