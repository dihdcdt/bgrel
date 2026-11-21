"""Section F box overlays: exact reuse of sections A/D drawers; never part of Grad-CAM (TRAIN only)."""
import dataclasses
import inspect
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))

import numpy as np  # noqa: E402

import bgrel_infer as bi  # noqa: E402
import gc_overlays as gco  # noqa: E402
import gradcam as gc  # noqa: E402
from test_gradcam import bundle  # noqa: E402

POS_K0 = "00000032_007.png"     # 2 expert boxes, both lesions K = 0
POS_K1 = "00000061_001.png"     # 1 expert box, 1 frozen reference region (1a)


def red_mask(img):
    return np.all(img == np.array(bi.GT_RED, np.uint8), axis=-1)


def blue_mask(img):
    return np.all(img == np.array(bi.CONTROL_BLUE, np.uint8), axis=-1)


class OverlayData(unittest.TestCase):
    def test_test_samples_are_as_expected(self):
        cat, ri = bi.train_catalog(), bi.region_index()
        self.assertEqual(len(cat.boxes(POS_K0)), 2)
        self.assertEqual([l.k for l in ri.for_image(POS_K0)], [0, 0])
        self.assertEqual(len(cat.boxes(POS_K1)), 1)
        self.assertEqual([(l.k, [c.label for c in l.controls]) for l in ri.for_image(POS_K1)], [(1, ["1a"])])


class ExactReuse(unittest.TestCase):
    def test_gt_layer_reproduces_section_a_drawing_exactly(self):
        cat = bi.train_catalog()
        for iid in (POS_K0, POS_K1):
            pix = cat.load(iid).pixel01
            boxes = cat.boxes(iid)
            mine = gco.apply_layers(gco.grey_rgb(pix), [gco.gt_layer(boxes)])
            np.testing.assert_array_equal(mine, bi.draw_ground_truth(pix, boxes))

    def test_reference_layer_is_exactly_the_section_d_control_drawing(self):
        cat, ri = bi.train_catalog(), bi.region_index()
        pix = cat.load(POS_K1).pixel01
        lesions = ri.for_image(POS_K1)
        ref = gco.reference_layer(lesions)
        lesion_only = bi.draw_regions(np.zeros((1024, 1024), np.float32),
                                      tuple(dataclasses.replace(l, controls=()) for l in lesions))
        lesion_layer = (lesion_only.any(axis=-1), lesion_only)
        # reference layer + section-D lesion drawing == section D's full drawing, pixel for pixel
        np.testing.assert_array_equal(gco.apply_layers(gco.grey_rgb(pix), [ref, lesion_layer]),
                                      bi.draw_regions(pix, lesions))
        # the reference layer alone contains blue and no lesion red outline
        only_ref = gco.apply_layers(gco.grey_rgb(pix), [ref])
        self.assertTrue(blue_mask(only_ref).any())
        c = lesions[0].controls[0].xyxy
        x0, y0 = int(np.floor(c[0])), int(np.floor(c[1]))
        self.assertTrue(blue_mask(only_ref)[y0, x0])          # same pixel bounds as section D
        self.assertFalse(red_mask(only_ref).any())

    def test_no_reference_layer_content_when_k0(self):
        ri = bi.region_index()
        mask, _ = gco.reference_layer(ri.for_image(POS_K0))
        self.assertFalse(mask.any())


class FourPanels(unittest.TestCase):
    def test_same_red_boxes_on_original_and_all_three_heatmaps(self):
        b, cat, ri = bundle(), bi.train_catalog(), bi.region_index()
        for iid in (POS_K0, POS_K1):
            with self.subTest(image_id=iid):
                pix = cat.load(iid).pixel01
                x = bi.preprocess(pix)
                results = gc.gradcam_all(b, x)
                layers = [gco.gt_layer(cat.boxes(iid))]
                if any(l.k for l in ri.for_image(iid)):
                    layers.insert(0, gco.reference_layer(ri.for_image(iid)))
                vmax = max(float(r.cam.max()) for r in results)
                plain = [gco.grey_rgb(pix)] + [
                    gc.overlay(pix, gc.normalise(gc.upsample(r.cam, pix.shape), vmax)[0], 0.55)
                    for r in results]
                panels = [gco.apply_layers(p, layers) for p in plain]
                union = np.zeros((1024, 1024), bool)
                for m, _ in layers:
                    union |= m
                ref_red = red_mask(panels[0])
                self.assertTrue(ref_red.any())
                for p, base in zip(panels, plain):
                    np.testing.assert_array_equal(red_mask(p), ref_red)            # identical boxes
                    np.testing.assert_array_equal(blue_mask(p), blue_mask(panels[0]))
                    np.testing.assert_array_equal(p[~union], base[~union])          # heatmap kept
                self.assertEqual(blue_mask(panels[0]).any(), iid == POS_K1)

    def test_disabled_layers_leave_panels_unchanged(self):
        pix = bi.train_catalog().load(POS_K1).pixel01
        base = gco.grey_rgb(pix)
        np.testing.assert_array_equal(gco.apply_layers(base, []), base)

    def test_gradcam_independent_of_boxes(self):
        """Grad-CAM API takes no boxes; maps are identical before/after building box layers."""
        for fn in (gc.gradcam, gc.gradcam_all):
            params = set(inspect.signature(fn).parameters)
            self.assertFalse(params & {"boxes", "gt_boxes", "lesions", "regions", "controls"})
        b, cat, ri = bundle(), bi.train_catalog(), bi.region_index()
        x = bi.preprocess(cat.load(POS_K1).pixel01)
        before = [r.cam.copy() for r in gc.gradcam_all(b, x)]
        gco.gt_layer(cat.boxes(POS_K1)); gco.reference_layer(ri.for_image(POS_K1))
        after = [r.cam for r in gc.gradcam_all(b, x)]
        for a, c in zip(before, after):
            np.testing.assert_array_equal(a, c)
        self.assertNotIn("gc_overlays", inspect.getsource(gc))


if __name__ == "__main__":
    unittest.main(verbosity=2)
