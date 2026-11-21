"""Tests for the SEPARATE external PNG/JPG path (letterbox to 1024x1024, non-standard).

Synthetic images are generated in memory. The only dataset access is one TRAIN DICOM
(00000003_000, already used in the smoke test) for the PNG-vs-DICOM equivalence check.
"""
import io
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import bgrel_infer as bi  # noqa: E402


def synthetic(width, height, seed=0) -> np.ndarray:
    """Textured uint8 image with four distinct corner markers and a unique darkest pixel."""
    rng = np.random.default_rng(seed)
    a = rng.integers(60, 200, (height, width)).astype(np.uint8)
    mh, mw = height // 8, width // 8
    a[:mh, :mw] = 230          # top-left
    a[:mh, -mw:] = 240         # top-right
    a[-mh:, :mw] = 250         # bottom-left
    a[-mh:, -mw:] = 220        # bottom-right
    a[height // 2, width // 2] = 10    # darkest value lies INSIDE the content
    return a


def png_bytes(arr, mode=None) -> bytes:
    buf = io.BytesIO()
    (Image.fromarray(arr) if mode is None else Image.fromarray(arr).convert(mode)).save(buf, "PNG")
    return buf.getvalue()


def reference_resize(arr, size) -> np.ndarray:
    """Independent re-implementation: same Pillow call, outside the demo code."""
    if size == (arr.shape[1], arr.shape[0]):
        return arr.astype(np.float32)
    return np.asarray(Image.fromarray(arr.astype(np.float32)).resize(size, Image.Resampling.BILINEAR),
                      dtype=np.float32)


class RasterGeometry(unittest.TestCase):
    CASES = {(1024, 1024): ((1024, 1024), (0, 0, 0, 0)),
             (2048, 1536): ((1024, 768), (0, 0, 128, 128)),     # landscape: pad top/bottom
             (1536, 2048): ((768, 1024), (128, 128, 0, 0))}     # portrait: pad left/right

    def check_case(self, w, h, data=None):
        arr = synthetic(w, h)
        r = bi.decode_raster_bytes(png_bytes(arr) if data is None else data)
        o, p = r.ops, r.ops["pad"]
        nw, nh = o["resized"]
        self.assertEqual(r.pixel01.shape, (1024, 1024))
        self.assertEqual(r.pixel01.dtype, np.float32)
        self.assertEqual(o["original_size"], (w, h))
        # Fits exactly in the canvas, limiting side = 1024, canvas fully accounted for.
        self.assertEqual(max(nw, nh), 1024)
        self.assertEqual(p["left"] + nw + p["right"], 1024)
        self.assertEqual(p["top"] + nh + p["bottom"], 1024)
        # Aspect ratio preserved (rounding of the non-limiting side only: < 0.5 px).
        self.assertLess(abs(nw - w * o["scale"]), 0.5 + 1e-9)
        self.assertLess(abs(nh - h * o["scale"]), 0.5 + 1e-9)
        self.assertAlmostEqual(o["scale"], min(1024 / w, 1024 / h))
        # Content = the WHOLE original resized (no cropping, no stretching), bit-identical to an
        # independent Pillow resize; canvas is then min-max normalised exactly like the DICOM path.
        ref = reference_resize(arr, (nw, nh))
        lo, hi = float(ref.min()), float(ref.max())
        content = r.pixel01[p["top"]:p["top"] + nh, p["left"]:p["left"] + nw]
        np.testing.assert_array_equal(content, ((ref - lo) / (hi - lo)).astype(np.float32))
        # Padding: exactly the content minimum (-> 0.0); no new minimum, max unchanged.
        mask = np.ones((1024, 1024), bool)
        mask[p["top"]:p["top"] + nh, p["left"]:p["left"] + nw] = False
        self.assertEqual(o["pad_value"], lo)
        if mask.any():
            self.assertTrue((r.pixel01[mask] == 0.0).all())
        self.assertEqual(float(r.pixel01.min()), 0.0)
        self.assertEqual(float(content.min()), 0.0)          # the content itself holds the minimum
        self.assertEqual(float(r.pixel01.max()), 1.0)
        # Corner markers survive at the corners of the content box (nothing cut off).
        for (ys, xs), val in (((slice(0, 4), slice(0, 4)), 230), ((slice(0, 4), slice(-4, None)), 240),
                              ((slice(-4, None), slice(0, 4)), 250), ((slice(-4, None), slice(-4, None)), 220)):
            got = content[ys, xs] * (hi - lo) + lo
            self.assertTrue(np.allclose(got, val, atol=1e-3), (w, h, val))
        # Through the unchanged research preprocess().
        t = bi.preprocess(r.pixel01)
        self.assertEqual(tuple(t.shape), (3, 224, 224))
        return r

    def test_1024x1024_no_resize_no_padding(self):
        r = self.check_case(1024, 1024)
        self.assertEqual(r.ops["resized"], (1024, 1024))
        self.assertEqual(r.ops["pad"], {"left": 0, "right": 0, "top": 0, "bottom": 0})
        self.assertTrue(r.ops["interpolation"].startswith("none"))
        a = synthetic(1024, 1024).astype(np.float32)
        np.testing.assert_array_equal(r.pixel01, (a - a.min()) / (a.max() - a.min()))

    def test_2048x1536_landscape(self):
        r = self.check_case(2048, 1536)
        self.assertEqual((r.ops["resized"], tuple(r.ops["pad"].values())), self.CASES[(2048, 1536)])

    def test_1536x2048_portrait(self):
        r = self.check_case(1536, 2048)
        self.assertEqual((r.ops["resized"], tuple(r.ops["pad"].values())), self.CASES[(1536, 2048)])

    def test_other_shapes_upscale_and_odd_padding(self):
        for w, h in ((512, 400), (3000, 1000), (1000, 3001), (777, 1024), (1024, 1023)):
            with self.subTest(size=(w, h)):
                self.check_case(w, h)

    def test_letterbox_geometry_table(self):
        g = bi.letterbox_geometry(3000, 1000)
        self.assertEqual(g["resized"], (1024, 341))
        self.assertEqual(g["pad"], {"left": 0, "right": 0, "top": 341, "bottom": 342})


class RasterFormats(unittest.TestCase):
    def test_rgb_uses_pillow_luminance(self):
        rgb = np.stack([synthetic(600, 800, 1), synthetic(600, 800, 2), synthetic(600, 800, 3)], -1)
        buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "PNG")
        r = bi.decode_raster_bytes(buf.getvalue())
        self.assertIn("luminance", r.metadata["Greyscale conversion"])
        lum = np.asarray(Image.fromarray(rgb).convert("L"))
        ref = reference_resize(lum, r.ops["resized"])
        p = r.ops["pad"]; nw, nh = r.ops["resized"]
        np.testing.assert_array_equal(r.pixel01[p["top"]:p["top"] + nh, p["left"]:p["left"] + nw],
                                      ((ref - ref.min()) / (ref.max() - ref.min())).astype(np.float32))

    def test_16bit_png_kept_lossless(self):
        a = (synthetic(1024, 1024).astype(np.uint16) * 200)          # values up to 50,000
        buf = io.BytesIO(); Image.fromarray(a).save(buf, "PNG")
        r = bi.decode_raster_bytes(buf.getvalue())
        self.assertTrue(r.metadata["Mode"].startswith("I"))
        self.assertIn("50000", r.metadata["Grey range (original)"])
        f = a.astype(np.float32)
        np.testing.assert_array_equal(r.pixel01, (f - f.min()) / (f.max() - f.min()))

    def test_jpeg_accepted(self):
        buf = io.BytesIO(); Image.fromarray(synthetic(1200, 900)).save(buf, "JPEG", quality=92)
        r = bi.decode_raster_bytes(buf.getvalue())
        self.assertEqual((r.metadata["Format"], r.ops["resized"]), ("JPEG", (1024, 768)))

    def test_fully_opaque_alpha_is_dropped_losslessly(self):
        """Browsers/screenshot tools save opaque PNGs as RGBA: same result as without alpha."""
        a = synthetic(2048, 1536)
        rgb = np.dstack([a, np.roll(a, 7, 1), np.roll(a, 3, 0)])
        opaque = np.full(a.shape, 255, np.uint8)
        for with_alpha, without in ((np.dstack([rgb, opaque]), rgb), (np.dstack([a, opaque]), a)):
            with self.subTest(channels=with_alpha.shape[-1]):
                r1 = bi.decode_raster_bytes(png_bytes(with_alpha))
                r2 = bi.decode_raster_bytes(png_bytes(without))
                np.testing.assert_array_equal(r1.pixel01, r2.pixel01)
                self.assertIn("fully opaque alpha channel dropped", r1.metadata["Greyscale conversion"])
        semi = np.dstack([rgb, opaque]); semi[0, 0, 3] = 254           # one non-opaque pixel
        with self.assertRaises(bi.RasterInputError):
            bi.decode_raster_bytes(png_bytes(semi))

    def test_invalid_images_rejected(self):
        a = synthetic(300, 300)
        gif = io.BytesIO(); Image.fromarray(a).save(gif, "GIF")
        bmp = io.BytesIO(); Image.fromarray(a).save(bmp, "BMP")
        rgba = io.BytesIO(); Image.fromarray(np.dstack([a, a, a, a])).save(rgba, "PNG")
        la = io.BytesIO(); Image.fromarray(np.dstack([a, a])).save(la, "PNG")
        const = png_bytes(np.full((500, 400), 77, np.uint8))
        path, _ = bi.train_reference_path("00000003_000.png")
        cases = {"empty": b"", "garbage": b"not an image", "truncated_png": png_bytes(a)[:200],
                 "gif": gif.getvalue(), "bmp": bmp.getvalue(), "rgba": rgba.getvalue(),
                 "la": la.getvalue(), "constant": const, "dicom_bytes": path.read_bytes()}
        for name, data in cases.items():
            with self.subTest(case=name), self.assertRaises(bi.RasterInputError):
                bi.decode_raster_bytes(data)

    def test_raster_result_never_matches_train(self):
        """No DICOM identity -> match_upload must never succeed for a raster."""
        r = bi.decode_raster_bytes(png_bytes(synthetic(1024, 1024)))
        self.assertEqual(r._identity, {})
        s, reason = bi.train_catalog().match_upload(r)
        self.assertIsNone(s)


class RasterVsDicom(unittest.TestCase):
    def test_png_of_train_dicom_pixels_equals_dicom_path(self):
        """1024x1024 MONOCHROME2 8-bit: raster path == DICOM path bit-for-bit, same scores."""
        import pydicom
        cat = bi.train_catalog()
        path, row = bi.train_reference_path("00000003_000.png")
        self.assertEqual(row["photometric_interpretation"], "MONOCHROME2")
        raw = np.asarray(pydicom.dcmread(str(path)).pixel_array)
        self.assertEqual(raw.dtype, np.uint8)
        r = bi.decode_raster_bytes(png_bytes(raw))
        d = cat.load("00000003_000.png")
        np.testing.assert_array_equal(r.pixel01, d.pixel01)
        self.assertTrue(torch.equal(bi.preprocess(r.pixel01), bi.preprocess(d.pixel01)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
