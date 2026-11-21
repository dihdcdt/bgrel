"""Headless end-to-end tests of app.py with Streamlit AppTest (TRAIN images only)."""
import re
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))

import numpy as np  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import bgrel_infer as bi  # noqa: E402
from test_regression import SCORE_TOL, SMOKE_REFERENCE  # noqa: E402

ALL_SECTIONS = ["A. Image input", "B. Model predictions", "C. Model comparison", "D. Regional evidence", "E. About BG-REL", "F. Grad-CAM Explainability"]


def fresh_app():
    at = AppTest.from_file(str(DEMO / "app.py"), default_timeout=240)
    at.run()
    return at


def check_scores(tc, at, expected):
    """Comparison cards are rendered in bi.RUN_IDS order (B0, B1, A3)."""
    tc.assertEqual(len(at.metric), 3)
    for metric, run_id in zip(at.metric, bi.RUN_IDS):
        tc.assertEqual(metric.label, "Score")
        tc.assertLessEqual(abs(float(metric.value) - expected[run_id][1]), SCORE_TOL + 5e-5)
    svgs = [m.value for m in at.markdown if m.value.startswith("<svg")]
    tc.assertEqual(len(svgs), 1)
    tc.assertEqual(svgs[0].count("<path "), 3)
    for metric in at.metric:  # bar labels come from the same real scores
        labels = [float(v) for v in re.findall(r'fill="#0b0b0b">(\d\.\d{3})<', svgs[0])]
        tc.assertLessEqual(min(abs(v - float(metric.value)) for v in labels), 6e-4)


def rendered_text(at) -> str:
    parts = []
    for kind in ("markdown", "caption", "code", "header", "subheader", "info", "error", "warning",
                 "success", "text"):
        parts += [str(e.value) for e in getattr(at, kind)]
    for sb in at.selectbox:
        parts += [str(o) for o in sb.options]
    parts += [f"{m.label} {m.value}" for m in at.metric]
    return "\n".join(parts)


class AppEndToEnd(unittest.TestCase):
    def test_default_is_train_sample_with_verified_positive(self):
        at = fresh_app()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(len(at.error), 0, [e.value for e in at.error])
        self.assertEqual(at.title[0].value, "BG-REL: Lung-Aware Box-Guided Relative Evidence Learning")
        self.assertEqual(at.radio(key="source").value, "Select Sample from TRAIN Dataset")
        self.assertIn("Select Sample from TRAIN Dataset", [s.value for s in at.subheader])
        self.assertEqual([h.value for h in at.header], ALL_SECTIONS)
        self.assertEqual(at.selectbox(key="sample_Positive").value.image_id, "00000032_007.png")
        check_scores(self, at, SMOKE_REFERENCE["00000032_007.png"][1])

    def test_select_each_verified_train_example(self):
        at = fresh_app()
        cat = bi.train_catalog()
        for image_id, (label, expected) in SMOKE_REFERENCE.items():
            s = cat.by_id[image_id]
            with self.subTest(image_id=image_id):
                at.radio(key="category").set_value(s.category).run()
                at.selectbox(key=f"sample_{s.category}").set_value(s).run()
                self.assertFalse(at.exception, at.exception)
                self.assertEqual(len(at.error), 0, [e.value for e in at.error])
                self.assertEqual([h.value for h in at.header], ALL_SECTIONS)
                text = rendered_text(at)
                self.assertIn(f"`{s.display_id}`", text)        # image ID
                self.assertIn(label, text)                      # ground-truth label
                self.assertIn(f"Category**  \n{s.category}", text)
                # X-ray shown; positives also get the section-D region overlay.
                self.assertEqual(len(at.get("image")), 2 if s.category == "Positive" else 1)
                check_scores(self, at, expected)
                # Privacy: no identifiers from the manifest row anywhere on the page.
                _, row = bi.train_reference_path(image_id)
                for field in ("dicom_patient_id", "study_uid", "series_uid", "sop_uid", "dicom_path"):
                    self.assertNotIn(row[field], text, field)
                self.assertNotIn("PatientID", text)

    def test_ground_truth_panel_positive_and_negatives(self):
        at = fresh_app()
        cat = bi.train_catalog()
        for image_id, n_boxes in (("00000032_007.png", 2), ("00000003_003.png", 0), ("00000003_000.png", 0)):
            s = cat.by_id[image_id]
            with self.subTest(image_id=image_id):
                at.radio(key="category").set_value(s.category).run()
                at.selectbox(key=f"sample_{s.category}").set_value(s).run()
                self.assertFalse(at.exception, at.exception)
                text = rendered_text(at)
                self.assertEqual(len(at.get("image")), 2 if n_boxes else 1)
                if n_boxes:
                    self.assertIn(bi.GT_LEGEND, text)
                    self.assertIn(f"Annotated lesion regions: {n_boxes}", text)
                    self.assertIn("552.2", text)            # box 1 x
                    self.assertIn("522.3", text)            # box 2 height
                    self.assertTrue(at.toggle(key="show_gt").value)
                    at.toggle(key="show_gt").set_value(False).run()   # boxes can be hidden
                    self.assertFalse(at.exception, at.exception)
                    at.toggle(key="show_gt").set_value(True).run()
                else:
                    self.assertIn("No ground-truth lesion box is available for this image.", text)
                    self.assertNotIn(bi.GT_LEGEND, text)
                    self.assertNotIn("Annotated lesion regions", text)
                    self.assertEqual(len(at.toggle), 0)

    def test_upload_matched_vs_unmatched_ground_truth(self):
        import io as _io
        import pydicom
        at = fresh_app()
        at.radio(key="source").set_value("Upload a DICOM file").run()
        path, _ = bi.train_reference_path("00000032_007.png")
        at.file_uploader[0].set_value((path.name, path.read_bytes(), "application/dicom")).run()
        text = rendered_text(at)
        self.assertTrue(any("identical to a verified TRAIN record" in s.value for s in at.success))
        self.assertIn(bi.GT_LEGEND, text)
        self.assertIn("Annotated lesion regions: 2", text)
        # Same pixels but a different PatientID: not a confident match -> no GT shown.
        ds = pydicom.dcmread(_io.BytesIO(path.read_bytes())); ds.PatientID = "someone-else"
        buf = _io.BytesIO(); ds.save_as(buf)
        at.file_uploader[0].set_value(("other.dcm", buf.getvalue(), "application/dicom")).run()
        self.assertFalse(at.exception, at.exception)
        text = rendered_text(at)
        self.assertTrue(any("No confident match" in i.value for i in at.info))
        self.assertNotIn(bi.GT_LEGEND, text)
        self.assertNotIn("**Ground-truth label**", text)      # no sample card
        self.assertNotIn("Lung Opacity (Positive)", text)    # no ground truth in section B
        self.assertEqual(len(at.metric), 3)                  # predictions still run

    def test_regional_evidence_reference_image(self):
        at = fresh_app()
        at.toggle(key="only_k").set_value(True).run()
        opts = at.selectbox(key="sample_Positive_k").options
        self.assertEqual(len(opts), 2421)
        self.assertTrue(opts[0].startswith("00000032_058"))
        self.assertEqual(at.selectbox(key="sample_Positive_k").value.image_id, bi.REGION_REFERENCE_ID)
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(len(at.error), 0, [e.value for e in at.error])
        self.assertEqual([h.value for h in at.header], ALL_SECTIONS)
        text = rendered_text(at)
        self.assertIn("2 lesion region(s); 1 with frozen reference regions", text)
        self.assertIn("TRAIN, in-sample", text)
        self.assertIn("SEARCH_EXHAUSTED", text)
        self.assertIn(bi.REGION_LEGEND, text)
        self.assertEqual(len(at.get("image")), 2)            # radiograph + region overlay
        # The table shows exactly the values the verified scorer produces.
        ri, b = bi.region_index(), bi.ModelBundle()
        x = bi.preprocess(bi.train_catalog().load(bi.REGION_REFERENCE_ID).pixel01)
        ev = b.region_evidence(x, bi.REGION_REFERENCE_ID, ri.for_image(bi.REGION_REFERENCE_ID), ri)
        for run_id in bi.RUN_IDS:
            rem = ev[run_id]["lesions"][1]["rem"]
            self.assertIn(f"**{rem:+.3f}**", text)
        for field in ("annotation_id", "A_xogovk", "A_e58zm6"):
            self.assertNotIn(field, text)                    # no internal keys shown

    def test_regional_evidence_verified_examples(self):
        at = fresh_app()
        text = rendered_text(at)                             # default: 00000032_007
        self.assertIn("2 lesion region(s); 0 with frozen reference regions", text)
        self.assertIn("IMAGE_SKIP_REL", text)
        self.assertIn("REM is undefined", text)
        cat = bi.train_catalog()
        for neg in ("00000003_003.png", "00000003_000.png"):
            s = cat.by_id[neg]
            at.radio(key="category").set_value("Negative").run()
            at.selectbox(key="sample_Negative").set_value(s).run()
            self.assertFalse(at.exception, at.exception)
            self.assertIn("negative TRAIN image: it has no expert lesion box", rendered_text(at))

    def test_regional_evidence_not_inferred_for_unmatched_upload(self):
        import io as _io
        import pydicom
        at = fresh_app()
        at.radio(key="source").set_value("Upload a DICOM file").run()
        path, _ = bi.train_reference_path(bi.REGION_REFERENCE_ID)
        ds = pydicom.dcmread(_io.BytesIO(path.read_bytes())); ds.PatientID = "someone-else"
        buf = _io.BytesIO(); ds.save_as(buf)
        at.file_uploader[0].set_value(("x.dcm", buf.getvalue(), "application/dicom")).run()
        self.assertFalse(at.exception, at.exception)
        text = rendered_text(at)
        self.assertIn("Regional evidence is available only for TRAIN images", text)
        self.assertNotIn("Reference logits", text)
        # The genuine file matches the TRAIN record and gets the region view.
        at.file_uploader[0].set_value((path.name, path.read_bytes(), "application/dicom")).run()
        self.assertIn("1 with frozen reference regions", rendered_text(at))

    def test_raster_upload_landscape(self):
        from test_raster import png_bytes, synthetic
        at = fresh_app()
        at.radio(key="source").set_value("Upload PNG/JPG (external raster)").run()
        self.assertIn(bi.RASTER_LABEL, rendered_text(at))               # label before upload
        at.file_uploader(key="raster_upload").set_value(
            ("scan.png", png_bytes(synthetic(2048, 1536)), "image/png")).run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(len(at.error), 0, [e.value for e in at.error])
        self.assertEqual([h.value for h in at.header], ALL_SECTIONS)
        text = rendered_text(at)
        self.assertIn("out-of-distribution", text)
        self.assertIn("not validated probabilities", text)
        self.assertIn("2048 × 1536 px", text)                            # original size
        self.assertIn("1024 × 768 px", text)                             # resized
        self.assertIn("0 / 0 / 128 / 128 px", text)                      # padding
        self.assertEqual(len(at.metric), 3)
        # Never any ground truth or regional evidence for external images.
        for banned in (bi.GT_LEGEND, "Annotated lesion regions", "**Ground-truth label**",
                       "Reference logits", "REM |", bi.REGION_LEGEND):
            self.assertNotIn(banned, text)
        self.assertIn("Regional evidence is available only for TRAIN images", text)
        self.assertEqual(len(at.get("image")), 1)

    def test_raster_with_train_name_and_pixels_gets_no_ground_truth(self):
        """Filename and pixels of a positive TRAIN image: still no GT (no DICOM identity)."""
        import pydicom
        from test_raster import png_bytes
        at = fresh_app()
        at.radio(key="source").set_value("Upload PNG/JPG (external raster)").run()
        path, _ = bi.train_reference_path("00000032_007.png")
        raw = np.asarray(pydicom.dcmread(str(path)).pixel_array)
        at.file_uploader(key="raster_upload").set_value(
            ("00000032_007.png", png_bytes(raw), "image/png")).run()
        self.assertFalse(at.exception, at.exception)
        text = rendered_text(at)
        self.assertNotIn(bi.GT_LEGEND, text)
        self.assertNotIn("Reference logits", text)
        self.assertNotIn("**Ground-truth label**", text)
        self.assertNotIn("Matched TRAIN record", text)
        check_scores(self, at, SMOKE_REFERENCE["00000032_007.png"][1])   # same pixels -> same scores

    def test_raster_invalid_upload(self):
        at = fresh_app()
        at.radio(key="source").set_value("Upload PNG/JPG (external raster)").run()
        at.file_uploader(key="raster_upload").set_value(("bad.png", b"not an image", "image/png")).run()
        self.assertFalse(at.exception, at.exception)
        self.assertTrue(any("cannot be used" in e.value for e in at.error))
        self.assertNotIn("B. Model predictions", [h.value for h in at.header])

    # ---- F. Grad-CAM ------------------------------------------------------------------
    def _gradcam_panels(self, at):
        return [m.value for m in at.markdown if re.match(r"\*\*(B0|B1|A3)\*\* · logit", m.value)]

    def test_gradcam_train_sample_and_no_stale_results(self):
        at = fresh_app()                                          # default: 00000032_007
        self.assertIn("F. Grad-CAM Explainability", [h.value for h in at.header])
        n_img = len(at.get("image"))
        self.assertEqual(self._gradcam_panels(at), [])            # nothing until requested
        at.button(key="gradcam_button").click().run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(len(at.error), 0, [e.value for e in at.error])
        self.assertEqual(len(at.get("image")), n_img + 4)         # original + B0 + B1 + A3
        panels = self._gradcam_panels(at)
        self.assertEqual([p[:6] for p in panels], ["**B0**", "**B1**", "**A3**"])
        for p, run_id in zip(panels, bi.RUN_IDS):                 # same logits as the predictions
            self.assertIn(f"logit {SMOKE_REFERENCE['00000032_007.png'][1][run_id][0]:+.3f}", p)
        text = rendered_text(at)
        self.assertIn("Grad-CAM is **not** a replacement for them", text)
        self.assertIn("post-hoc visualisation of prediction evidence", text)
        self.assertIn("All three panels share one scale", text)
        results = at.session_state["gradcam"]["results"]
        # Changing display options re-renders without recomputing.
        at.radio(key="gc_scale").set_value("Per-model scale").run()
        self.assertIs(at.session_state["gradcam"]["results"], results)
        self.assertIn("NOT comparable across models", rendered_text(at))
        # Changing the image must not show the previous image's heatmaps.
        neg = bi.train_catalog().by_id["00000003_003.png"]
        at.radio(key="category").set_value("Negative").run()
        at.selectbox(key="sample_Negative").set_value(neg).run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(self._gradcam_panels(at), [])
        self.assertNotIn("gradcam", at.session_state)
        at.button(key="gradcam_button").click().run()
        panels = self._gradcam_panels(at)
        self.assertEqual(len(panels), 3)
        self.assertIn(f"logit {SMOKE_REFERENCE['00000003_003.png'][1]['b0_dev_s42'][0]:+.3f}", panels[0])

    def test_gradcam_raster_upload_keeps_ood_warnings_and_no_ground_truth(self):
        from test_raster import png_bytes, synthetic
        at = fresh_app()
        at.radio(key="source").set_value("Upload PNG/JPG (external raster)").run()
        at.file_uploader(key="raster_upload").set_value(
            ("scan.png", png_bytes(synthetic(1536, 2048)), "image/png")).run()
        at.button(key="gradcam_button").click().run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(len(at.error), 0, [e.value for e in at.error])
        self.assertEqual(len(self._gradcam_panels(at)), 3)
        text = rendered_text(at)
        self.assertIn("Grad-CAM below describes the models' response to a converted external image", text)
        self.assertIn("out-of-distribution", text)
        for banned in (bi.GT_LEGEND, "Annotated lesion regions", "**Ground-truth label**",
                       "Reference logits", bi.REGION_LEGEND):
            self.assertNotIn(banned, text)

    def test_gradcam_dicom_upload(self):
        at = fresh_app()
        at.radio(key="source").set_value("Upload a DICOM file").run()
        path, _ = bi.train_reference_path("00000003_000.png")
        at.file_uploader[0].set_value((path.name, path.read_bytes(), "application/dicom")).run()
        at.button(key="gradcam_button").click().run()
        self.assertFalse(at.exception, at.exception)
        panels = self._gradcam_panels(at)
        self.assertEqual(len(panels), 3)
        self.assertIn(f"logit {SMOKE_REFERENCE['00000003_000.png'][1]['a3_l100_s42'][0]:+.3f}", panels[2])

    # ---- F. Grad-CAM box overlays (visual only) ----------------------------------------
    BOX_NOTE = ("Red boxes: expert-annotated lesion regions. Blue dashed boxes: frozen reference "
                "regions. Boxes are shown for visual comparison only and are not used to calculate "
                "Grad-CAM.")

    def _select_positive(self, at, image_id):
        s = bi.train_catalog().by_id[image_id]
        at.radio(key="category").set_value("Positive").run()
        at.selectbox(key="sample_Positive").set_value(s).run()

    def test_gradcam_boxes_k0_sample(self):
        at = fresh_app()                                        # default 00000032_007
        at.button(key="gradcam_button").click().run()
        self.assertIn(self.BOX_NOTE, rendered_text(at))
        self.assertFalse(at.checkbox(key="gc_show_gt").disabled)
        self.assertTrue(at.checkbox(key="gc_show_ref").disabled)   # K = 0: no reference data
        self.assertIn("No valid frozen reference region exists for this image", rendered_text(at))
        self.assertIn("IMAGE_SKIP_REL", rendered_text(at))
        results = at.session_state["gradcam"]["results"]
        n_img = len(at.get("image"))
        at.checkbox(key="gc_show_gt").check().run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(len(at.get("image")), n_img)
        self.assertIs(at.session_state["gradcam"]["results"], results)   # no recomputation
        at.checkbox(key="gc_show_gt").uncheck().run()
        self.assertFalse(at.exception, at.exception)

    def test_gradcam_boxes_k1_sample(self):
        at = fresh_app()
        self._select_positive(at, "00000061_001.png")
        at.button(key="gradcam_button").click().run()
        self.assertFalse(at.checkbox(key="gc_show_gt").disabled)
        self.assertFalse(at.checkbox(key="gc_show_ref").disabled)
        self.assertNotIn("No valid frozen reference region exists", rendered_text(at))
        results = at.session_state["gradcam"]["results"]
        at.checkbox(key="gc_show_gt").check().run()
        at.checkbox(key="gc_show_ref").check().run()
        self.assertFalse(at.exception, at.exception)
        self.assertEqual(len(at.error), 0, [e.value for e in at.error])
        self.assertIs(at.session_state["gradcam"]["results"], results)
        self.assertEqual(len(self._gradcam_panels(at)), 3)
        at.checkbox(key="gc_show_ref").uncheck().run()
        at.checkbox(key="gc_show_gt").uncheck().run()
        self.assertFalse(at.exception, at.exception)

    def test_gradcam_boxes_never_for_raster_or_unmatched_dicom(self):
        import io as _io
        import pydicom
        from test_raster import png_bytes
        path, _ = bi.train_reference_path("00000061_001.png")
        raw = path.read_bytes()
        ds = pydicom.dcmread(_io.BytesIO(raw)); ds.PatientID = "someone-else"
        buf = _io.BytesIO(); ds.save_as(buf)
        cases = (("Upload PNG/JPG (external raster)", "raster_upload",
                  ("00000061_001.png", png_bytes(np.asarray(pydicom.dcmread(str(path)).pixel_array)), "image/png")),
                 ("Upload a DICOM file", None, ("x.dcm", buf.getvalue(), "application/dicom")))
        for source, key, upload in cases:
            with self.subTest(source=source):
                at = fresh_app()
                at.radio(key="source").set_value(source).run()
                up = at.file_uploader(key=key) if key else at.file_uploader[0]
                up.set_value(upload).run()
                at.button(key="gradcam_button").click().run()
                self.assertFalse(at.exception, at.exception)
                self.assertTrue(at.checkbox(key="gc_show_gt").disabled)
                self.assertTrue(at.checkbox(key="gc_show_ref").disabled)
                self.assertIn("No ground-truth lesion boxes or reference regions are shown",
                              rendered_text(at))
                self.assertNotIn(bi.GT_LEGEND, rendered_text(at))

    def test_gradcam_without_image(self):
        at = fresh_app()
        at.radio(key="source").set_value("Upload a DICOM file").run()
        self.assertFalse(at.exception, at.exception)
        self.assertIn("to generate Grad-CAM", rendered_text(at))
        self.assertEqual(len([b for b in at.button if b.key == "gradcam_button"]), 0)

    def test_category_lists_and_no_val_test_options(self):
        import csv
        at = fresh_app()
        with (bi.RESEARCH_ROOT / "data/processed/split.csv").open(encoding="utf-8-sig") as f:
            non_train = {r["image_id"][:-4] for r in csv.DictReader(f) if r["split"] != "train"}
        for category, n in (("Positive", 5685), ("Negative", 18063)):
            with self.subTest(category=category):
                at.radio(key="category").set_value(category).run()
                opts = at.selectbox(key=f"sample_{category}").options
                self.assertEqual(len(opts), n)
                shown_ids = {o.split(" — ")[0] for o in opts}
                self.assertFalse(shown_ids & non_train)

    def test_random_sample_button(self):
        at = fresh_app()
        at.radio(key="category").set_value("Negative").run()
        at.button[0].click().run()
        self.assertFalse(at.exception, at.exception)
        s = at.selectbox(key="sample_Negative").value
        self.assertEqual(s.category, "Negative")
        self.assertEqual(len(at.metric), 3)

    def test_upload_train_images_end_to_end(self):
        at = fresh_app()
        at.radio(key="source").set_value("Upload a DICOM file").run()
        self.assertEqual([h.value for h in at.header], ["A. Image input", "E. About BG-REL", "F. Grad-CAM Explainability"])
        for image_id, (label, expected) in SMOKE_REFERENCE.items():
            with self.subTest(image_id=image_id):
                path, _ = bi.train_reference_path(image_id)
                at.file_uploader[0].set_value((path.name, path.read_bytes(), "application/dicom"))
                at.run()
                self.assertFalse(at.exception, at.exception)
                self.assertEqual(len(at.error), 0, [e.value for e in at.error])
                self.assertEqual([h.value for h in at.header], ALL_SECTIONS)
                check_scores(self, at, expected)
                self.assertNotIn("pyarrow", sys.modules)

    def test_invalid_upload_shows_error(self):
        at = fresh_app()
        at.radio(key="source").set_value("Upload a DICOM file").run()
        at.file_uploader[0].set_value(("broken.dcm", b"this is not a DICOM file", "application/dicom"))
        at.run()
        self.assertFalse(at.exception, at.exception)
        self.assertTrue(any("cannot be used" in e.value for e in at.error))
        self.assertNotIn("B. Model predictions", [h.value for h in at.header])


if __name__ == "__main__":
    unittest.main(verbosity=2)
