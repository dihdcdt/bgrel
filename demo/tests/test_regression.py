"""Regression tests for the BG-REL demo inference path (TRAIN images only).

Reference values are the outputs of the verified GPU smoke test (scratchpad
smoke_infer.py, RTX 3050, PyTorch default TF32 policy), printed to 4 decimals.
These are TRAIN examples seen during training - a consistency check, not an
evaluation. No VAL/TEST image is accessed.

Run (from demo/):  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import bgrel_infer as bi  # noqa: E402

# image_id: (ground-truth label, {run_id: (logit, P(Lung Opacity))})
SMOKE_REFERENCE = {
    "00000032_007.png": ("Lung Opacity", {"b0_dev_s42": (0.5989, 0.6454),
                                          "b1_boxpos_s42": (0.7578, 0.6809),
                                          "a3_l100_s42": (1.1528, 0.7600)}),
    "00000003_003.png": ("Normal", {"b0_dev_s42": (-4.2012, 0.0148),
                                    "b1_boxpos_s42": (-4.4181, 0.0119),
                                    "a3_l100_s42": (-3.6683, 0.0249)}),
    "00000003_000.png": ("No Lung Opacity / Not Normal", {"b0_dev_s42": (-4.6786, 0.0092),
                                                          "b1_boxpos_s42": (-3.7731, 0.0225),
                                                          "a3_l100_s42": (-4.1594, 0.0154)}),
}
# 4-decimal rounding (5e-5) + run-to-run GPU/TF32 variation.
LOGIT_TOL = 2e-3
SCORE_TOL = 5e-4

_BUNDLE = None


def bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = bi.ModelBundle()
    return _BUNDLE


class DemoRegression(unittest.TestCase):
    def test_no_torchxrayvision_loaded(self):
        self.assertNotIn("torchxrayvision", sys.modules)

    def test_tf32_policy_matches_smoke_test(self):
        self.assertTrue(torch.backends.cudnn.allow_tf32)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)

    def test_dataset_diff_is_path_only(self):
        bi.dataset_diff_is_path_only()  # raises IntegrityError otherwise

    def test_strict_checked_loading_all_three(self):
        b = bundle()
        self.assertEqual(tuple(b.models), bi.RUN_IDS)
        for run_id, rec in b.records.items():
            self.assertEqual(rec["epoch"], 3, run_id)
            self.assertEqual(len(rec["checkpoint_sha256"]), 64, run_id)

    def test_refuses_non_train_image(self):
        import csv
        with (bi.RESEARCH_ROOT / "data/processed/split.csv").open(encoding="utf-8-sig") as f:
            val_id = next(r["image_id"] for r in csv.DictReader(f) if r["split"] == "val")
        with self.assertRaises(PermissionError):
            bi.train_reference_path(val_id)  # raises before any file is opened

    def test_preprocessing_bitwise_equal_to_research_dataset(self):
        from dataset import RSNAImageDataset, decode_dicom
        for image_id in SMOKE_REFERENCE:
            with self.subTest(image_id=image_id):
                path, row = bi.train_reference_path(image_id)
                mine = bi.decode_dicom_bytes(path.read_bytes())
                self.assertTrue(np.array_equal(mine.pixel01, decode_dicom(bi.RESEARCH_ROOT, row)))
                ref, _ = RSNAImageDataset(bi.RESEARCH_ROOT, [row], train=False, seed=42, epoch=0)[0]
                self.assertTrue(torch.equal(bi.preprocess(mine.pixel01), ref))

    def test_predictions_match_smoke_test(self):
        b = bundle()
        print()
        for image_id, (label, expected) in SMOKE_REFERENCE.items():
            path, row = bi.train_reference_path(image_id)
            self.assertEqual(row["final_label"], label)
            x = bi.preprocess(bi.decode_dicom_bytes(path.read_bytes()).pixel01)
            for p in b.predict(x):
                exp_logit, exp_score = expected[p.run_id]
                print(f"  {image_id} {p.label}: logit={p.logit:+.4f} (ref {exp_logit:+.4f}) "
                      f"score={p.score:.4f} (ref {exp_score:.4f}) |d|={abs(p.score - exp_score):.1e} "
                      f"{p.ms:.2f} ms {p.device}")
                with self.subTest(image_id=image_id, run=p.run_id):
                    self.assertLessEqual(abs(p.logit - exp_logit), LOGIT_TOL)
                    self.assertLessEqual(abs(p.score - exp_score), SCORE_TOL)
                    self.assertEqual(p.note, "")

    def test_train_catalog_is_train_only_and_joined_correctly(self):
        import csv
        cat = bi.train_catalog()
        with (bi.RESEARCH_ROOT / "data/processed/split.csv").open(encoding="utf-8-sig") as f:
            split = {r["image_id"]: r["split"] for r in csv.DictReader(f)}
        listed = [s.image_id for c in ("Positive", "Negative") for s in cat.by_category[c]]
        self.assertEqual(len(listed), 23748)
        self.assertEqual(len(set(listed)), 23748)
        self.assertEqual({split[i] for i in listed}, {"train"})        # never VAL/TEST
        self.assertEqual(len(cat.by_category["Positive"]), 5685)
        self.assertEqual(len(cat.by_category["Negative"]), 18063)
        self.assertEqual({s.final_label for s in cat.by_category["Positive"]}, {"Lung Opacity"})
        self.assertEqual({s.final_label for s in cat.by_category["Negative"]},
                         {"Normal", "No Lung Opacity / Not Normal"})
        self.assertEqual(set(cat._rows), set(listed))                   # no VAL/TEST rows retained

    def test_verified_examples_listed_first_with_correct_category(self):
        cat = bi.train_catalog()
        expect = {"00000032_007.png": ("Lung Opacity", "Positive"),
                  "00000003_003.png": ("Normal", "Negative"),
                  "00000003_000.png": ("No Lung Opacity / Not Normal", "Negative")}
        for image_id, (label, category) in expect.items():
            s = cat.by_id[image_id]
            self.assertEqual((s.final_label, s.category), (label, category))
        self.assertEqual(cat.options("Positive")[0].image_id, "00000032_007.png")
        self.assertEqual([s.image_id for s in cat.options("Negative")[:2]],
                         ["00000003_003.png", "00000003_000.png"])

    def test_sample_fields_expose_no_identifiers(self):
        import dataclasses
        fields = {f.name for f in dataclasses.fields(bi.TrainSample)}
        self.assertEqual(fields, {"image_id", "final_label", "category", "view_position"})

    def test_catalog_refuses_val_and_test_ids(self):
        import csv
        with (bi.RESEARCH_ROOT / "data/processed/split.csv").open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        cat = bi.train_catalog()
        for part in ("val", "test"):
            image_id = next(r["image_id"] for r in rows if r["split"] == part)
            with self.subTest(part=part):
                self.assertNotIn(image_id, cat.by_id)
                with self.assertRaises(PermissionError):
                    cat.load(image_id)          # raises before any file is opened

    def test_catalog_load_matches_upload_path_and_smoke_test(self):
        """Selecting a sample = manifest-resolved DICOM through the same decoder + models."""
        b, cat = bundle(), bi.train_catalog()
        for image_id, (label, expected) in SMOKE_REFERENCE.items():
            path, _ = bi.train_reference_path(image_id)
            self.assertNotEqual(path.stem, image_id[:-4])   # DICOM filename is the SOP UID, not the ID
            via_catalog = cat.load(image_id)
            via_upload = bi.decode_dicom_bytes(path.read_bytes())
            self.assertTrue(np.array_equal(via_catalog.pixel01, via_upload.pixel01))
            for p in b.predict(bi.preprocess(via_catalog.pixel01)):
                with self.subTest(image_id=image_id, run=p.run_id):
                    self.assertLessEqual(abs(p.score - expected[p.run_id][1]), SCORE_TOL)

    # ---- Ground-truth lesion boxes --------------------------------------------------
    EXPECTED_BOXES = {  # adjudicated RSNA "Calculated" annotations (annotation JSON)
        "00000032_007.png": [(1, 552.17937, 146.9417, 236.48427, 262.8879),
                             (2, 231.89238, 194.00897, 219.26456, 522.33181)],
        "00000003_003.png": [],
        "00000003_000.png": [],
    }

    def test_ground_truth_boxes_for_verified_examples(self):
        cat = bi.train_catalog()
        for image_id, expected in self.EXPECTED_BOXES.items():
            with self.subTest(image_id=image_id):
                got = [(b.number, b.x, b.y, b.width, b.height) for b in cat.boxes(image_id)]
                self.assertEqual(len(got), len(expected))
                for g, e in zip(got, expected):
                    self.assertEqual(g[0], e[0])
                    for a, c in zip(g[1:], e[1:]):
                        self.assertAlmostEqual(a, c, places=4)

    def test_boxes_match_annotation_json_calculated_group(self):
        """Independent check of boxes.csv against the raw adjudicated annotation JSON."""
        import json
        with open(bi.RESEARCH_ROOT / "data/raw/pneumonia-challenge-annotations-"
                  "adjudicated-kaggle_2018.json", encoding="utf-8-sig") as f:   # read-only
            ann = json.load(f)
        calc = next(g for g in ann["labelGroups"] if g["name"] == "Calculated")
        lo = next(l["id"] for l in calc["labels"] if l["name"] == "Lung Opacity")
        cat = bi.train_catalog()
        sops = {cat._rows[i]["sop_uid"]: i for i in self.EXPECTED_BOXES}
        found = {i: [] for i in self.EXPECTED_BOXES}
        for a in ann["datasets"][0]["annotations"]:
            if a.get("SOPInstanceUID") in sops and a.get("labelId") == lo:
                self.assertEqual((a["width"], a["height"]), (1024, 1024))   # canvas = DICOM frame
                d = a["data"]
                found[sops[a["SOPInstanceUID"]]].append((d["x"], d["y"], d["width"], d["height"]))
        for image_id in self.EXPECTED_BOXES:
            mine = sorted((b.x, b.y, b.width, b.height) for b in cat.boxes(image_id))
            self.assertEqual(len(mine), len(found[image_id]))
            for m, j in zip(mine, sorted(found[image_id])):
                for a, c in zip(m, j):
                    self.assertAlmostEqual(a, c, places=4)

    def test_overlay_alignment_on_1024_frame(self):
        cat = bi.train_catalog()
        decoded = cat.load("00000032_007.png")
        boxes = cat.boxes("00000032_007.png")
        ov = bi.draw_ground_truth(decoded.pixel01, boxes)
        grey = (decoded.pixel01 * 255).round().astype(np.uint8)
        self.assertEqual(ov.shape, (1024, 1024, 3))
        red = (ov[..., 0] == bi.GT_RED[0]) & (ov[..., 1] == bi.GT_RED[1]) & (ov[..., 2] == bi.GT_RED[2])
        changed = ~((ov[..., 0] == grey) & (ov[..., 1] == grey) & (ov[..., 2] == grey))
        inside_any = np.zeros((1024, 1024), bool)
        for b in boxes:
            x0, y0, x1, y1 = b.pixel_bounds()
            self.assertEqual((x0, y0), (int(b.x), int(b.y)))
            # outline corners sit exactly on the box's pixel bounds ...
            for (yy, xx) in ((y0, x0), (y0, x1), (y1, x0), (y1, x1)):
                self.assertTrue(red[yy, xx], (b.number, yy, xx))
            # ... and the pixel just outside each corner is untouched
            for (yy, xx) in ((y0 - 1, x0 - 1), (y1 + 1, x1 + 1)):
                self.assertFalse(changed[yy, xx], (b.number, yy, xx))
            # full red extent of this box's outline equals its bounds
            sub = red[y0:y1 + 1, x0:x1 + 1]
            self.assertTrue(sub[0, :].all() and sub[-1, :].all() and sub[:, 0].all() and sub[:, -1].all())
            inside_any[y0:y1 + 1, x0:x1 + 1] = True
        self.assertFalse(changed[~inside_any].any())   # nothing drawn outside the boxes

    def test_no_boxes_drawn_for_negatives(self):
        cat = bi.train_catalog()
        for image_id in ("00000003_003.png", "00000003_000.png"):
            decoded = cat.load(image_id)
            ov = bi.draw_ground_truth(decoded.pixel01, cat.boxes(image_id))
            grey = (decoded.pixel01 * 255).round().astype(np.uint8)
            self.assertTrue((ov == grey[..., None]).all())

    def test_boxes_refuse_non_train(self):
        import csv
        with (bi.RESEARCH_ROOT / "data/processed/split.csv").open(encoding="utf-8-sig") as f:
            val_id = next(r["image_id"] for r in csv.DictReader(f) if r["split"] == "val")
        with self.assertRaises(PermissionError):
            bi.train_catalog().boxes(val_id)

    def test_upload_matching_requires_full_identity_and_pixels(self):
        import io as _io
        import pydicom
        cat = bi.train_catalog()
        path, _ = bi.train_reference_path("00000032_007.png")
        raw = path.read_bytes()
        s, reason = cat.match_upload(bi.decode_dicom_bytes(raw))
        self.assertEqual((s.image_id if s else None, reason), ("00000032_007.png", "matched"))
        for tag, value in (("PatientID", "someone-else"), ("SOPInstanceUID", "1.2.3.4.5"),
                           ("StudyInstanceUID", "1.2.3")):
            with self.subTest(tampered=tag):
                ds = pydicom.dcmread(_io.BytesIO(raw))
                setattr(ds, tag, value)
                buf = _io.BytesIO(); ds.save_as(buf)
                s, reason = cat.match_upload(bi.decode_dicom_bytes(buf.getvalue()))
                self.assertIsNone(s)
                self.assertNotEqual(reason, "matched")

    # ---- Regional evidence (frozen P3 TRAIN controls) ----------------------------------
    def test_region_index_matches_raw_p3_csvs(self):
        """Lesion/control geometry equals the raw frozen CSV rows for the reference image."""
        import csv
        ri = bi.region_index()
        self.assertEqual(len(ri.images_with_controls), 2421)
        les = ri.for_image(bi.REGION_REFERENCE_ID)
        self.assertEqual([(l.box_number, l.k, l.skip_reason) for l in les],
                         [(1, 0, "SEARCH_EXHAUSTED"), (2, 2, "")])
        with open(bi.RESEARCH_ROOT / "results/qa/qa_controls.csv", encoding="utf-8-sig") as f:
            raw = sorted((r for r in csv.DictReader(f) if r["image_id"] == bi.REGION_REFERENCE_ID),
                         key=lambda r: r["control_id"])
        self.assertEqual(len(raw), 2)
        for c, r in zip(les[1].controls, raw):
            self.assertEqual(c.xyxy, tuple(float(r[k]) for k in ("x0", "y0", "x1", "y1")))
            self.assertEqual((c.tier, c.side), (r["tier"], r["side"]))
            self.assertGreaterEqual(c.occupancy, 0.80)
        box2 = next(b for b in bi.train_catalog().boxes(bi.REGION_REFERENCE_ID) if b.number == 2)
        self.assertEqual(les[1].lesion_xyxy, (box2.x, box2.y, box2.x2, box2.y2))

    def test_region_scores_self_consistent_and_rem_definition(self):
        b, ri = bundle(), bi.region_index()
        iid = bi.REGION_REFERENCE_ID
        x = bi.preprocess(bi.train_catalog().load(iid).pixel01)
        preds = {p.run_id: p.logit for p in b.predict(x)}
        ev = b.region_evidence(x, iid, ri.for_image(iid), ri)   # raises on parity failure
        for run_id in bi.RUN_IDS:
            r = ev[run_id]
            self.assertAlmostEqual(r["image_logit"], preds[run_id], delta=1e-4)
            self.assertAlmostEqual(r["whole_image_region_logit"], r["image_logit"], delta=1e-4)
            k0, k2 = r["lesions"]
            self.assertEqual((k0["k"], k0["control_logits"], k0["rem"], k0["top1"]), (0, [], None, None))
            self.assertEqual(len(k2["control_logits"]), 2)
            self.assertAlmostEqual(k2["rem"], k2["lesion_logit"] - sum(k2["control_logits"]) / 2, places=6)
            self.assertEqual(k2["top1"], k2["lesion_logit"] > max(k2["control_logits"]))

    def test_region_scores_equal_independent_research_recomputation(self):
        """Recompute with the research functions directly (no demo code in the scoring path)."""
        import torch as _t
        from regions import scale_boxes_xyxy, shared_region_scores
        b, ri = bundle(), bi.region_index()
        iid = bi.REGION_REFERENCE_ID
        les = ri.for_image(iid)[1]
        x = bi.preprocess(bi.train_catalog().load(iid).pixel01)
        ev = b.region_evidence(x, iid, ri.for_image(iid), ri)
        for run_id, m in b.models.items():
            with _t.inference_mode():
                f = m.forward_features(x[None].to(b.device))
                boxes = _t.tensor([les.lesion_xyxy, *[c.xyxy for c in les.controls]],
                                  dtype=_t.float32, device=b.device)
                s = shared_region_scores(f, m.classifier, scale_boxes_xyxy(boxes),
                                         _t.zeros(3, dtype=_t.int64, device=b.device)).cpu()
            rec = ev[run_id]["lesions"][1]
            self.assertAlmostEqual(float(s[0]), rec["lesion_logit"], delta=1e-5)
            for a, c in zip(s[1:], rec["control_logits"]):
                self.assertAlmostEqual(float(a), c, delta=1e-5)

    def test_verified_examples_regions(self):
        ri = bi.region_index()
        les = ri.for_image("00000032_007.png")
        self.assertEqual([(l.box_number, l.k, l.skip_reason) for l in les],
                         [(1, 0, "IMAGE_SKIP_REL"), (2, 0, "IMAGE_SKIP_REL")])
        self.assertNotIn("00000032_007.png", ri.images_with_controls)
        for neg in ("00000003_003.png", "00000003_000.png"):
            self.assertEqual(ri.for_image(neg), ())

    def test_regions_refuse_non_train(self):
        import csv
        with (bi.RESEARCH_ROOT / "data/processed/split.csv").open(encoding="utf-8-sig") as f:
            ids = {r["split"]: r["image_id"] for r in csv.DictReader(f)}
        for part in ("val", "test"):
            with self.subTest(part=part), self.assertRaises(PermissionError):
                bi.region_index().for_image(ids[part])

    def test_region_overlay_alignment(self):
        ri, cat = bi.region_index(), bi.train_catalog()
        iid = bi.REGION_REFERENCE_ID
        pix = cat.load(iid).pixel01
        ov = bi.draw_regions(pix, ri.for_image(iid))
        grey = (pix * 255).round().astype(np.uint8)
        changed = ~((ov[..., 0] == grey) & (ov[..., 1] == grey) & (ov[..., 2] == grey))
        blue = np.all(ov == np.array(bi.CONTROL_BLUE, np.uint8), axis=-1)
        red = np.all(ov == np.array(bi.GT_RED, np.uint8), axis=-1)
        inside = np.zeros((1024, 1024), bool)
        for les in ri.for_image(iid):
            x0, y0, x1, y1 = (int(np.floor(les.lesion_xyxy[0])), int(np.floor(les.lesion_xyxy[1])),
                              int(np.ceil(les.lesion_xyxy[2])) - 1, int(np.ceil(les.lesion_xyxy[3])) - 1)
            self.assertTrue(red[y0, x0] and red[y1, x1])
            inside[y0:y1 + 1, x0:x1 + 1] = True
            for c in les.controls:
                cx0, cy0 = int(np.floor(c.xyxy[0])), int(np.floor(c.xyxy[1]))
                cx1, cy1 = int(np.ceil(c.xyxy[2])) - 1, int(np.ceil(c.xyxy[3])) - 1
                self.assertTrue(blue[cy0, cx0] and blue[cy1, cx1], c.label)   # dashes start at corners
                inside[cy0:cy1 + 1, cx0:cx1 + 1] = True
        self.assertFalse(changed[~inside].any())

    def test_invalid_inputs_rejected_gracefully(self):
        for data in (b"", b"not a dicom file at all", b"\x00" * 200):
            with self.subTest(data=data[:10]):
                with self.assertRaises(bi.DicomInputError):
                    bi.decode_dicom_bytes(data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
