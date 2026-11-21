"""Unit tests for frozen P3-to-P4 REL bridge (no medical images needed)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rel_adapter import FrozenControlIndex, LesionControls, pair_scores_from_frozen, loss_from_frozen_train_batch, _box, _read_verified_csv


def synthetic_index() -> FrozenControlIndex:
    image_groups = {
        "matched.png": (
            LesionControls("A1", (101.2, 152.4, 330.7, 400.9), ((550.1, 152.2, 768.4, 401.8), (560.2, 430.7, 765.4, 625.6))),
            LesionControls("A2", (180, 600, 240, 730), ((630.4, 600, 690.4, 730),)),
        ),
    }
    return FrozenControlIndex(image_groups, {"matched.png":1,"k0.png":1,"negative.png":0}, 2, 1, 3, 1)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.index = synthetic_index()
        self.ids = ["matched.png", "k0.png", "negative.png"]
        self.targets = torch.tensor([1.0, 1.0, 0.0])
        torch.manual_seed(17)
        self.features = torch.randn(3, 5, 4, 4, requires_grad=True)
        self.classifier = nn.Linear(5, 1)

    def test_matched_one_and_two_controls(self):
        paired = pair_scores_from_frozen(self.index, self.ids, self.targets, self.features, self.classifier)
        self.assertEqual([len(s) for s in paired], [2, 0, 0])
        self.assertEqual([p[1].numel() for p in paired[0]], [2, 1])

    def test_combined_loss_and_gradients(self):
        parts = loss_from_frozen_train_batch(self.index, self.ids, self.targets, self.features, self.classifier, lambda_value=.1)
        self.assertEqual((parts.eligible_images, parts.eligible_lesions), (1, 2))
        self.assertTrue(bool(torch.isfinite(parts.total)))
        parts.total.backward()
        self.assertTrue(bool(torch.isfinite(self.features.grad).all()))
        self.assertGreater(float(self.features.grad.abs().sum()), 0)
        self.assertTrue(bool(torch.isfinite(self.classifier.weight.grad).all()))

    def test_k0_and_negative_still_receive_bce_gradients(self):
        empty = FrozenControlIndex({}, self.index.train_targets, 0, 0, 0, 0)
        feats = self.features.detach().clone().requires_grad_()
        parts = loss_from_frozen_train_batch(empty, self.ids, self.targets, feats, self.classifier, lambda_value=.1)
        self.assertEqual(parts.eligible_images, 0)
        self.assertEqual(parts.eligible_lesions, 0)
        self.assertEqual(float(parts.rel.detach()), 0.0)
        parts.total.backward()
        self.assertTrue(bool((feats.grad[1].abs().sum() > 0) & (feats.grad[2].abs().sum() > 0)))

    def test_lambda_zero_exact_bce(self):
        feats = self.features.detach().clone().requires_grad_()
        parts = loss_from_frozen_train_batch(self.index, self.ids, self.targets, feats, self.classifier, lambda_value=0.0)
        direct = F.binary_cross_entropy_with_logits(self.classifier(feats.mean(dim=(2,3))).flatten(), self.targets)
        self.assertTrue(torch.equal(parts.total, direct))
        grad_parts = torch.autograd.grad(parts.total, feats, retain_graph=True)[0]
        grad_direct = torch.autograd.grad(direct, feats)[0]
        self.assertTrue(torch.equal(grad_parts, grad_direct))
        self.assertEqual(parts.eligible_images, 0)

    def test_unknown_or_non_train_image_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-TRAIN"):
            pair_scores_from_frozen(self.index, ["val.png"], torch.tensor([1.]), self.features[:1], self.classifier)

    def test_bad_target_rejected(self):
        with self.assertRaisesRegex(ValueError, "disagrees"):
            pair_scores_from_frozen(self.index, self.ids, torch.tensor([0.,1.,0.]), self.features, self.classifier)

    def test_duplicate_images_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            pair_scores_from_frozen(self.index, ["matched.png"]*3, self.targets, self.features, self.classifier)

    def test_unmatched_metadata_returns_empty_pairs(self):
        pairs = pair_scores_from_frozen(self.index, ["k0.png", "negative.png"], torch.tensor([1.,0.]), self.features[:2], self.classifier)
        self.assertEqual(pairs, [[], []])

    def test_float_xyxy_and_fractional_coordinates(self):
        b = _box({"x0":"0.25","y0":"1.75","x1":"30.0625","y1":"90.5"}, lesion=False)
        self.assertAlmostEqual(b[0], .25)
        self.assertAlmostEqual(b[3], 90.5)

    def test_invalid_box_outside_bounds(self):
        with self.assertRaises(ValueError):
            _box({"x0":"0","y0":"20","x1":"1100","y1":"30"}, lesion=False)

    def test_frozen_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"wrong.csv"
            path.write_text("a,b\n1,2\n")
            with self.assertRaisesRegex(ValueError,"SHA-256"):
                _read_verified_csv(path, "0"*64)

    def test_invalid_lambda_rejected(self):
        for value in (-.1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                loss_from_frozen_train_batch(self.index, self.ids, self.targets, self.features, self.classifier,lambda_value=value)

    def test_image_batch_dimensions_must_match(self):
        with self.assertRaises(ValueError):
            pair_scores_from_frozen(self.index, self.ids, self.targets, self.features[:2], self.classifier)


if __name__ == '__main__': unittest.main()
