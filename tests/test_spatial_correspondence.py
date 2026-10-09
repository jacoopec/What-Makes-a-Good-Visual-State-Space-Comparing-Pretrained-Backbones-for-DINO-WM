"""Offline scientific invariants; run with unittest discovery (no weights downloaded)."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spatial_correspondence as analysis


class GeometryTests(unittest.TestCase):
    def test_rectangular_image_and_grid_boundaries(self):
        self.assertEqual(analysis.patch_at((199.9, 99.9), (100, 200), (2, 4)), (1, 3))
        self.assertEqual(analysis.patch_at((50, 50), (100, 200), (2, 4)), (1, 1))
        self.assertEqual(analysis.patch_bounds(1, 3, (100, 200), (2, 4)), (150, 50, 200, 100))
        self.assertEqual(analysis.patch_center(1, 3, (100, 200), (2, 4)), (175, 75))
        for point in ((200, 0), (0, 100), (-1, 0), (float("nan"), 0)):
            with self.assertRaises(ValueError):
                analysis.patch_at(point, (100, 200), (2, 4))

    def test_raster_order_and_fixed_reference_during_motion(self):
        first = torch.eye(6)
        reference = first[1].clone()
        moved = first[[0, 2, 3, 4, 5, 1]]
        scores = analysis.cosine_map(reference, moved, (2, 3))
        self.assertEqual(np.unravel_index(scores.argmax(), scores.shape), (1, 2))
        self.assertEqual(float(scores[0, 1]), 0.0)
        self.assertEqual(float(scores[1, 2]), 1.0)
        torch.testing.assert_close(reference, first[1])

    def test_negative_cosine_and_upsampling_are_not_minmax_scaled(self):
        scores = analysis.cosine_map(torch.tensor([1., 0.]),
                                     torch.tensor([[-1., 0.], [0., 1.], [1., 0.], [.6, .8]]), (2, 2))
        np.testing.assert_allclose(scores, [[-1, 0], [1, .6]], atol=1e-6)
        pixels = analysis.upsample(scores, (10, 20))
        self.assertEqual(pixels.shape, (10, 20))
        self.assertAlmostEqual(float(pixels[-1, -1]), .6, places=6)

    def test_optional_metrics_and_point_error(self):
        scores = np.array([[0., .25], [-.5, 1.]], dtype=np.float32)
        pixels = analysis.upsample(scores, (20, 40))
        plain = analysis.measure(scores, pixels, (20, 40), {})
        self.assertEqual((plain["predicted_x"], plain["predicted_y"]), (30, 15))
        self.assertEqual(plain["temporal_error_pixels"], "")
        region = np.zeros((20, 40), dtype=bool)
        region[10:, 20:] = True
        result = analysis.measure(scores, pixels, (20, 40), {
            "region": region, "region_type": "mask", "correspondence_xy": [27, 11]})
        self.assertEqual(result["object_localization_hit"], 1)
        self.assertEqual(result["temporal_error_pixels"], 5)
        self.assertGreater(result["object_background_contrast"], 0)
        empty = analysis.measure(scores, pixels, (20, 40), {"region": region & False, "region_type": "mask"})
        self.assertEqual(empty["object_background_contrast"], "")

    def test_reject_global_tokens_and_unknown_overrides(self):
        with self.assertRaisesRegex(ValueError, "CLS"):
            analysis.load_encoder_config("dino_cls", [])
        with self.assertRaisesRegex(ValueError, "not a supported spatial"):
            analysis.load_encoder_config("resnet", [])


class RecordingEncoder(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.kind = kind
        self.name = kind
        self.patch_size = 2
        self.input_size = 4
        self.emb_dim = 3
        self.visual = SimpleNamespace(input_resolution=4)
        self.clip_mean = torch.tensor([.48, .45, .40])
        self.clip_std = torch.tensor([.26, .26, .27])
        config = SimpleNamespace(image_size=4, patch_size=2, hidden_size=3, to_dict=lambda: {})
        self.base_model = SimpleNamespace(config=config)
        self.last_input = None

    def forward(self, x):
        self.last_input = x.clone()
        return torch.ones(1, 4, 3, device=x.device)


class AdapterTests(unittest.TestCase):
    def test_wrapper_inputs_shapes_and_freezing(self):
        from omegaconf import OmegaConf
        for kind in ("dino", "clip", "siglip", "ijepa"):
            with self.subTest(kind=kind):
                encoder = RecordingEncoder(kind)
                with patch("hydra.utils.instantiate", return_value=encoder):
                    adapter = analysis.SpatialEncoder(OmegaConf.create({}), kind, torch.device("cpu"))
                tokens = adapter.encode(np.full((7, 11, 3), 255, dtype=np.uint8))
                self.assertEqual(tuple(tokens.shape), (4, 3))
                self.assertFalse(encoder.training)
                self.assertFalse(encoder.weight.requires_grad)
                self.assertEqual(tuple(encoder.last_input.shape), (1, 3, 4, 4))
                if kind == "dino":
                    expected = (torch.ones(3) - torch.tensor(analysis.IMAGENET_MEAN)) / torch.tensor(analysis.IMAGENET_STD)
                    torch.testing.assert_close(encoder.last_input[0, :, 0, 0], expected)
                else:
                    torch.testing.assert_close(encoder.last_input, torch.ones_like(encoder.last_input))

    def test_mae_bypasses_random_projection_removes_cls_and_preserves_order(self):
        from omegaconf import OmegaConf
        encoder = RecordingEncoder("mae")
        encoder.register_buffer("image_mean", torch.tensor(analysis.IMAGENET_MEAN).view(1, 3, 1, 1))
        encoder.register_buffer("image_std", torch.tensor(analysis.IMAGENET_STD).view(1, 3, 1, 1))
        calls = []

        class Base(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config

            def forward(self, pixel_values, noise, return_dict):
                calls.append(noise.clone())
                patches = torch.arange(1, 13, dtype=torch.float32).reshape(1, 4, 3)
                return SimpleNamespace(last_hidden_state=torch.cat([torch.full((1, 1, 3), -99.), patches], 1))

        encoder.base_model = Base(encoder.base_model.config)
        with patch("hydra.utils.instantiate", return_value=encoder):
            adapter = analysis.SpatialEncoder(OmegaConf.create({}), "mae", torch.device("cpu"))
        tokens = adapter.encode(np.zeros((4, 4, 3), dtype=np.uint8))
        torch.testing.assert_close(tokens, torch.arange(1, 13, dtype=torch.float32).reshape(4, 3))
        torch.testing.assert_close(calls[0], torch.arange(4, dtype=torch.float32)[None])
        self.assertIsNone(encoder.last_input)  # wrapper's trainable projection was never reached

    def test_unexpected_special_token_count_is_rejected(self):
        from omegaconf import OmegaConf
        encoder = RecordingEncoder("dino")
        encoder.forward = lambda x: torch.ones(1, 5, 3)
        with patch("hydra.utils.instantiate", return_value=encoder):
            adapter = analysis.SpatialEncoder(OmegaConf.create({}), "dino", torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "Expected spatial tokens"):
            adapter.encode(np.zeros((4, 4, 3), dtype=np.uint8))


class PipelineTests(unittest.TestCase):
    def test_end_to_end_variable_frame_sizes_and_annotation_files(self):
        class FakeAdapter:
            def __init__(self, cfg, kind, device):
                self.grid_hw, self.dim = (2, 2), 4
                self.metadata = {}
                self.count = 0

            def encode(self, rgb):
                self.count += 1
                return torch.eye(4) if self.count == 1 else torch.eye(4)[[1, 2, 3, 0]]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "images"
            source.mkdir()
            Image.new("RGB", (20, 10)).save(source / "frame2.png")
            Image.new("RGB", (40, 20)).save(source / "frame10.png")
            annotations = root / "annotations.json"
            annotations.write_text(json.dumps({"reference_xy": [2, 2], "frames": {
                "0": {"correspondence_xy": [2, 2]},
                "1": {"bbox_xyxy": [20, 10, 40, 20], "correspondence_xy": [30, 15]}}}))
            args = analysis.parse_args(["--source-type", "images", "--source", str(source),
                                        "--frames", "0", "1", "--reference", "2", "2",
                                        "--encoders", "dino", "--annotations", str(annotations),
                                        "--output", str(root / "output")])
            with patch.object(analysis, "SpatialEncoder", FakeAdapter):
                analysis.run(args)
            output = root / "output"
            for name in ("reference_patch.png", "comparison.png", "metadata.json",
                         "dino/spatial_similarity.png", "dino/frame_000001_heatmap.png"):
                self.assertTrue((output / name).is_file(), name)
            data = json.loads((output / "metadata.json").read_text())
            self.assertEqual(data["status"], "complete")
            with (output / "correspondence_metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[1]["experiment"], "fixed_reference_cross_frame")
            self.assertEqual(float(rows[1]["temporal_error_pixels"]), 0.)
            self.assertEqual(rows[1]["object_localization_hit"], "1")
            self.assertEqual(np.load(output / "dino/frame_000001_similarity_pixels.npy").shape, (20, 40))


if __name__ == "__main__":
    unittest.main()
