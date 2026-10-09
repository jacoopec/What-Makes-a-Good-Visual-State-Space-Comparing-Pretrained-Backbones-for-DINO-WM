#!/usr/bin/env python3
"""Headless spatial similarity and fixed-reference temporal correspondence.

See SPATIAL_CORRESPONDENCE.md for preprocessing, annotations and examples.
No world model or trained projection checkpoint is used.
"""

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
TARGETS = {
    "models.dino.DinoV2Encoder": "dino",
    "models.encoder.clip_encoder.CLIPVisionEncoder": "clip",
    "models.encoder.siglip.SigLIPEncoder": "siglip",
    "models.encoder.mae.MAEEncoder": "mae",
    "models.encoder.ijepa.IJEPAEncoder": "ijepa",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pair(value):
    return (int(value), int(value)) if isinstance(value, int) else tuple(map(int, value))


def natural_key(path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path.name)]


def load_frames(args):
    """Keep only selected original RGB frames on CPU; dataset transforms are disabled."""
    source = Path(args.source).expanduser().resolve()
    info = {"kind": args.source_type, "path": str(source), "trajectory": args.trajectory}
    if args.source_type == "pusht":
        from datasets.pusht_dset import PushTDataset
        dataset = PushTDataset(
            data_path=str(source), n_rollout=args.trajectory + 1,
            transform=None, normalize_action=False, with_velocity=False,
        )
        if args.trajectory >= len(dataset):
            raise ValueError("Trajectory index is outside the PushT dataset")
        if max(args.frames) >= dataset.get_seq_length(args.trajectory):
            raise ValueError("Requested frame is beyond the selected trajectory")
        obs, _, _, _ = dataset.get_frames(args.trajectory, args.frames)
        rgb = obs["visual"].permute(0, 2, 3, 1).cpu().numpy()
        images = [np.rint(x * 255).clip(0, 255).astype(np.uint8) for x in rgb]
        info["dataset_class"] = "datasets.pusht_dset.PushTDataset"
    elif args.source_type == "video":
        import cv2
        capture = cv2.VideoCapture(str(source))
        images = []
        try:
            if not capture.isOpened():
                raise ValueError("Cannot open video: {}".format(source))
            for frame in args.frames:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, bgr = capture.read()
                if not ok:
                    raise ValueError("Cannot read video frame {}".format(frame))
                images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
    else:
        files = sorted((p for p in source.iterdir() if p.suffix.lower() in
                        {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}),
                       key=natural_key) if source.is_dir() else [source]
        if max(args.frames) >= len(files):
            raise ValueError("Frame indices exceed the number of images ({})".format(len(files)))
        selected = [files[i] for i in args.frames]
        images = []
        for path in selected:
            with Image.open(path) as image:
                images.append(np.asarray(image.convert("RGB")).copy())
        info["files"] = list(map(str, selected))
    info["frames"] = [{"index": i, "height": im.shape[0], "width": im.shape[1],
                       "rgb_sha256": hashlib.sha256(im.tobytes()).hexdigest()}
                      for i, im in zip(args.frames, images)]
    return images, info


def load_encoder_config(name, overrides):
    from omegaconf import OmegaConf
    path = Path(name)
    if not path.is_file():
        path = ROOT / "conf" / "encoder" / (name + ".yaml")
    cfg = OmegaConf.load(path)
    label = path.stem
    for override in overrides:
        owner, assignment = override.split(":", 1)
        if owner == label:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([assignment]))
    target = cfg.get("_target_")
    if target not in TARGETS:
        raise ValueError("{} is not a supported spatial encoder; global/CLS/dummy features "
                         "cannot be mapped to image patches".format(target))
    kind = TARGETS[target]
    if kind == "dino" and cfg.get("feature_key") != "x_norm_patchtokens":
        raise ValueError("DINO requires feature_key=x_norm_patchtokens; CLS is not spatial")
    return cfg, label, kind, path.resolve()


class SpatialEncoder:
    """Thin analysis adapter around the repository's Hydra-instantiated encoders."""

    def __init__(self, cfg, kind, device):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        self.kind = kind
        self.device = device
        self.encoder = instantiate(cfg).eval().requires_grad_(False).to(device)
        enc = self.encoder
        self.patch_hw = pair(enc.patch_size)
        if kind == "clip":
            self.input_hw = pair(enc.visual.input_resolution)
        elif kind == "siglip":
            self.input_hw = pair(enc.base_model.config.image_size)
        elif kind == "mae":
            self.input_hw = pair(enc.base_model.config.image_size)
            self.patch_hw = pair(enc.base_model.config.patch_size)
        else:
            self.input_hw = pair(enc.input_size)
        if any(s <= 0 or s % p for s, p in zip(self.input_hw, self.patch_hw)):
            raise ValueError("Input dimensions must be positive multiples of the patch size")
        self.grid_hw = tuple(s // p for s, p in zip(self.input_hw, self.patch_hw))
        self.dim = int(enc.base_model.config.hidden_size if kind == "mae" else enc.emb_dim)
        mean = IMAGENET_MEAN if kind in ("dino", "mae") else (0.5,) * 3
        std = IMAGENET_STD if kind in ("dino", "mae") else (0.5,) * 3
        if kind == "clip":
            mean = enc.clip_mean.flatten().tolist()
            std = enc.clip_std.flatten().tolist()
        self.metadata = {
            "hydra_config": OmegaConf.to_container(cfg, resolve=True),
            "model_name": enc.name, "adapter": kind,
            "input_hw": self.input_hw, "patch_hw": self.patch_hw,
            "grid_hw": self.grid_hw, "feature_dimension": self.dim,
            "effective_normalization_mean": mean, "effective_normalization_std": std,
            "resize": "full frame, no crop, PIL " + ("bilinear" if kind in ("mae", "ijepa") else "bicubic"),
            "feature_stage": "pretrained MAE patches before trainable projection" if kind == "mae"
                else "repository encoder patch output",
            "special_tokens": "MAE CLS removed explicitly; other wrappers return patch tokens only",
            "frozen": True,
        }
        base = getattr(enc, "base_model", None)
        if base is not None and hasattr(base, "config"):
            self.metadata["pretrained_model_config"] = base.config.to_dict()

    @torch.inference_mode()
    def encode(self, rgb):
        # Full-image resize intentionally avoids unobserved margins from a center crop.
        resample = Image.Resampling.BILINEAR if self.kind in ("mae", "ijepa") else Image.Resampling.BICUBIC
        image = Image.fromarray(rgb).resize(self.input_hw[::-1], resample)
        x = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0)
        x = x.to(self.device, dtype=torch.float32) / 255.0
        enc = self.encoder
        if self.kind == "dino":
            mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
            std = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
            tokens = enc((x - mean) / std)
        elif self.kind == "mae":
            # Reuse the already loaded backbone and its wrapper's normalization buffers.
            # Do not use enc.proj: its weights are random without a WM checkpoint.
            x = (x - enc.image_mean) / enc.image_std
            n = self.grid_hw[0] * self.grid_hw[1]
            noise = torch.arange(n, device=self.device, dtype=x.dtype).unsqueeze(0)
            output = enc.base_model(pixel_values=x, noise=noise, return_dict=True)
            if output.last_hidden_state.shape[1] != n + 1:
                raise ValueError("MAE must return one CLS plus every unmasked spatial patch")
            # Increasing noise keeps MAE's shuffle in raster order, even at mask_ratio=0.
            tokens = output.last_hidden_state[:, 1:, :]
        else:
            # CLIP converts this to its own normalization; SigLIP and I-JEPA use it as-is.
            tokens = enc(x * 2.0 - 1.0)
        expected = (1, self.grid_hw[0] * self.grid_hw[1], self.dim)
        if not isinstance(tokens, torch.Tensor) or tuple(tokens.shape) != expected:
            raise ValueError("Expected spatial tokens {}, got {}".format(expected, getattr(tokens, "shape", None)))
        if not torch.isfinite(tokens).all() or (tokens.norm(dim=-1) <= 1e-12).any():
            raise ValueError("Encoder produced nonfinite or zero-norm patch features")
        return tokens[0].float().cpu()


def patch_at(xy, image_hw, grid_hw):
    """Coordinates use image-edge axes: top-left edge (0,0), centers (x+.5,y+.5)."""
    x, y = xy
    h, w = image_hw
    gh, gw = grid_hw
    if not np.isfinite([x, y]).all() or not (0 <= x < w and 0 <= y < h):
        raise ValueError("Reference (x,y) must be inside the original first image")
    return int(y * gh / h), int(x * gw / w)


def patch_bounds(row, col, image_hw, grid_hw):
    h, w = image_hw
    gh, gw = grid_hw
    return (col * w / gw, row * h / gh, (col + 1) * w / gw, (row + 1) * h / gh)


def patch_center(row, col, image_hw, grid_hw):
    x0, y0, x1, y1 = patch_bounds(row, col, image_hw, grid_hw)
    return (x0 + x1) / 2, (y0 + y1) / 2


def cosine_map(reference, tokens, grid_hw):
    if reference.ndim != 1 or tokens.ndim != 2 or tokens.shape[1] != len(reference):
        raise ValueError("Reference and target feature dimensions must agree")
    scores = F.normalize(tokens, dim=-1) @ F.normalize(reference, dim=0)
    return scores.clamp(-1, 1).reshape(grid_hw).numpy()


def upsample(scores, image_hw):
    return F.interpolate(torch.from_numpy(scores)[None, None], size=image_hw,
                         mode="bilinear", align_corners=False)[0, 0].numpy()


def load_annotations(path, frames, images):
    """Optional JSON with per-frame masks/boxes and genuine point correspondences."""
    if path is None:
        return {}, {}
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for frame, image in zip(frames, images):
        item = dict(data.get("frames", {}).get(str(frame), {}))
        h, w = image.shape[:2]
        if "mask" in item:
            with Image.open(path.parent / item["mask"]) as mask:
                region = np.asarray(mask.convert("L")) > 0
            if region.shape != (h, w):
                raise ValueError("Object mask must match original frame resolution")
            item["region"] = region
            item["region_type"] = "mask"
        elif "bbox_xyxy" in item:
            x0, y0, x1, y1 = item["bbox_xyxy"]
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise ValueError("Box must have positive area inside the original image")
            yy, xx = np.mgrid[:h, :w] + 0.5
            item["region"] = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
            item["region_type"] = "bbox_proxy"
        if "correspondence_xy" in item:
            point = np.asarray(item["correspondence_xy"], dtype=float)
            if point.shape != (2,) or not np.isfinite(point).all():
                raise ValueError("correspondence_xy must be a finite [x,y] point")
        result[frame] = item
    return result, data


def measure(scores, pixel_map, image_hw, annotation):
    row, col = np.unravel_index(np.argmax(scores), scores.shape)
    x, y = patch_center(row, col, image_hw, scores.shape)
    result = {"max_cosine_similarity": float(scores[row, col]),
              "predicted_x": x, "predicted_y": y,
              "predicted_patch_row": int(row), "predicted_patch_col": int(col),
              "region_type": "", "object_localization_hit": "",
              "object_background_contrast": "", "temporal_error_pixels": ""}
    region = annotation.get("region")
    if region is not None:
        result["region_type"] = annotation["region_type"]
        if annotation["region_type"] == "bbox_proxy":
            x0, y0, x1, y1 = annotation["bbox_xyxy"]
            result["object_localization_hit"] = int(x0 <= x < x1 and y0 <= y < y1)
        else:
            result["object_localization_hit"] = int(region[int(y), int(x)])
        if region.any() and (~region).any():
            result["object_background_contrast"] = float(pixel_map[region].mean() - pixel_map[~region].mean())
    if "correspondence_xy" in annotation:
        result["temporal_error_pixels"] = float(np.linalg.norm(np.array([x, y]) - annotation["correspondence_xy"]))
    return result


def show_rgb(ax, rgb):
    h, w = rgb.shape[:2]
    ax.imshow(rgb, extent=(0, w, h, 0))
    ax.set_xticks([])
    ax.set_yticks([])


def show_heat(ax, heat, args, alpha=1.0):
    h, w = heat.shape
    return ax.imshow(heat, extent=(0, w, h, 0), cmap="coolwarm",
                     vmin=args.vmin, vmax=args.vmax, alpha=alpha)


def mark_reference(ax, bounds, xy):
    x0, y0, x1, y1 = bounds
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=1.5))
    ax.plot(*xy, marker="+", color="black", markersize=9)


def save_spatial(rgb, heat, bounds, label, args, output):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), constrained_layout=True)
    titles = ["Original RGB", "Reference patch", "Within-frame cosine", "Similarity overlay"]
    for ax, title in zip(axes, titles):
        show_rgb(ax, rgb)
        ax.set_title(title)
    mark_reference(axes[1], bounds, args.reference)
    artist = show_heat(axes[2], heat, args)
    show_heat(axes[3], heat, args, args.alpha)
    mark_reference(axes[3], bounds, args.reference)
    fig.colorbar(artist, ax=list(axes[2:]), label="Cosine similarity", shrink=0.8)
    fig.suptitle("{} | reference frame {}".format(label, args.frames[0]))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_summary(images, labels, all_maps, rows, reference_bounds, args, output):
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4), squeeze=False, constrained_layout=True)
    for ax, label, bounds in zip(axes[0], labels, reference_bounds):
        show_rgb(ax, images[0])
        mark_reference(ax, bounds, args.reference)
        ax.set_title(label)
    fig.suptitle("Reference patch in frame {}\nRequested (x,y)={}".format(args.frames[0], args.reference), fontsize=11)
    fig.savefig(output / "reference_patch.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(len(labels), len(images), figsize=(3.5 * len(images), 3.5 * len(labels)),
                             squeeze=False, constrained_layout=True)
    for i, (label, maps) in enumerate(zip(labels, all_maps)):
        for j, (frame, rgb, heat) in enumerate(zip(args.frames, images, maps)):
            ax = axes[i, j]
            show_rgb(ax, rgb)
            show_heat(ax, heat, args, args.alpha)
            metric = rows[i * len(images) + j]
            ax.plot(metric["predicted_x"], metric["predicted_y"], "x", color="black")
            if j == 0:
                mark_reference(ax, reference_bounds[i], args.reference)
                ax.set_ylabel(label)
            ax.set_title("t={}{}\nmax={:.3f}".format(frame, " (reference)" if j == 0 else "",
                                                    metric["max_cosine_similarity"]))
    # An opaque colorbar describes numerical values independently of RGB overlay alpha.
    scale = plt.cm.ScalarMappable(norm=plt.Normalize(args.vmin, args.vmax), cmap="coolwarm")
    fig.colorbar(scale, ax=list(axes.flat), label="Cosine to fixed reference feature", shrink=0.8)
    fig.suptitle("Fixed-reference temporal correspondence\nBlack x: maximum-similarity patch", fontsize=11)
    fig.savefig(output / "comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(args):
    images, source_info = load_frames(args)
    patch_at(args.reference, images[0].shape[:2], (1, 1))  # validate before loading weights
    annotations, annotation_data = load_annotations(args.annotations, args.frames, images)
    if any("correspondence_xy" in v for v in annotations.values()):
        declared = np.asarray(annotation_data.get("reference_xy", []), dtype=float)
        if declared.shape != (2,) or not np.allclose(declared, args.reference, atol=1e-6, rtol=0):
            raise ValueError("Point annotations must declare reference_xy matching --reference")
        first_point = annotations.get(args.frames[0], {}).get("correspondence_xy")
        if first_point is None or not np.allclose(first_point, args.reference, atol=1e-6, rtol=0):
            raise ValueError("The reference frame must annotate the same reference point")
    configs = [load_encoder_config(name, args.encoder_override) for name in args.encoders]
    labels = [entry[1] for entry in configs]
    if len(set(labels)) != len(labels):
        raise ValueError("Encoder config stems must be unique (used as output labels)")
    for override in args.encoder_override:
        if override.split(":", 1)[0] not in labels:
            raise ValueError("Override does not name a selected encoder: " + override)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use an empty output directory to prevent mixing experiments")
    (output / "observations").mkdir()
    for frame, rgb in zip(args.frames, images):
        Image.fromarray(rgb).save(output / "observations" / "frame_{:06d}.png".format(frame))
    device = torch.device(args.device)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("Device must be CPU or CUDA")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)
        if args.cuda_memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device)
    metadata = {
        "arguments": vars(args), "source": source_info,
        "reference_frame": args.frames[0], "reference_xy": args.reference,
        "experiment": "fixed initial reference feature vs every selected frame; first column is self-similarity",
        "coordinate_convention": "original image edge coordinates; pixel centers at (x+0.5,y+0.5)",
        "contrast_definition": "mean bilinearly upsampled similarity in object region minus complement",
        "temporal_error_definition": "patch-center distance to annotated physical reference point; includes patch quantization",
        "annotations": annotation_data, "encoders": [],
        "warnings": ["Raw cosine magnitudes are not comparable quality scores across architectures.",
                     "A transformer patch has global context; its grid cell is not its full receptive field.",
                     "No object center is used as point ground truth; region complement includes agent and target."],
        "versions": {}, "status": "running",
    }
    for package in ("torch", "torchvision", "numpy", "matplotlib", "hydra-core", "transformers", "Pillow"):
        try:
            metadata["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows, all_maps, reference_bounds = [], [], []
    for cfg, label, kind, config_path in configs:
        print("Loading {} ({})".format(label, kind), flush=True)
        adapter = SpatialEncoder(cfg, kind, device)
        folder = output / label
        folder.mkdir()
        row, col = patch_at(args.reference, images[0].shape[:2], adapter.grid_hw)
        bounds = patch_bounds(row, col, images[0].shape[:2], adapter.grid_hw)
        reference_bounds.append(bounds)
        maps = []
        reference = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for index, (frame, rgb) in enumerate(zip(args.frames, images)):
            tokens = adapter.encode(rgb)
            if reference is None:
                reference = tokens[row * adapter.grid_hw[1] + col].clone()
                np.save(folder / "reference_feature.npy", reference.numpy())
            scores = cosine_map(reference, tokens, adapter.grid_hw)
            heat = upsample(scores, rgb.shape[:2])
            maps.append(heat)
            record = {"encoder": label, "frame": frame, "reference_frame": args.frames[0],
                      "experiment": "within_frame_reference" if index == 0 else "fixed_reference_cross_frame",
                      "grid_height": adapter.grid_hw[0], "grid_width": adapter.grid_hw[1],
                      "feature_dimension": adapter.dim}
            record.update(measure(scores, heat, rgb.shape[:2], annotations.get(frame, {})))
            rows.append(record)
            stem = "frame_{:06d}".format(frame)
            np.save(folder / (stem + "_similarity.npy"), scores)
            np.save(folder / (stem + "_similarity_pixels.npy"), heat)
            fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
            show_rgb(ax, rgb)
            artist = show_heat(ax, heat, args)
            ax.plot(record["predicted_x"], record["predicted_y"], "x", color="black")
            ax.set_title("{} | t={} | max={:.3f}".format(label, frame, record["max_cosine_similarity"]))
            fig.colorbar(artist, ax=ax, label="Cosine similarity", shrink=0.8)
            fig.savefig(folder / (stem + "_heatmap.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)
            if index == 0:
                save_spatial(rgb, heat, bounds, label, args, folder / "spatial_similarity.png")
            print("  frame {}: {} tokens x {}; max {:.4f} at ({:.2f}, {:.2f})".format(
                frame, len(tokens), adapter.dim, record["max_cosine_similarity"],
                record["predicted_x"], record["predicted_y"]), flush=True)
            del tokens
        adapter.metadata.update({"label": label, "config_path": str(config_path),
                                 "reference_patch_row_col": [row, col], "reference_patch_bounds_xyxy": bounds})
        if device.type == "cuda":
            adapter.metadata["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        metadata["encoders"].append(adapter.metadata)
        all_maps.append(maps)
        # Release each backbone before constructing the next; no dataset/features live on GPU.
        del adapter, reference
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        with (output / "correspondence_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    save_summary(images, labels, all_maps, rows, reference_bounds, args, output)
    metadata["status"] = "complete"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved analysis to {}".format(output), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-type", choices=("pusht", "images", "video"), required=True)
    parser.add_argument("--source", required=True, help="PushT split directory, RGB image/directory, or video")
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 5, 10, 20], help="Increasing original frame indices; first is reference")
    parser.add_argument("--reference", nargs=2, type=float, required=True, metavar=("X", "Y"))
    parser.add_argument("--encoders", nargs="+", default=["dino", "clip_encoder", "siglip", "mae", "ijepa"], help="conf/encoder names or YAML paths")
    parser.add_argument("--encoder-override", action="append", default=[], metavar="NAME:KEY=VALUE")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--annotations", help="Optional JSON: object masks/boxes and tracked reference points")
    parser.add_argument("--vmin", type=float, default=-1.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--cuda-memory-fraction", type=float, help="Optional PyTorch allocator cap (0,1]; not total process VRAM")
    args = parser.parse_args(argv)
    if args.trajectory < 0 or args.frames[0] < 0 or args.frames != sorted(set(args.frames)):
        parser.error("Trajectory must be nonnegative and frames strictly increasing, nonnegative")
    if not (-1 <= args.vmin < args.vmax <= 1) or not 0 <= args.alpha <= 1:
        parser.error("Require -1 <= vmin < vmax <= 1 and 0 <= alpha <= 1")
    if args.cuda_memory_fraction is not None and not 0 < args.cuda_memory_fraction <= 1:
        parser.error("CUDA memory fraction must be in (0,1]")
    for override in args.encoder_override:
        if ":" not in override or "=" not in override.split(":", 1)[1]:
            parser.error("Encoder overrides use NAME:KEY=VALUE")
    return args


if __name__ == "__main__":
    run(parse_args())
