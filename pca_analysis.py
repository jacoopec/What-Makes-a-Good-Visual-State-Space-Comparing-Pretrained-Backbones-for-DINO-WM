#!/usr/bin/env python3
"""Visualize the first three PCA components of one DINO-WM backbone's patch features.

Run from the root of your DINO-WM project, e.g.:
    python visualize_backbone_pca.py --encoder mae --image /path/to/frame.png
    python visualize_backbone_pca.py --encoder siglip --video /path/to/rollout.mp4 --frame 25

The script uses the project's encoder configuration and assumes the custom encoder
interface accepts image tensors in [-1, 1] and returns [B, N_patches, D].
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless SLURM nodes
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf

IMAGE_SIZE = 224


def get_image(args):
    """Read a still image or a zero-indexed video frame as RGB."""
    if args.image is not None:
        with Image.open(args.image) as im:
            image = im.convert("RGB")
    else:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("Reading MP4 files requires opencv-python.") from exc
        capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            raise RuntimeError("Could not open video: {}".format(args.video))
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        success, bgr = capture.read()
        capture.release()
        if not success:
            raise RuntimeError("Could not read frame {} of {}".format(args.frame, args.video))
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def create_encoder(name_or_config, device):
    """Instantiate the real encoder, using the same Hydra config as training."""
    path = Path(name_or_config)
    if not path.is_file():
        path = Path("conf/encoder") / (name_or_config + ".yaml")
    if not path.is_file():
        raise FileNotFoundError(
            "Encoder config not found: {}. Run from the project root or "
            "pass --encoder /absolute/path/to/config.yaml".format(path)
        )
    cfg = OmegaConf.load(str(path))
    encoder = instantiate(cfg)
    encoder = encoder.to(device).eval()
    return encoder, path.stem


def patch_grid(encoder, num_tokens):
    """Prefer metadata, but infer the grid from the actual token count."""
    grid = getattr(encoder, "grid_size", None)
    if isinstance(grid, int):
        grid = (grid, grid)
    if grid is not None and len(grid) == 2:
        h, w = int(grid[0]), int(grid[1])
        if h * w == num_tokens:
            return h, w

    side = int(np.sqrt(num_tokens))
    if side * side != num_tokens or num_tokens < 4:
        raise ValueError(
            "Expected a square grid of spatial patch tokens, got {} tokens. "
            "CLS/global embeddings cannot be spatially visualized.".format(num_tokens)
        )
    return side, side


def extract_features(encoder, rgb, device):
    # DINO-WM's *custom* encoder contract: [B, 3, H, W] in [-1, 1].
    # Resizing / backbone-specific normalization must happen in the encoder.
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
    tensor = tensor.unsqueeze(0).to(device)
    tensor = tensor * 2.0 - 1.0

    with torch.inference_mode():
        tokens = encoder(tensor)

    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3 or tokens.shape[0] != 1:
        raise ValueError(
            "Encoder must return a tensor shaped [1, N_patches, D]; "
            "got {}".format(getattr(tokens, "shape", type(tokens)))
        )
    return tokens[0].detach().float().cpu().numpy()  # [N_patches, D]


def upsample_scores(scores, height, width):
    """Turn [H_p, W_p, 3] PCA scores into a 224x224 spatial map."""
    patch_map = torch.from_numpy(scores).float().permute(2, 0, 1).unsqueeze(0)
    pixel_map = F.interpolate(
        patch_map, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    )
    return pixel_map[0].permute(1, 2, 0).numpy()


def rgb_from_scores(scores):
    """Map PC1->red, PC2->green, PC3->blue, each scaled independently."""
    low = np.percentile(scores, 2, axis=(0, 1))
    high = np.percentile(scores, 98, axis=(0, 1))
    return np.clip((scores - low) / np.maximum(high - low, 1e-8), 0, 1)


def plot_result(original, scores, explained, grid_shape, encoder_name, output, alpha):
    pca_rgb = rgb_from_scores(scores)
    overlaid = (1.0 - alpha) * original + alpha * pca_rgb

    fig, axes = plt.subplots(2, 3, figsize=(13, 9), constrained_layout=True)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original (224 x 224)")

    axes[0, 1].imshow(pca_rgb)
    axes[0, 1].set_title("PCA RGB: PC1=R, PC2=G, PC3=B")

    axes[0, 2].imshow(overlaid)
    axes[0, 2].set_title("RGB PCA overlay (alpha={:.2f})".format(alpha))

    for k, ax in enumerate(axes[1]):
        heat = scores[:, :, k]
        limit = float(np.percentile(np.abs(heat), 99))
        limit = max(limit, 1e-8)
        ax.imshow(original)
        ax.imshow(heat, cmap="coolwarm", vmin=-limit, vmax=limit, alpha=alpha)
        ax.set_title("PC{} ({:.1f}% variance)".format(k + 1, explained[k] * 100))

    fig.suptitle(
        "{} | patch grid {}x{} | explained variance (top 3): {:.1f}%".format(
            encoder_name, grid_shape[0], grid_shape[1], sum(explained) * 100
        ),
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved {}".format(output.resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoder", required=True,
        help="Config name under conf/encoder (e.g. dino, clip, siglip, mae), or YAML path",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="PNG/JPEG frame from an environment")
    source.add_argument("--video", type=Path, help="MP4 rollout; use --frame to choose frame")
    parser.add_argument("--frame", type=int, default=0, help="Zero-indexed frame (default: 0)")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--alpha", type=float, default=0.55, help="Overlay strength in [0, 1]")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path")
    args = parser.parse_args()

    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")
    if args.frame < 0:
        parser.error("--frame cannot be negative")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    rgb = get_image(args)
    encoder, name = create_encoder(args.encoder, device)
    features = extract_features(encoder, rgb, device)
    num_patches, dim = features.shape
    h, w = patch_grid(encoder, num_patches)

    # Each spatial patch is ONE observation; embedding channels are PCA variables.
    pca = PCA(n_components=3, svd_solver="full")
    scores = pca.fit_transform(features.astype(np.float64))  # [N_patches, 3]
    spatial_scores = upsample_scores(scores.reshape(h, w, 3), h, w)

    stem = (args.image if args.image is not None else args.video).stem
    output = args.output or Path("pca_outputs") / "{}_{}_pca.png".format(stem, name)
    plot_result(rgb, spatial_scores, pca.explained_variance_ratio_, (h, w), name, output, args.alpha)
    print("Backbone: {} | feature tensor: [{}, {}]".format(name, num_patches, dim))
    print("Explained variance PC1/PC2/PC3: {}".format(
        ", ".join("{:.2f}%".format(v * 100) for v in pca.explained_variance_ratio_)
    ))


if __name__ == "__main__":
    main()
