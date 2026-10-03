import os
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import ViTMAEConfig, ViTMAEModel


# ============================================================
# HARDCODE YOUR PATHS HERE
# ============================================================

DATASET_PATH = "/work/cvcs2026/DINO_WM/wall_single"

CACHE_DIR = "/homes/jpecchini/.cache/huggingface"

MODEL_NAME = "facebook/vit-mae-base"

EPISODE_INDEX = 0
FRAME_INDEX = 0


# ============================================================
# CORRECTED MAE ENCODER
# ============================================================

class MAEEncoder(nn.Module):
    def __init__(
        self,
        model_name=MODEL_NAME,
        cache_dir=CACHE_DIR,
        local_files_only=True,
    ):
        super().__init__()

        config = ViTMAEConfig.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        # We want ALL spatial patches.
        config.mask_ratio = 0.0

        self.base_model = ViTMAEModel.from_pretrained(
            model_name,
            config=config,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        self.name = model_name
        self.emb_dim = config.hidden_size       # 768
        self.patch_size = config.patch_size     # 16
        self.latent_ndim = 2

        image_size = config.image_size

        if isinstance(image_size, (tuple, list)):
            image_size = image_size[0]

        patch_size = config.patch_size

        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]

        self.num_patches = (image_size // patch_size) ** 2

        # ImageNet normalization expected by pretrained MAE
        self.register_buffer(
            "image_mean",
            torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "image_std",
            torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 3, 1, 1),
        )

        # Freeze MAE
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False


    def prepare_input(self, images):
        """
        DINO-WM input:
            approximately [-1, 1]

        MAE input:
            [0,1] -> ImageNet normalized
        """

        # DINO-WM [-1,1] -> RGB [0,1]
        images_01 = (images + 1.0) / 2.0

        # MAE / ImageNet normalization
        images_mae = (
            images_01 - self.image_mean.to(images.dtype)
        ) / self.image_std.to(images.dtype)

        return images_01, images_mae


    def forward(self, images):

        _, images_mae = self.prepare_input(images)

        batch_size = images.shape[0]

        # ----------------------------------------------------
        # IMPORTANT:
        # keep all 196 patches in their ORIGINAL spatial order
        # ----------------------------------------------------

        noise = torch.arange(
            self.num_patches,
            device=images.device,
            dtype=torch.float32,
        )

        noise = noise.unsqueeze(0).expand(
            batch_size,
            -1,
        )

        outputs = self.base_model(
            pixel_values=images_mae,
            noise=noise,
            return_dict=True,
        )

        # [B, 197, 768]
        tokens = outputs.last_hidden_state

        # Remove CLS
        # [B, 196, 768]
        patch_tokens = tokens[:, 1:, :]

        return patch_tokens, outputs.ids_restore


# ============================================================
# LOAD ONE REAL WALL IMAGE
# ============================================================

def load_wall_frame():

    episode_path = os.path.join(
        DATASET_PATH,
        "obses",
        f"episode_{EPISODE_INDEX:03d}.pth",
    )

    print("Loading:")
    print(episode_path)

    if not os.path.isfile(episode_path):
        raise FileNotFoundError(
            f"\nCould not find:\n{episode_path}\n\n"
            "Check DATASET_PATH."
        )

    episode = torch.load(
        episode_path,
        map_location="cpu",
    )

    print("\nRaw episode shape:", episode.shape)
    print("Raw dtype:", episode.dtype)
    print(
        "Raw range:",
        episode.min().item(),
        episode.max().item(),
    )

    image = episode[FRAME_INDEX]

    print("\nSelected frame shape:", image.shape)

    # --------------------------------------------------------
    # Handle possible channel-last representation
    # --------------------------------------------------------

    if image.ndim != 3:
        raise RuntimeError(
            f"Expected one image with 3 dimensions, got {image.shape}"
        )

    if image.shape[-1] == 3 and image.shape[0] != 3:
        image = image.permute(2, 0, 1)

    if image.shape[0] != 3:
        raise RuntimeError(
            f"Expected RGB image, got shape {image.shape}"
        )

    image = image.float() / 255.0

    # Same transform used by DINO-WM:
    #
    # Resize(224)
    # CenterCrop(224)
    # Normalize(0.5, 0.5)
    #
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),
        ]
    )

    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    return image


# ============================================================
# TEST
# ============================================================

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("MAE DINO-WM ENCODER TEST")
    print("=" * 70)

    print("\nDevice:", device)

    # --------------------------------------------------------
    # Load real DINO-WM image
    # --------------------------------------------------------

    images = load_wall_frame().to(device)

    print("\n" + "=" * 70)
    print("DINO-WM INPUT")
    print("=" * 70)

    print("Shape:", tuple(images.shape))
    print("Min:  ", images.min().item())
    print("Max:  ", images.max().item())
    print("Mean: ", images.mean().item())
    print("Std:  ", images.std().item())

    # We expect approximately [-1, 1]
    if images.min() < -1.01 or images.max() > 1.01:
        print("\nWARNING: image is not approximately in [-1,1]")
    else:
        print("\nPASS: DINO-WM input range looks correct.")

    # --------------------------------------------------------
    # Load MAE
    # --------------------------------------------------------

    print("\nLoading MAE...")

    encoder = MAEEncoder().to(device)
    encoder.eval()

    # --------------------------------------------------------
    # Check MAE preprocessing
    # --------------------------------------------------------

    with torch.no_grad():
        images_01, images_mae = encoder.prepare_input(images)

    print("\n" + "=" * 70)
    print("MAE PREPROCESSING")
    print("=" * 70)

    print("\nAfter [-1,1] -> [0,1]")
    print("Min: ", images_01.min().item())
    print("Max: ", images_01.max().item())
    print("Mean:", images_01.mean().item())

    print("\nAfter ImageNet normalization")
    print("Min: ", images_mae.min().item())
    print("Max: ", images_mae.max().item())
    print("Mean:", images_mae.mean().item())
    print("Std: ", images_mae.std().item())

    # --------------------------------------------------------
    # Encode SAME IMAGE TWICE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DETERMINISM TEST")
    print("=" * 70)

    with torch.no_grad():

        z1, ids_restore_1 = encoder(images)
        z2, ids_restore_2 = encoder(images)

    print("\nFeature shape:")
    print(tuple(z1.shape))

    print("\nFeature statistics:")
    print("Min:  ", z1.min().item())
    print("Max:  ", z1.max().item())
    print("Mean: ", z1.mean().item())
    print("Std:  ", z1.std().item())

    print("\nNaN present:", torch.isnan(z1).any().item())
    print("Inf present:", torch.isinf(z1).any().item())

    max_difference = (z1 - z2).abs().max().item()
    mean_difference = (z1 - z2).abs().mean().item()

    print("\nSame-image comparison:")
    print("Max difference: ", max_difference)
    print("Mean difference:", mean_difference)

    # --------------------------------------------------------
    # Check expected shape
    # --------------------------------------------------------

    expected_shape = (1, 196, 768)

    if tuple(z1.shape) == expected_shape:
        print(
            "\nPASS: feature shape is correct:",
            expected_shape,
        )
    else:
        print(
            "\nFAIL: expected",
            expected_shape,
            "but obtained",
            tuple(z1.shape),
        )

    # --------------------------------------------------------
    # Check determinism
    # --------------------------------------------------------

    if max_difference < 1e-5:
        print(
            "PASS: encoding is deterministic."
        )
    else:
        print(
            "FAIL: same image produces different MAE features!"
        )

    # --------------------------------------------------------
    # Check patch order
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PATCH ORDER TEST")
    print("=" * 70)

    expected_order = torch.arange(
        encoder.num_patches,
        device=device,
    )

    actual_order = ids_restore_1[0]

    print("\nFirst 20 ids_restore:")
    print(actual_order[:20].cpu())

    print("\nLast 20 ids_restore:")
    print(actual_order[-20:].cpu())

    patch_order_correct = torch.equal(
        actual_order,
        expected_order,
    )

    if patch_order_correct:
        print(
            "\nPASS: patches remain in original 0..195 spatial order."
        )
    else:
        print(
            "\nFAIL: MAE patches are being permuted!"
        )

    # --------------------------------------------------------
    # Overall result
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    all_ok = (
        tuple(z1.shape) == expected_shape
        and max_difference < 1e-5
        and not torch.isnan(z1).any()
        and not torch.isinf(z1).any()
        and patch_order_correct
    )

    if all_ok:
        print(
            """
PASS

The basic MAE encoder test succeeded:

  - real DINO-WM image loaded
  - output shape = [1, 196, 768]
  - no NaN / Inf
  - same image gives same features
  - patch order is 0..195

The next test should be the tiny decoder-overfitting test.
"""
        )

    else:
        print(
            """
FAIL

Do NOT start full retraining yet.

At least one of these is wrong:
  - feature shape
  - determinism
  - numerical values
  - spatial patch ordering
"""
        )


if __name__ == "__main__":
    main()