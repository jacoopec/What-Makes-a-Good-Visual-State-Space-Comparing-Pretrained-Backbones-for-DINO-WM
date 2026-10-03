import os
import torch
import torch.nn as nn

from torchvision import transforms, utils
from transformers import ViTMAEConfig, ViTMAEModel

from models.vqvae import VQVAE


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = "/work/cvcs2026/DINO_WM/wall_single"

CACHE_DIR = "/homes/jpecchini/.cache/huggingface"
MODEL_NAME = "facebook/vit-mae-base"

EPISODE_INDEX = 0

# Only a few images: we WANT the decoder to memorize them.
N_IMAGES = 4

# This should be enough to see whether it can overfit.
NUM_STEPS = 1500

LEARNING_RATE = 3e-4

SAVE_EVERY = 100

OUTPUT_DIR = "mae_decoder_overfit_test"


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

        # Keep every patch
        config.mask_ratio = 0.0

        self.base_model = ViTMAEModel.from_pretrained(
            model_name,
            config=config,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        self.emb_dim = config.hidden_size       # 768
        self.patch_size = config.patch_size     # 16
        self.latent_ndim = 2

        image_size = config.image_size
        patch_size = config.patch_size

        if isinstance(image_size, (tuple, list)):
            image_size = image_size[0]

        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]

        self.num_patches = (
            image_size // patch_size
        ) ** 2                                # 196

        # MAE / ImageNet normalization
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
        for p in self.base_model.parameters():
            p.requires_grad = False


    def forward(self, images):

        # ----------------------------------------------------
        # DINO-WM images are approximately [-1, 1].
        #
        # Convert:
        # [-1,1] -> [0,1]
        # ----------------------------------------------------

        images = (images + 1.0) / 2.0

        # ----------------------------------------------------
        # ImageNet normalization expected by MAE
        # ----------------------------------------------------

        images = (
            images - self.image_mean.to(images.dtype)
        ) / self.image_std.to(images.dtype)

        B = images.shape[0]

        # ----------------------------------------------------
        # CRITICAL:
        #
        # ViTMAE applies random_masking() even when
        # mask_ratio == 0.
        #
        # Ordered noise forces:
        #
        # patch 0 -> patch 0
        # patch 1 -> patch 1
        # ...
        # patch 195 -> patch 195
        # ----------------------------------------------------

        noise = torch.arange(
            self.num_patches,
            device=images.device,
            dtype=torch.float32,
        )

        noise = noise.unsqueeze(0).expand(B, -1)

        outputs = self.base_model(
            pixel_values=images,
            noise=noise,
            return_dict=True,
        )

        # [B, 197, 768]
        tokens = outputs.last_hidden_state

        # Remove CLS token
        # [B, 196, 768]
        patch_tokens = tokens[:, 1:, :]

        return patch_tokens


# ============================================================
# LOAD A FEW REAL WALL IMAGES
# ============================================================

def load_images():

    episode_path = os.path.join(
        DATASET_PATH,
        "obses",
        f"episode_{EPISODE_INDEX:03d}.pth",
    )

    print(f"\nLoading:\n{episode_path}")

    if not os.path.isfile(episode_path):
        raise FileNotFoundError(episode_path)

    episode = torch.load(
        episode_path,
        map_location="cpu",
    )

    print("\nRaw episode:")
    print("shape:", tuple(episode.shape))
    print("dtype:", episode.dtype)
    print(
        "range:",
        episode.min().item(),
        episode.max().item(),
    )

    # --------------------------------------------------------
    # Expected Wall episode:
    #
    # [T, 3, H, W]
    #
    # but handle channels-last too.
    # --------------------------------------------------------

    if episode.ndim != 4:
        raise RuntimeError(
            f"Expected 4D episode, got {episode.shape}"
        )

    if episode.shape[-1] == 3 and episode.shape[1] != 3:

        episode = episode.permute(
            0, 3, 1, 2
        )

    if episode.shape[1] != 3:
        raise RuntimeError(
            f"Expected RGB frames, got {episode.shape}"
        )

    # --------------------------------------------------------
    # Select frames spread across trajectory
    # --------------------------------------------------------

    indices = torch.linspace(
        0,
        len(episode) - 1,
        steps=N_IMAGES,
    ).long()

    print("\nUsing frame indices:")
    print(indices.tolist())

    images = episode[indices].float() / 255.0

    # --------------------------------------------------------
    # Same DINO-WM image transform:
    #
    # [0,1]
    #   ↓
    # resize/crop
    #   ↓
    # normalize -> [-1,1]
    # --------------------------------------------------------

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

    images = transform(images)

    print("\nProcessed images:")
    print("shape:", tuple(images.shape))
    print("min:  ", images.min().item())
    print("max:  ", images.max().item())
    print("mean: ", images.mean().item())
    print("std:  ", images.std().item())

    return images


# ============================================================
# SAVE TARGET + RECONSTRUCTION
# ============================================================

def save_comparison(
    target,
    reconstructed,
    filename,
):

    # Row 1 = original images
    # Row 2 = reconstructions

    comparison = torch.cat(
        [
            target.detach().cpu(),
            reconstructed.detach().cpu(),
        ],
        dim=0,
    )

    utils.save_image(
        comparison,
        filename,
        nrow=N_IMAGES,
        normalize=True,
        value_range=(-1, 1),
    )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    torch.manual_seed(0)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("MAE -> DINO-WM DECODER TINY OVERFIT TEST")
    print("=" * 70)

    print("\nDevice:", device)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # ========================================================
    # 1. LOAD IMAGES
    # ========================================================

    images = load_images().to(device)

    # ========================================================
    # 2. LOAD CORRECTED FROZEN MAE
    # ========================================================

    print("\nLoading MAE...")

    mae = MAEEncoder().to(device)
    mae.eval()

    # ========================================================
    # 3. COMPUTE MAE FEATURES ONCE
    #
    # Encoder is frozen, so there is no need to recompute
    # them during every optimization step.
    # ========================================================

    print("\nEncoding images...")

    with torch.no_grad():

        features = mae(images)

    print("\nMAE features:")
    print("shape:", tuple(features.shape))
    print("min:  ", features.min().item())
    print("max:  ", features.max().item())
    print("mean: ", features.mean().item())
    print("std:  ", features.std().item())

    if features.shape != (
        N_IMAGES,
        196,
        768,
    ):
        raise RuntimeError(
            f"Wrong MAE shape: {features.shape}"
        )

    if torch.isnan(features).any():
        raise RuntimeError(
            "MAE features contain NaN!"
        )

    if torch.isinf(features).any():
        raise RuntimeError(
            "MAE features contain Inf!"
        )

    # --------------------------------------------------------
    # DINO-WM decoder expects:
    #
    # [B, T, P, D]
    #
    # Add T = 1
    # --------------------------------------------------------

    decoder_input = features.unsqueeze(1)

    print(
        "\nDecoder input:",
        tuple(decoder_input.shape),
    )

    # Expected:
    #
    # [4, 1, 196, 768]

    # ========================================================
    # 4. CREATE A COMPLETELY FRESH DINO-WM DECODER
    # ========================================================

    print("\nCreating fresh decoder...")

    decoder = VQVAE(
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        emb_dim=768,
        n_embed=512,

        # Same idea as normal DINO-WM:
        # don't quantize the visual representation.
        quantize=False,
    ).to(device)

    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.MSELoss()

    # ========================================================
    # INITIAL RECONSTRUCTION
    # ========================================================

    decoder.eval()

    with torch.no_grad():

        reconstruction, _ = decoder(
            decoder_input
        )

        initial_loss = criterion(
            reconstruction,
            images,
        ).item()

    print("\nInitial reconstruction:")
    print(
        "shape:",
        tuple(reconstruction.shape),
    )

    print(
        "range:",
        reconstruction.min().item(),
        reconstruction.max().item(),
    )

    print(
        "Initial MSE:",
        initial_loss,
    )

    save_comparison(
        images,
        reconstruction,
        os.path.join(
            OUTPUT_DIR,
            "step_0000.png",
        ),
    )

    # ========================================================
    # 5. OVERFIT THE FOUR IMAGES
    # ========================================================

    print("\nStarting overfit test...")
    print()

    for step in range(
        1,
        NUM_STEPS + 1,
    ):

        decoder.train()

        reconstruction, _ = decoder(
            decoder_input
        )

        loss = criterion(
            reconstruction,
            images,
        )

        optimizer.zero_grad()

        loss.backward()

        # ----------------------------------------------------
        # First iteration: verify decoder receives gradients
        # ----------------------------------------------------

        if step == 1:

            grad_norm = 0.0

            for p in decoder.parameters():

                if p.grad is not None:

                    grad_norm += (
                        p.grad.detach().norm().item()
                    )

            print(
                "Decoder gradient norm:",
                grad_norm,
            )

            if grad_norm == 0:
                raise RuntimeError(
                    "Decoder is receiving no gradients!"
                )

        optimizer.step()

        # ====================================================
        # LOG / SAVE
        # ====================================================

        if (
            step == 1
            or step % SAVE_EVERY == 0
            or step == NUM_STEPS
        ):

            decoder.eval()

            with torch.no_grad():

                reconstruction, _ = decoder(
                    decoder_input
                )

                current_loss = criterion(
                    reconstruction,
                    images,
                ).item()

            print(
                f"step {step:4d} | "
                f"MSE = {current_loss:.8f}"
            )

            save_comparison(
                images,
                reconstruction,
                os.path.join(
                    OUTPUT_DIR,
                    f"step_{step:04d}.png",
                ),
            )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    final_loss = current_loss

    improvement = (
        initial_loss / final_loss
        if final_loss > 0
        else float("inf")
    )

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"Initial MSE : {initial_loss:.8f}"
    )

    print(
        f"Final MSE   : {final_loss:.8f}"
    )

    print(
        f"Improvement : {improvement:.2f}x"
    )

    print(
        f"\nImages saved to:\n{OUTPUT_DIR}"
    )

    print(
        """
Open:

    step_0000.png
    step_0500.png
    step_1000.png
    step_1500.png

Each image contains:

    FIRST ROW  = ground truth
    SECOND ROW = decoder reconstruction
"""
    )


if __name__ == "__main__":
    main()