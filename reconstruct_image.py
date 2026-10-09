import argparse
import sys
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from torchvision import transforms
from torchvision.utils import save_image


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CONF_DIR = ROOT / "conf"

sys.path.insert(0, str(ROOT))

# Register custom Hydra resolvers if the project uses them.
try:
    import custom_resolvers  # noqa: F401
except ImportError:
    pass


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def load_checkpoint(path, device):
    """
    DINO-WM checkpoints contain serialized modules such as:
        decoder
        predictor
        action_encoder
        proprio_encoder
        ...
    """

    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        # Compatibility with older PyTorch versions
        return torch.load(
            path,
            map_location=device,
        )


def to_image_range(x):
    """
    DINO-WM datasets normally store transformed images in [-1, 1].
    Convert to [0, 1] for saving.
    """

    x = x.detach().cpu()

    if x.min() < 0:
        x = (x + 1.0) / 2.0

    return x.clamp(0.0, 1.0)


def prepare_encoder_input(image, encoder, img_size):
    """
    Reproduce the special DINOv2 resizing used by VWorldModel.

    DINO-WM's VQ-VAE uses a 14x14 = 196 token grid.
    DINOv2 ViT-S/14 therefore receives 196x196 images:
        196 / 14 = 14

    CLIP / MAE / SigLIP wrappers already handle their expected
    image size themselves.
    """

    encoder_name = str(
        getattr(encoder, "name", "")
    ).lower()

    if "dino" in encoder_name:

        decoder_scale = 16

        num_side_patches = (
            img_size // decoder_scale
        )

        encoder_image_size = (
            num_side_patches * encoder.patch_size
        )

        resize = transforms.Resize(
            encoder_image_size
        )

        image = resize(image)

        print(
            f"DINO encoder input resized to "
            f"{encoder_image_size}x{encoder_image_size}"
        )

    return image


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        default="pusht",
        help="Dataset/environment config, e.g. pusht or wall",
    )

    parser.add_argument(
        "--encoder",
        default="dino",
        choices=[
            "dino",
            "clip",
            "mae",
            "siglip",
        ],
        help="Visual backbone",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model_latest.pth",
    )

    parser.add_argument(
        "--trajectory",
        type=int,
        default=0,
        help="Trajectory index",
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index inside trajectory",
    )

    parser.add_argument(
        "--n-rollout",
        type=int,
        default=1,
        help="Number of dataset rollouts to load",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # -----------------------------------------------------
    # Load Hydra configuration
    # -----------------------------------------------------

    with initialize_config_dir(
        config_dir=str(CONF_DIR),
        version_base=None,
    ):

        cfg = compose(
            config_name="train",
            overrides=[
                f"env={args.env}",
                f"encoder={args.encoder}",
                f"img_size={args.img_size}",
            ],
        )

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------

    print("\nLoading dataset...")

    datasets, traj_dsets = hydra.utils.call(
        cfg.env.dataset,

        # Only needed by the loader to construct sliced datasets.
        # We use traj_dsets directly below.
        num_hist=1,
        num_pred=1,
        frameskip=1,

        n_rollout=args.n_rollout,
    )

    trajectory_dataset = traj_dsets["train"]

    if args.trajectory >= len(trajectory_dataset):
        raise IndexError(
            f"Trajectory {args.trajectory} does not exist. "
            f"Dataset contains {len(trajectory_dataset)} trajectories."
        )

    obs, act, state, info = (
        trajectory_dataset[args.trajectory]
    )

    visual_sequence = obs["visual"]

    if args.frame >= visual_sequence.shape[0]:
        raise IndexError(
            f"Frame {args.frame} does not exist. "
            f"Trajectory has "
            f"{visual_sequence.shape[0]} frames."
        )

    # Shape:
    #
    #   C x H x W
    #
    image = visual_sequence[
        args.frame
    ]

    # Add batch dimension:
    #
    #   1 x C x H x W
    #
    image = image.unsqueeze(0)

    print(
        "Dataset image shape:",
        tuple(image.shape),
    )

    print(
        "Dataset image range:",
        image.min().item(),
        image.max().item(),
    )

    # -----------------------------------------------------
    # Instantiate selected encoder
    # -----------------------------------------------------

    print(
        f"\nLoading encoder: {args.encoder}"
    )

    encoder = hydra.utils.instantiate(
        cfg.encoder
    )

    encoder = encoder.to(device)
    encoder.eval()

    # -----------------------------------------------------
    # Load TRAINED VQ-VAE decoder
    # -----------------------------------------------------

    print(
        "\nLoading checkpoint:",
        args.checkpoint,
    )

    checkpoint = load_checkpoint(
        args.checkpoint,
        device,
    )

    if "decoder" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain a decoder. "
            "Use a checkpoint from a run where "
            "model.train_decoder=true."
        )

    decoder = checkpoint["decoder"]

    decoder = decoder.to(device)
    decoder.eval()

    print(
        "Decoder:",
        type(decoder).__name__,
    )

    # -----------------------------------------------------
    # Encode image
    # -----------------------------------------------------

    image = image.to(device)

    encoder_input = prepare_encoder_input(
        image,
        encoder,
        args.img_size,
    )

    with torch.inference_mode():

        # Encoder output:
        #
        #   B x P x D
        #
        features = encoder(
            encoder_input
        )

        print(
            "\nEncoder features:",
            tuple(features.shape),
        )

        if features.ndim != 3:
            raise RuntimeError(
                "Expected encoder output [B, P, D], "
                f"received {tuple(features.shape)}"
            )

        B, P, D = features.shape

        # -------------------------------------------------
        # Validate patch grid
        # -------------------------------------------------

        side = int(P ** 0.5)

        if side * side != P:
            raise RuntimeError(
                f"VQ-VAE requires a square patch grid, "
                f"but P={P}."
            )

        print(
            f"Patch grid: {side} x {side}"
        )

        # -------------------------------------------------
        # Check feature dimension against decoder
        # -------------------------------------------------

        if hasattr(decoder, "quantize_b"):

            expected_dim = getattr(
                decoder.quantize_b,
                "dim",
                None,
            )

            if (
                expected_dim is not None
                and D != expected_dim
            ):
                raise RuntimeError(
                    "\nEncoder/decoder mismatch:\n"
                    f"Encoder produces D={D}\n"
                    f"Decoder expects D={expected_dim}\n\n"
                    "Use the decoder checkpoint trained "
                    "with this same visual backbone."
                )

        # -------------------------------------------------
        # VQ-VAE expects:
        #
        #   B x T x P x D
        #
        # Here T = 1.
        # -------------------------------------------------

        features_vqvae = (
            features.unsqueeze(1)
        )

        print(
            "VQ-VAE input:",
            tuple(features_vqvae.shape),
        )

        # VQVAE.forward():
        #
        #   reconstruction, vq_loss
        #
        reconstruction, vq_loss = decoder(
            features_vqvae
        )

    print(
        "Reconstruction shape:",
        tuple(reconstruction.shape),
    )

    print(
        "Reconstruction range:",
        reconstruction.min().item(),
        reconstruction.max().item(),
    )

    # -----------------------------------------------------
    # Save feature tensor
    # -----------------------------------------------------

    feature_path = (
        f"features_{args.encoder}.pt"
    )

    torch.save(
        features.detach().cpu(),
        feature_path,
    )

    # -----------------------------------------------------
    # Save original image
    # -----------------------------------------------------

    input_img = to_image_range(
        image
    )

    reconstructed_img = to_image_range(
        reconstruction
    )

    input_path = "input_image.png"

    reconstruction_path = (
        f"reconstruction_{args.encoder}.png"
    )

    comparison_path = (
        f"comparison_{args.encoder}.png"
    )

    save_image(
        input_img,
        input_path,
    )

    save_image(
        reconstructed_img,
        reconstruction_path,
    )

    # -----------------------------------------------------
    # Save side-by-side comparison
    #
    # Left  = original
    # Right = reconstruction
    # -----------------------------------------------------

    comparison = torch.cat(
        [
            input_img,
            reconstructed_img,
        ],
        dim=0,
    )

    save_image(
        comparison,
        comparison_path,
        nrow=2,
    )

    print("\nSaved:")
    print(" ", input_path)
    print(" ", reconstruction_path)
    print(" ", comparison_path)
    print(" ", feature_path)


if __name__ == "__main__":
    main()