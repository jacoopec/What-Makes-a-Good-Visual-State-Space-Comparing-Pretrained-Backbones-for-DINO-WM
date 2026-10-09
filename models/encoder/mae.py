import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ViTMAEModel


class MAEEncoder(nn.Module):

    def __init__(
        self,
        model_name="facebook/vit-mae-base",
        local_files_only=True,
    ):
        super().__init__()

        # -----------------------------------------------------
        # Frozen pretrained MAE backbone
        # -----------------------------------------------------

        self.base_model = ViTMAEModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )

        # Keep all spatial patches
        self.base_model.config.mask_ratio = 0.0

        # Freeze MAE backbone
        for p in self.base_model.parameters():
            p.requires_grad = False

        self.base_model.eval()

        # -----------------------------------------------------
        # Trainable projection:
        # 768-D MAE token -> 384-D token
        # -----------------------------------------------------

        self.proj = nn.Linear(768, 384)

        # -----------------------------------------------------
        # Encoder metadata exposed to DINO-WM
        # -----------------------------------------------------

        # IMPORTANT:
        # DINO-WM now sees 384-D output features.
        self.emb_dim = 384

        self.patch_size = 16
        self.image_size = 224
        self.num_patches = 14 * 14
        self.latent_ndim = 2

        self.name = "mae_vit_base_patch16_proj384"

        # -----------------------------------------------------
        # MAE normalization
        # -----------------------------------------------------

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

    def train(self, mode=True):
        """
        DINO-WM may call encoder.train() during training.

        We want:
            projection -> training mode
            MAE backbone -> always eval mode
        """

        super().train(mode)

        # MAE stays permanently in eval mode
        self.base_model.eval()

        return self

    def forward(self, x):
        """
        Input:
            x: [B, 3, H, W], approximately in [-1, 1]

        Output:
            [B, 196, 384]
        """

        # -----------------------------------------------------
        # Resize if necessary
        # -----------------------------------------------------

        if x.shape[-2:] != (224, 224):
            x = F.interpolate(
                x,
                size=(224, 224),
                mode="bicubic",
                align_corners=False,
            )

        # -----------------------------------------------------
        # DINO-WM [-1,1] -> RGB [0,1]
        # -----------------------------------------------------

        x = (x + 1.0) / 2.0

        assert x.min() >= -1e-5
        assert x.max() <= 1.0 + 1e-5

        # -----------------------------------------------------
        # MAE pretrained normalization
        # -----------------------------------------------------

        x = (x - self.image_mean) / self.image_std

        B = x.shape[0]

        # -----------------------------------------------------
        # Keep all MAE patches in spatial order
        # -----------------------------------------------------

        noise = torch.arange(
            self.num_patches,
            device=x.device,
            dtype=x.dtype,
        )

        noise = noise.unsqueeze(0).expand(B, -1)

        # -----------------------------------------------------
        # Frozen MAE forward
        # -----------------------------------------------------

        outputs = self.base_model(
            pixel_values=x,
            noise=noise,
            return_dict=True,
        )

        tokens = outputs.last_hidden_state

        # CLS + 196 patches
        assert tokens.shape[1:] == (197, 768)

        # Remove CLS token
        patch_tokens = tokens[:, 1:, :]

        assert patch_tokens.shape[1:] == (196, 768)

        # -----------------------------------------------------
        # Trainable projection
        # -----------------------------------------------------

        patch_tokens = self.proj(patch_tokens)

        assert patch_tokens.shape[1:] == (196, 384)

        return patch_tokens