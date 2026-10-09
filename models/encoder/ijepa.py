import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import IJepaModel


class IJEPAEncoder(nn.Module):

    def __init__(
        self,
        model_name="facebook/ijepa_vith14_1k",
        cache_dir="/homes/jpecchini/.cache/huggingface",
        local_files_only=True,
        input_size=196,
    ):
        super().__init__()

        self.base_model = IJepaModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        self.name = "ijepa_vith14_1k"

        # I-JEPA ViT-H/14
        self.emb_dim = self.base_model.config.hidden_size    # 1280
        self.patch_size = self.base_model.config.patch_size  # 14

        # DINO-WM expects spatial tokens
        self.latent_ndim = 2

        # ----------------------------------------------------
        # We use 196x196 instead of I-JEPA's native 224x224.
        #
        # 196 / 14 = 14
        # 14 * 14 = 196 patches
        #
        # This keeps the same token grid used by DINO-WM.
        # ----------------------------------------------------

        self.input_size = input_size

        if self.input_size % self.patch_size != 0:
            raise ValueError(
                f"input_size={self.input_size} must be divisible by "
                f"patch_size={self.patch_size}"
            )

        self.grid_size = self.input_size // self.patch_size
        self.num_patches = self.grid_size ** 2

        # Frozen pretrained visual encoder
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

    def forward(self, x):

        # ----------------------------------------------------
        # Input from DINO-WM:
        #
        # [B, 3, H, W]
        # approximately in [-1, 1]
        #
        # This is already the normalization expected by the
        # official I-JEPA preprocessing.
        # ----------------------------------------------------

        if x.shape[-2:] != (self.input_size, self.input_size):

            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bicubic",
                align_corners=False,
            )

        outputs = self.base_model(
            pixel_values=x,

            # Required because pretrained positional embeddings
            # correspond to 224x224 (16x16 patches), while here
            # we use 196x196 (14x14 patches).
            interpolate_pos_encoding=True,

            return_dict=True,
        )

        patch_tokens = outputs.last_hidden_state

        # Expected:
        #
        # [B, 196, 1280]

        if patch_tokens.shape[1] != self.num_patches:
            raise RuntimeError(
                "Unexpected I-JEPA token count: "
                f"expected {self.num_patches}, "
                f"got {patch_tokens.shape[1]}"
            )

        if patch_tokens.shape[2] != self.emb_dim:
            raise RuntimeError(
                "Unexpected I-JEPA embedding dimension: "
                f"expected {self.emb_dim}, "
                f"got {patch_tokens.shape[2]}"
            )

        return patch_tokens