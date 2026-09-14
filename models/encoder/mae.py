import torch
import torch.nn as nn

from transformers import ViTMAEConfig, ViTMAEModel


class MAEEncoder(nn.Module):
    def __init__(
        self,
        model_name="facebook/vit-mae-base",
        cache_dir="/homes/jpecchini/.cache/huggingface",
        local_files_only=True,
    ):
        super().__init__()

        config = ViTMAEConfig.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        # DINO-WM needs every spatial patch.
        # The original MAE masks 75% of patches during pretraining.
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

        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

    def forward(self, images):
        outputs = self.base_model(
            pixel_values=images,
            interpolate_pos_encoding=False,
        )

        # Shape before removal: [batch, 197, 768]
        # Token 0 is CLS; tokens 1: are the 196 spatial patches.
        patch_tokens = outputs.last_hidden_state[:, 1:, :]

        return patch_tokens