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

        # Keep every patch
        config.mask_ratio = 0.0

        self.base_model = ViTMAEModel.from_pretrained(
            model_name,
            config=config,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        self.name = model_name
        self.emb_dim = config.hidden_size      # 768
        self.patch_size = config.patch_size    # 16
        self.latent_ndim = 2

        self.num_patches = (
            config.image_size // config.patch_size
        ) ** 2                                 # 196

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


    def forward(self, images):

        # DINO-WM gives images approximately in [-1, 1]
        # Convert back to [0, 1]
        images = (images + 1.0) / 2.0

        # Apply normalization expected by MAE
        images = (
            images - self.image_mean.to(images.dtype)
        ) / self.image_std.to(images.dtype)

        batch_size = images.shape[0]

        # IMPORTANT:
        # mask_ratio=0 still invokes random_masking().
        # Ordered noise keeps all patches in spatial order.
        noise = torch.arange(
            self.num_patches,
            device=images.device,
            dtype=torch.float32,
        )

        noise = noise.unsqueeze(0).expand(
            batch_size, -1
        )

        outputs = self.base_model(
            pixel_values=images,
            noise=noise,
            interpolate_pos_encoding=False,
        )

        # [B, 197, 768]
        tokens = outputs.last_hidden_state

        # Remove CLS:
        # [B, 196, 768]
        patch_tokens = tokens[:, 1:, :]

        return patch_tokens