import torch.nn as nn

from transformers import SiglipVisionModel


class SigLIPEncoder(nn.Module):
    def __init__(
        self,
        model_name="google/siglip-base-patch16-224",
        cache_dir="/homes/jpecchini/.cache/huggingface",
        local_files_only=True,
    ):
        super().__init__()

        self.base_model = SiglipVisionModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        self.name = model_name
        self.emb_dim = self.base_model.config.hidden_size   # 768
        self.patch_size = self.base_model.config.patch_size # 16
        self.latent_ndim = 2

        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

    def forward(self, images):
        outputs = self.base_model(
            pixel_values=images,
            return_dict=True,
        )

        # SigLIP-B/16 directly returns:
        # [batch_size, 196, 768]
        patch_tokens = outputs.last_hidden_state

        expected_patches = (
            images.shape[-2] // self.patch_size
        ) * (
            images.shape[-1] // self.patch_size
        )

        if patch_tokens.shape[1] != expected_patches:
            raise RuntimeError(
                "Unexpected SigLIP token count: expected {}, got {}".format(
                    expected_patches,
                    patch_tokens.shape[1],
                )
            )

        return patch_tokens