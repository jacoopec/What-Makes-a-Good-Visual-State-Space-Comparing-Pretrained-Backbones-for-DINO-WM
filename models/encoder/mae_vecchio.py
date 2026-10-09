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

        self.base_model = ViTMAEModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        
        self.base_model.eval()

        # We want ALL patches.
        self.base_model.config.mask_ratio = 0.0

        self.emb_dim = 768
        self.patch_size = 16
        self.image_size = 224
        self.num_patches = 14 * 14
        self.latent_ndim = 2
        self.name = "mae_vit_base_patch16"

        # Frozen visual encoder
        for p in self.base_model.parameters():
            p.requires_grad = False

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
        super().train(mode)

        # MAE is permanently frozen: keep the pretrained
        # backbone in evaluation mode.
        self.base_model.eval()

        return self

    def forward(self, x):
        # print(
        #     "MAE input:",
        #     "shape =", x.shape,
        #     "dtype =", x.dtype,
        #     "min =", x.min().item(),
        #     "max =", x.max().item(),
        #     "mean =", x.mean().item(),
        # )
        # x from DINO-WM:
        # [B,3,224,224], range approximately [-1,1]

        if x.shape[-2:] != (224, 224):
            x = F.interpolate(
                x,
                size=(224, 224),
                mode="bicubic",
                align_corners=False,
            )

        # ---------------------------------------
        # IMPORTANT FIX 1:
        # DINO-WM [-1,1] -> normal RGB [0,1]
        # ---------------------------------------

        x = (x + 1.0) / 2.0
        
#         print(
#             "After [-1,1] -> [0,1]:",
#             "min =", x.min().item(),
#             "max =", x.max().item(),
#             "mean =", x.mean().item(),
#         )
        
#         assert x.min() >= -1e-5, x.min()
#         assert x.max() <= 1.0 + 1e-5, x.max()

        # ---------------------------------------
        # MAE pretrained normalization
        # ---------------------------------------

        x = (x - self.image_mean) / self.image_std

        B = x.shape[0]
        
        
        # print(
        #     "After ImageNet normalization:",
        #     "min =", x.min().item(),
        #     "max =", x.max().item(),
        #     "mean =", x.mean().item(),
        # )

        # ---------------------------------------
        # IMPORTANT FIX 2:
        # Keep MAE patches in spatial order
        # ---------------------------------------

        noise = torch.arange(
            self.num_patches,
            device=x.device,
            dtype=x.dtype,
        )

        noise = noise.unsqueeze(0).expand(B, -1)

        outputs = self.base_model(
            pixel_values=x,
            noise=noise,
            return_dict=True,
        )

        tokens = outputs.last_hidden_state

        # [B,197,768]
        # assert tokens.shape[1] == 197

        # Remove CLS
        patch_tokens = tokens[:, 1:, :]
        
        # print("MAE mask ratio:", self.base_model.config.mask_ratio)
        # print("MAE training mode:", self.base_model.training)
        # print(
        #     "Trainable MAE params:",
        #     sum(p.requires_grad for p in self.base_model.parameters())
        # )
        # print(
        #     "MAE patch tokens:",
        #     "shape =", patch_tokens.shape,
        #     "min =", patch_tokens.min().item(),
        #     "max =", patch_tokens.max().item(),
        #     "mean =", patch_tokens.mean().item(),
        #     "std =", patch_tokens.std().item(),
        # )
        
        # [B,196,768]
        # assert patch_tokens.shape[1:] == (196, 768)

        return patch_tokens