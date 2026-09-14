import torch
import torch.nn as nn
import clip


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

class CLIPVisionEncoder(nn.Module):
    def __init__(
        self,
        name="ViT-B/16",
        return_projected_tokens=True,
        download_root="/homes/jpecchini/.cache/clip",
    ):
        super().__init__()

        clip_model, _ = clip.load(
            name,
            device="cpu",
            jit=False,
            download_root=download_root,
        )

        self.name = name
        self.visual = clip_model.visual.float()
        self.return_projected_tokens = return_projected_tokens

        if not hasattr(self.visual, "transformer"):
            raise ValueError("Usare un modello CLIP ViT, per esempio ViT-B/16")

        self.latent_ndim = 2
        self.patch_size = int(self.visual.conv1.kernel_size[0])

        transformer_dim = int(self.visual.conv1.out_channels)

        if return_projected_tokens and self.visual.proj is not None:
            self.emb_dim = int(self.visual.proj.shape[1])
        else:
            self.emb_dim = transformer_dim

        self.register_buffer(
            "clip_mean",
            torch.tensor(CLIP_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor(CLIP_STD).view(1, 3, 1, 1),
            persistent=False,
        )

        for parameter in self.visual.parameters():
            parameter.requires_grad = False

    def forward(self, images):
        # DINO-WM fornisce immagini normalizzate in [-1, 1].
        images = (images + 1.0) / 2.0

        # Normalizzazione richiesta da CLIP.
        images = (
            images - self.clip_mean.to(images.dtype)
        ) / self.clip_std.to(images.dtype)

        visual = self.visual
        images = images.to(visual.conv1.weight.dtype)

        x = visual.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        class_token = visual.class_embedding.to(x.dtype)
        class_token = class_token + torch.zeros(
            x.shape[0],
            1,
            x.shape[-1],
            dtype=x.dtype,
            device=x.device,
        )

        x = torch.cat([class_token, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)

        # Esclude il token CLS e conserva i 196 patch token.
        patch_tokens = visual.ln_post(x[:, 1:, :])

        if self.return_projected_tokens and visual.proj is not None:
            patch_tokens = patch_tokens @ visual.proj

        return patch_tokens.float()