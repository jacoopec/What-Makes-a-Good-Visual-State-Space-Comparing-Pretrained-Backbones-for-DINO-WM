import torch

from models.encoder.ijepa import IJEPAEncoder


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


encoder = IJEPAEncoder(
    model_name="facebook/ijepa_vith14_1k",
    cache_dir="/homes/jpecchini/.cache/huggingface",
    local_files_only=True,
)

encoder = encoder.to(device)
encoder.eval()


# Fake DINO-WM observation
x = torch.rand(
    2,
    3,
    224,
    224,
    device=device,
)

# [0,1] -> [-1,1]
x = x * 2.0 - 1.0


with torch.no_grad():

    z = encoder(x)


print("Input:")
print(x.shape)

print("\nOutput:")
print(z.shape)

print("\nExpected:")
print("[2, 196, 1280]")

print("\nStatistics:")
print("min: ", z.min().item())
print("max: ", z.max().item())
print("mean:", z.mean().item())
print("std: ", z.std().item())

print("\nFinite:", torch.isfinite(z).all().item())