import torch
import numpy as np
from PIL import Image

# CHANGE THIS IMPORT to wherever MAEEncoder is defined
from mae import MAEEncoder


# ============================================================
# CONFIG
# ============================================================

IMAGE_PATH = "a.bmp"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")
image = image.resize((224, 224), Image.Resampling.BICUBIC)

image_np = np.asarray(image).astype(np.float32) / 255.0

# [H,W,C] -> [C,H,W]
x = torch.from_numpy(image_np).permute(2, 0, 1)

# add batch dimension
x = x.unsqueeze(0)

# DINO-WM convention:
# [0,1] -> [-1,1]
x = x * 2.0 - 1.0

x = x.to(DEVICE)

print("Input shape:", x.shape)
print("Input range:", x.min().item(), x.max().item())


# ============================================================
# CREATE ENCODER
# ============================================================

encoder = MAEEncoder(
    model_name="facebook/vit-mae-base",
    local_files_only=True,
).to(DEVICE)

encoder.eval()


# ============================================================
# TEST NORMAL FORWARD
# ============================================================

with torch.no_grad():
    patch_features = encoder(x)

print("\n======================================")
print("ENCODER OUTPUT")
print("======================================")

print("Output shape:", patch_features.shape)

assert patch_features.shape == (1, 196, 768)

print("✓ Encoder returns exactly 196 patch tokens")
print("✓ CLS token has been removed")


# ============================================================
# REPRODUCE PREPROCESSING
# ============================================================

x_mae = (x + 1.0) / 2.0

x_mae = (
    x_mae - encoder.image_mean
) / encoder.image_std


# ============================================================
# CREATE THE SAME IDENTITY NOISE AS YOUR ENCODER
# ============================================================

noise = torch.arange(
    encoder.num_patches,
    device=DEVICE,
    dtype=x_mae.dtype,
)

noise = noise.unsqueeze(0)


# ============================================================
# RUN HUGGING FACE MAE DIRECTLY
# ============================================================

with torch.no_grad():

    outputs = encoder.base_model(
        pixel_values=x_mae,
        noise=noise,
        return_dict=True,
    )


print("\n======================================")
print("MAE INTERNAL INFORMATION")
print("======================================")

print("mask_ratio:", encoder.base_model.config.mask_ratio)

print("last_hidden_state shape:",
      outputs.last_hidden_state.shape)

print("mask shape:",
      outputs.mask.shape)

print("ids_restore shape:",
      outputs.ids_restore.shape)


# ============================================================
# TEST 1: NO PATCHES MASKED
# ============================================================

num_masked = outputs.mask.sum().item()

print("\n======================================")
print("TEST 1 — MASKING")
print("======================================")

print("Number of masked patches:", num_masked)

if num_masked == 0:
    print("PASS: No patches are masked.")
else:
    print("FAIL: Some patches are being masked!")


# ============================================================
# TEST 2: PATCH ORDER
# ============================================================

expected_order = torch.arange(
    196,
    device=DEVICE,
).unsqueeze(0)

identity_order = torch.equal(
    outputs.ids_restore,
    expected_order,
)

print("\n======================================")
print("TEST 2 — PATCH ORDER")
print("======================================")

print("ids_restore:")
print(outputs.ids_restore)

if identity_order:
    print("PASS: Patch ordering is unchanged.")
else:
    print("FAIL: Patches have been reordered!")


# ============================================================
# SHOW ORDER AS 14 x 14 IMAGE GRID
# ============================================================

print("\nPatch indices as spatial 14x14 grid:")

print(
    outputs.ids_restore[0]
    .reshape(14, 14)
    .cpu()
)


# ============================================================
# TEST 3: WRAPPER OUTPUT == DIRECT MAE OUTPUT
# ============================================================

direct_patch_features = outputs.last_hidden_state[:, 1:, :]

same_features = torch.allclose(
    patch_features,
    direct_patch_features,
    atol=1e-6,
    rtol=1e-5,
)

print("\n======================================")
print("TEST 3 — FEATURE EXTRACTION")
print("======================================")

print(
    "Maximum difference:",
    (patch_features - direct_patch_features)
    .abs()
    .max()
    .item()
)

if same_features:
    print("PASS: MAEEncoder returns the original MAE patch outputs unchanged.")
else:
    print("FAIL: Returned features differ from the direct MAE output.")


# ============================================================
# FINAL RESULT
# ============================================================

print("\n======================================")
print("FINAL RESULT")
print("======================================")

if (
    num_masked == 0
    and identity_order
    and same_features
):
    print(
        "ALL TESTS PASSED:\n"
        "- 196 patches retained\n"
        "- no masking\n"
        "- no patch permutation\n"
        "- CLS removed\n"
        "- MAE patch features returned unchanged"
    )
else:
    print("ONE OR MORE TESTS FAILED.")