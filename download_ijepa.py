from transformers import IJepaModel

model = IJepaModel.from_pretrained(
    "facebook/ijepa_vith14_1k",
    cache_dir="/homes/jpecchini/.cache/huggingface",
)

print("I-JEPA downloaded successfully")