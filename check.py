import transformers
import huggingface_hub

print("Transformers:", transformers.__version__)
print("HF Hub:", huggingface_hub.__version__)

from transformers import IJepaModel
print("IJepaModel import OK")