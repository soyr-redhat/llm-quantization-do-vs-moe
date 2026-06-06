from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.offload import dispatch_model

META_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct" # small decoder-only model
QWEN_MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B" # small MoE model

#------------------------META------------------------#
metaModel = AutoModelForCausalLM.from_pretrained(META_MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(META_MODEL_ID)

# Apply quantization.
oneshot(model=metaModel, recipe=recipe)

# Save to disk in compressed-tensors format.
SAVE_DIR = META_MODEL_ID.split("/")[1] + "-FP8-BLOCK"
metaModel.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

#----------------------------------------------------#