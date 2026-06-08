from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from compressed_tensors.offload import dispatch_model
from tokenize_dataset import get_dataset, MAX_SEQUENCE_LENGTH, NUM_CALIBRATION_SAMPLES

META_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct" # small decoder-only model
DEEPSEEK_MODEL_ID = "deepseek-ai/deepseek-moe-16b-chat" # small MoE model
# SCHEMES = ["W8A8", "W4A16", "FP8", "NVFP4"]
SCHEMES = ["W4A16", "FP8", "NVFP4"]
# metaTokenizer = AutoTokenizer.from_pretrained(META_MODEL_ID)
# mDs = get_dataset(metaTokenizer)

dsTokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_MODEL_ID, trust_remote_code=True)
dsDs = get_dataset(dsTokenizer)

#------------------------META------------------------#
for SCHEME in SCHEMES:
#     metaModel = AutoModelForCausalLM.from_pretrained(META_MODEL_ID, dtype="auto")
    
#     metaRecipe = GPTQModifier(
#         targets="Linear",
#         scheme=SCHEME,
#         ignore=["lm_head"],
#     )

#     # Apply quantization.
#     oneshot(
#         model=metaModel,
#         dataset=mDs,
#         recipe=metaRecipe,
#         max_seq_length=MAX_SEQUENCE_LENGTH,
#         num_calibration_samples=NUM_CALIBRATION_SAMPLES
#     )

#     # Save to disk in compressed-tensors format.
#     SAVE_DIR = "/output/" + "meta/" + META_MODEL_ID.split("/")[1] + f"-{SCHEME}"
#     metaModel.save_pretrained(SAVE_DIR)
#     metaTokenizer.save_pretrained(SAVE_DIR)

#---------------------DEEPSEEK-----------------------#
    dsModel = AutoModelForCausalLM.from_pretrained(DEEPSEEK_MODEL_ID, dtype="auto", trust_remote_code=True)

    dsRecipe = GPTQModifier(
        targets="Linear",
        scheme=SCHEME,
        ignore=["lm_head", "re:.*mlp\.gate$", "model.layers.0.mlp.down_proj"],
    )

    # Apply quantization.
    oneshot(
        model=dsModel,
        dataset=dsDs,
        recipe=dsRecipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        trust_remote_code_model=True,
    )

    # Save to disk in compressed-tensors format
    SAVE_DIR = "/output/" + "deepseek/" + DEEPSEEK_MODEL_ID.split("/")[1] + f"-{SCHEME}"
    dsModel.save_pretrained(SAVE_DIR)
    dsTokenizer.save_pretrained(SAVE_DIR)
#------------------------------------------------------#
