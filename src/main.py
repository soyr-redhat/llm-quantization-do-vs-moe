from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from compressed_tensors.offload import dispatch_model
from tokenize_dataset import get_dataset, MAX_SEQUENCE_LENGTH, NUM_CALIBRATION_SAMPLES

META_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct" # small decoder-only model
QWEN_MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B" # small MoE model
SCHEMES = ["W8A8", "W4A16", "FP8", "NVFP4"]
#------------------------META------------------------#
for SCHEME in SCHEMES:
    metaModel = AutoModelForCausalLM.from_pretrained(META_MODEL_ID, dtype="auto")
    metaTokenizer = AutoTokenizer.from_pretrained(META_MODEL_ID)
    ds = get_dataset(metaTokenizer)

    metaRecipe = GPTQModifier(
        targets="Linear",
        scheme=SCHEME,
        ignore=["lm_head"],
    )
    # Apply quantization.
    oneshot(
        model=metaModel,
        dataset=ds,
        recipe=metaRecipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES
    )

    # Save to disk in compressed-tensors format.
    SAVE_DIR = "meta/" + META_MODEL_ID.split("/")[1] + f"-{SCHEME}"
    metaModel.save_pretrained(SAVE_DIR)
    metaTokenizer.save_pretrained(SAVE_DIR)
#----------------------------------------------------#
