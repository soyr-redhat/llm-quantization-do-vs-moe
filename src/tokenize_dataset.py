from datasets import load_dataset

DATASET = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft" # name of split in ds
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

def get_dataset(tokenizer):
    ds = load_dataset(DATASET, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=8000).select(range(NUM_CALIBRATION_SAMPLES))

    def preprocess(ex):
        return {
            "text" : tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
            )
        }
    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )
    
    ds = ds.map(tokenize, remove_columns=ds.column_names)
    return ds