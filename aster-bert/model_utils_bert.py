# /aster-bert/model_utils_bert.py

import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import config_bert as config

# Set the Hugging Face endpoint to the mirror
# This will affect all subsequent calls to from_pretrained
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def load_model_and_tokenizer():
    """
    Loads the BERT model and its corresponding tokenizer from the specified mirror.
    The base model is used as the 'teacher' and for its architecture.
    """
    print(f"--- Loading Model and Tokenizer for {config.MODEL_ID} from HF Mirror ---")
    device = config.DEVICE

    # 1. Load Model from Hugging Face Hub (via mirror)
    # The number of labels is determined by the dataset (SST-2 has 2 labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_ID,
        output_hidden_states=True,
        num_labels=2
    ).to(device)
    model.eval()  # Base model is frozen and used for feature extraction

    # 2. Load Tokenizer from Hugging Face Hub (via mirror)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)

    print("Model and tokenizer loaded successfully.")
    return model, tokenizer