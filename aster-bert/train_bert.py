# /aster-bert/train_bert.py (Corrected call to trainer)

import torch
import os
import argparse
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
import random
import numpy as np

import config_bert as config
from model_utils_bert import load_model_and_tokenizer
from components import ScoringModel, DynamicAdapter
from bert_trainer import ASTERTrainerBERT

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    """Main entry point for training the ASTER-DistilBERT framework."""
    set_seed(config.SEED)
    print("--- ASTER-DistilBERT Training Script ---")

    # 1. Load Base Model and Tokenizer
    print("Loading base DistilBERT model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    # 2. Load and Preprocess Dataset
    print(f"Loading dataset {config.DATASET_NAME}/{config.DATASET_CONFIG} from HF Hub...")
    try:
        raw_datasets = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)
    except Exception:
        print("Could not reach Hugging Face Hub. Attempting to use local cache.")
        raw_datasets = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, trust_remote_code=True)

    def preprocess_function(examples):
        return tokenizer(
            examples[config.DATASET_TEXT_COLUMN],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH
        )

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(
        [c for c in ['sentence', 'idx'] if c in tokenized_datasets['train'].column_names])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataset = tokenized_datasets["train"]
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.BATCH_SIZE)
    print("Dataset processed and DataLoader created.")

    # 3. Initialize ASTER Components
    print("Initializing ASTER components (Scorer and Adapter)...")
    hidden_dim = model.config.hidden_size
    # [MODIFIED] Get num_layers from the DistilBERT transformer structure.
    num_layers = len(model.distilbert.transformer.layer)
    model_dtype = next(model.parameters()).dtype
    print(f"Base model dtype: {model_dtype}, hidden_dim: {hidden_dim}, num_layers: {num_layers}")

    scorer = ScoringModel(hidden_dim, config.SCORER_HIDDEN_DIM, num_layers, model_dtype).to(config.DEVICE)
    adapter = DynamicAdapter(hidden_dim, config.ADAPTER_BOTTLENECK_DIM, num_layers, model_dtype).to(config.DEVICE)

    # 4. Initialize Optimizer
    optimizer = AdamW(list(scorer.parameters()) + list(adapter.parameters()), lr=config.LEARNING_RATE)

    # 5. Resume from Checkpoint if specified
    start_epoch = 0
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "aster_bert_checkpoint.pt")
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        scorer.load_state_dict(checkpoint['scorer_state_dict'])
        adapter.load_state_dict(checkpoint['adapter_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed successfully. Starting from epoch {start_epoch + 1}.")
    elif args.resume:
        print(f"WARNING: --resume specified, but no checkpoint found at {checkpoint_path}.")

    # 6. Setup and Run Trainer
    print("Setting up ASTER-DistilBERT trainer...")
    aster_trainer = ASTERTrainerBERT(model, scorer, adapter, optimizer)

    print(f"Starting training from epoch {start_epoch + 1}...")
    aster_trainer.train(train_dataloader, start_epoch=start_epoch)

    print("--- Training finished successfully! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ASTER framework on BERT.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last saved checkpoint.")
    args = parser.parse_args()
    main(args)