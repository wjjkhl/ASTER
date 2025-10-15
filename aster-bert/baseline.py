import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import os
from tqdm import tqdm
import argparse
import numpy as np
import random

import config_bert as config
from model_utils_bert import load_model_and_tokenizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_baseline_evaluation(args):
    # [MODIFIED] Set seed at the beginning of the script for reproducibility.
    set_seed(config.SEED)

    # [MODIFIED] Updated print statements to be model-agnostic.
    print(f"--- Starting Baseline Evaluation (Full Model) ---")
    print(f"--- Random Seed set to: {config.SEED} ---")

    print(f"Loading original pre-trained model '{config.MODEL_ID}' for baseline test.")
    model, tokenizer = load_model_and_tokenizer()
    model.eval()

    print(f"Loading and processing evaluation dataset ({config.DATASET_NAME}/{config.DATASET_CONFIG})...")
    raw_datasets = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)

    def preprocess_function(examples):
        return tokenizer(examples[config.DATASET_TEXT_COLUMN], padding="max_length", truncation=True,
                         max_length=config.MAX_SEQ_LENGTH)

    eval_dataset = raw_datasets["validation"].map(preprocess_function, batched=True)

    # Make column removal more robust
    columns_to_remove = [config.DATASET_TEXT_COLUMN]
    if 'idx' in eval_dataset.column_names:
        columns_to_remove.append('idx')
    eval_dataset = eval_dataset.remove_columns(columns_to_remove)

    eval_dataset = eval_dataset.rename_column("label", "labels")
    eval_dataset.set_format("torch")

    # DataLoader should not be shuffled for evaluation.
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        # [MODIFIED] Updated progress bar description.
        for batch in tqdm(eval_dataloader, desc=f"Evaluating {os.path.basename(config.MODEL_ID)}"):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            predictions = torch.argmax(outputs.logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples

    print("\n" + "=" * 60)
    # [MODIFIED] Updated result summary.
    print("--- Model Baseline Evaluation Final Results ---")
    print(f"  Evaluated Model: {config.MODEL_ID} (Full, no skipping)")
    print(f"  Evaluation Dataset: {config.DATASET_NAME}/{config.DATASET_CONFIG} (validation split)")
    print("-" * 60)
    print(f"  Top-1 Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a baseline Hugging Face model without any layer skipping.")
    args = parser.parse_args()
    run_baseline_evaluation(args)