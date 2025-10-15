import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import argparse
import os
from tqdm import tqdm
import random
import numpy as np

import config_bert as config
from model_utils_bert import load_model_and_tokenizer
from components import ScoringModel, DynamicAdapter

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def set_seed(seed):
    """
    Sets the seed for reproducibility across all relevant libraries and settings.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict_greedy(model, scorer, adapter, input_ids, attention_mask):
    # [MODIFIED] Get num_layers from the DistilBERT transformer structure.
    num_layers = len(model.distilbert.transformer.layer)
    with torch.no_grad():
        # [MODIFIED] Use model.distilbert.embeddings for DistilBERT.
        student_hidden_state = model.distilbert.embeddings(input_ids=input_ids)
        l_curr = 0
        executed_layers_count = 0

        while l_curr < num_layers - 1:
            executed_layers_count += 1
            # Note: DistilBERT layers internally handle the attention mask.
            student_hidden_state = model.distilbert.transformer.layer[l_curr](student_hidden_state, attn_mask=attention_mask)[0]
            cls_state = student_hidden_state[:, 0, :]

            required_future_steps = config.MIN_TOTAL_EXECUTED_LAYERS - executed_layers_count
            if required_future_steps > 0:
                max_allowed_l_next = num_layers - required_future_steps
            else:
                max_allowed_l_next = num_layers

            start_candidate = l_curr + 1
            end_candidate = min(num_layers, max_allowed_l_next + 1)
            candidate_layers = list(range(start_candidate, end_candidate))

            if not candidate_layers and start_candidate < num_layers:
                candidate_layers = [start_candidate]

            if not candidate_layers: break

            scores = scorer(h_cls=cls_state, l_curr=l_curr, candidate_layers=candidate_layers)
            action_index = torch.argmax(scores, dim=-1).item()
            l_next = candidate_layers[action_index]

            if l_next > l_curr + 1:
                student_hidden_state = adapter(student_hidden_state, l_curr, l_next)

            l_curr = l_next

        for final_l in range(l_curr, num_layers):
            student_hidden_state = model.distilbert.transformer.layer[final_l](student_hidden_state, attn_mask=attention_mask)[0]
            if final_l >= l_curr:
                executed_layers_count += 1

        cls_token_final = student_hidden_state[:, 0]
        # [MODIFIED] Handle DistilBERT's pre_classifier layer.
        logits = model.classifier(model.pre_classifier(cls_token_final)) if hasattr(model, 'pre_classifier') else model.classifier(cls_token_final)
        prediction = torch.argmax(logits, dim=-1).item()

    return prediction, executed_layers_count


def run_evaluation(args):
    print("--- Starting ASTER-DistilBERT Evaluation ---")
    set_seed(config.SEED)

    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "aster_bert_checkpoint.pt")
        print(f"Checkpoint path not provided, using default: {args.checkpoint_path}")

    if not os.path.exists(args.checkpoint_path):
        print(f"[FATAL ERROR] Checkpoint file not found at {args.checkpoint_path}")
        return

    model, tokenizer = load_model_and_tokenizer()
    model.eval()

    hidden_dim = model.config.hidden_size
    # [MODIFIED] Get num_layers from the DistilBERT transformer structure.
    num_layers = len(model.distilbert.transformer.layer)
    model_dtype = next(model.parameters()).dtype

    scorer = ScoringModel(hidden_dim, config.SCORER_HIDDEN_DIM, num_layers, model_dtype).to(config.DEVICE)
    adapter = DynamicAdapter(hidden_dim, config.ADAPTER_BOTTLENECK_DIM, num_layers, model_dtype).to(config.DEVICE)

    print(f"Loading trained weights from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=config.DEVICE)
    scorer.load_state_dict(checkpoint['scorer_state_dict'])
    adapter.load_state_dict(checkpoint['adapter_state_dict'])
    print("Successfully loaded Scorer and Adapter weights.")
    scorer.eval()
    adapter.eval()

    print(f"Loading and processing evaluation dataset ({config.DATASET_NAME}/{config.DATASET_CONFIG})...")
    try:
        raw_datasets = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)
    except Exception:
        print("Could not reach Hugging Face Hub. Attempting to use local cache.")
        raw_datasets = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, trust_remote_code=True)

    def preprocess_function(examples):
        return tokenizer(examples[config.DATASET_TEXT_COLUMN], padding="max_length", truncation=True,
                         max_length=config.MAX_SEQ_LENGTH)

    eval_dataset = raw_datasets["validation"].map(preprocess_function, batched=True)
    eval_dataset = eval_dataset.remove_columns([c for c in ['sentence', 'idx'] if c in eval_dataset.column_names])
    eval_dataset = eval_dataset.rename_column("label", "labels")
    eval_dataset.set_format("torch")

    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    total_correct, total_executed_layers, total_samples = 0, 0, 0

    for batch in tqdm(eval_dataloader, desc="Evaluating with ASTER-DistilBERT"):
        input_ids = batch['input_ids'].to(config.DEVICE)
        attention_mask = batch['attention_mask'].to(config.DEVICE)
        label = batch['labels'].item()
        prediction, executed_layers = predict_greedy(model, scorer, adapter, input_ids, attention_mask)
        if prediction == label: total_correct += 1
        total_executed_layers += executed_layers
        total_samples += 1

    if total_samples == 0:
        print("No samples were evaluated.")
        return

    accuracy = total_correct / total_samples
    avg_executed_layers = total_executed_layers / total_samples
    speedup_ratio = num_layers / avg_executed_layers

    print("\n" + "=" * 60)
    print("--- ASTER-DistilBERT Evaluation Final Results ---")
    print(f"  Evaluated Checkpoint: {args.checkpoint_path}")
    print(f"  Evaluation Dataset: {config.DATASET_NAME}/{config.DATASET_CONFIG} (validation split)")
    print(f"  Decision Strategy: Greedy Search")
    print("-" * 60)
    print(f"  Top-1 Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")
    print(f"  Average Executed Layers: {avg_executed_layers:.2f} / {num_layers}")
    print(f"  Computational Savings: {100 * (1 - avg_executed_layers / num_layers):.2f}%")
    print(f"  Effective Speedup vs Full Model: {speedup_ratio:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained ASTER-BERT model.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Full path to the saved checkpoint file.")
    args = parser.parse_args()
    run_evaluation(args)