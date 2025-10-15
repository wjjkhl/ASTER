import torch
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm

import config
from model_utils import load_model_and_processor
from components import ScoringModel, DynamicAdapter
from deit_trainer import ImageNetDataset  # Reuse the Dataset class

def predict_greedy(model, scorer, adapter, images):

    num_layers = len(model.vit.encoder.layer)
    with torch.no_grad():

        student_hidden_state = model.vit.embeddings(images)

        l_curr = 0
        executed_layers_count = 0

        while l_curr < num_layers - 1:
            # Execute one encoder layer
            student_hidden_state = model.vit.encoder.layer[l_curr](student_hidden_state)[0]
            executed_layers_count += 1

            # Decide next step based on the [CLS] token's state
            cls_state = student_hidden_state[:, 0, :]

            candidate_layers = list(range(l_curr + 1, num_layers))
            if not candidate_layers:
                break

            # Use scorer to get scores for all possible next layers
            scores = scorer(h_cls=cls_state, l_curr=l_curr, candidate_layers=candidate_layers)

            # Greedily choose the action with the highest score
            action_index = torch.argmax(scores, dim=-1).item()
            l_next = candidate_layers[action_index]

            # Apply adapter if we are skipping one or more layers
            if l_next > l_curr + 1:
                student_hidden_state = adapter(student_hidden_state, l_curr, l_next)

            # Move to the next chosen layer
            l_curr = l_next

        # Final classification
        cls_token_final = student_hidden_state[:, 0]
        logits = model.classifier(cls_token_final)
        prediction = torch.argmax(logits, dim=-1).item()

    return prediction, executed_layers_count


def run_evaluation(args):
    """
    Main function to run the evaluation process.
    """
    print("--- Starting ASTER-DeiT Evaluation ---")

    # 1. Load Model, Processor, and ASTER Components
    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "aster_deit_checkpoint.pt")
        print(f"Checkpoint path not provided, using default: {args.checkpoint_path}")

    if not os.path.exists(args.checkpoint_path):
        print(f"[FATAL ERROR] Checkpoint file not found at {args.checkpoint_path}")
        return

    # Load base model and image transforms
    model, image_transform = load_model_and_processor()
    model.eval()

    # Initialize ASTER components
    hidden_dim = model.config.hidden_size
    num_layers = len(model.vit.encoder.layer)
    model_dtype = next(model.parameters()).dtype

    scorer = ScoringModel(hidden_dim, config.SCORER_HIDDEN_DIM, num_layers, model_dtype).to(config.DEVICE)
    adapter = DynamicAdapter(hidden_dim, config.ADAPTER_BOTTLENECK_DIM, num_layers, model_dtype).to(config.DEVICE)

    # Load trained weights from checkpoint
    print(f"Loading trained weights from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=config.DEVICE)
    # The trainer saves the '.module' state dict, so we can load it directly here.
    scorer.load_state_dict(checkpoint['scorer_state_dict'])
    adapter.load_state_dict(checkpoint['adapter_state_dict'])
    print("Successfully loaded Scorer and Adapter weights.")
    scorer.eval()
    adapter.eval()

    # 2. Load Dataset
    # We'll evaluate on the test split we created
    eval_dataset_path = config.TEST_DATA_PATH
    print(f"Loading evaluation data from: {eval_dataset_path}")
    try:
        eval_dataset = ImageNetDataset(eval_dataset_path, image_transform)
    except FileNotFoundError:
        print(f"ERROR: Evaluation data not found at {eval_dataset_path}")
        return

    # IMPORTANT: Batch size must be 1 for greedy per-sample evaluation
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 3. Run Evaluation Loop
    total_correct = 0
    total_executed_layers = 0
    total_samples = 0

    for images, labels in tqdm(eval_dataloader, desc="Evaluating with ASTER-DeiT"):
        images = images.to(config.DEVICE)
        # label is a tensor, get the scalar value
        label = labels.item()

        prediction, executed_layers = predict_greedy(model, scorer, adapter, images)

        if prediction == label:
            total_correct += 1

        total_executed_layers += executed_layers
        total_samples += 1

    # 4. Calculate and Display Results
    if total_samples == 0:
        print("No samples were evaluated.")
        return

    accuracy = total_correct / total_samples
    avg_executed_layers = total_executed_layers / total_samples
    speedup_ratio = num_layers / avg_executed_layers

    print("\n" + "=" * 60)
    print("--- ASTER-DeiT Evaluation Final Results ---")
    print(f"  Evaluated Checkpoint: {args.checkpoint_path}")
    print(f"  Evaluation Dataset: {eval_dataset_path}")
    print(f"  Decision Strategy: Greedy Search")
    print("-" * 60)
    print(f"  Top-1 Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")
    print(f"  Average Executed Layers: {avg_executed_layers:.2f} / {num_layers}")
    print(f"  Computational Savings: {100 * (1 - avg_executed_layers / num_layers):.2f}%")
    print(f"  Effective Speedup vs Full Model: {speedup_ratio:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained ASTER-DeiT model.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Full path to the saved checkpoint file. If not provided, uses the default from config.py."
    )
    args = parser.parse_args()
    run_evaluation(args)