# /aster-deit/train_deit.py (Corrected)

import torch
import os
import argparse
from torch.optim import AdamW

import config
from model_utils import load_model_and_processor
from components import ScoringModel, DynamicAdapter
from deit_trainer import ASTERTrainerDeiT


def main(args):
    print("--- ASTER-DeiT Training Script ---")

    # 1. Load Base Model and Image Transform
    print("Loading base DeiT model and image transform...")
    model, image_transform = load_model_and_processor()

    # 2. Initialize ASTER Components
    print("Initializing ASTER components (Scorer and Adapter)...")
    hidden_dim = model.config.hidden_size

    num_layers = len(model.vit.encoder.layer)


    model_dtype = next(model.parameters()).dtype
    print(f"Base model dtype: {model_dtype}, hidden_dim: {hidden_dim}, num_layers: {num_layers}")

    scorer = ScoringModel(hidden_dim, config.SCORER_HIDDEN_DIM, num_layers, model_dtype).to(config.DEVICE)
    adapter = DynamicAdapter(hidden_dim, config.ADAPTER_BOTTLENECK_DIM, num_layers, model_dtype).to(config.DEVICE)

    optimizer = AdamW(
        list(scorer.parameters()) + list(adapter.parameters()),
        lr=config.LEARNING_RATE
    )

    start_epoch = 0
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "aster_deit_checkpoint.pt")

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

    print("Setting up ASTER-DeiT trainer...")
    aster_trainer = ASTERTrainerDeiT(model, image_transform, scorer, adapter, optimizer, config)

    print(f"Starting training from epoch {start_epoch + 1}...")
    aster_trainer.train(start_epoch=start_epoch)

    print("--- Training finished successfully! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ASTER framework on DeiT.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last saved checkpoint."
    )
    args = parser.parse_args()

    main(args)
