# /aster-deit/train_deit_distributed.py

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
import os
import argparse

import config
from model_utils import load_model_and_processor
from components import ScoringModel, DynamicAdapter
from deit_trainer import ASTERTrainerDeiT, ImageNetDataset

def setup(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # A free port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def main_process(rank, world_size, args):
    """ The main function for each process. """
    print(f"Running DDP training on rank {rank}.")
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # 1. Load Model and Processor
    # The base model is loaded on each process/GPU.
    if rank == 0: print("Loading base model and image processor...")
    model, image_transform = load_model_and_processor()
    model.to(device) # The base model does not need DDP as it's frozen
    model.eval()

    # 2. Initialize ASTER Components
    if rank == 0: print("Initializing ASTER components (Scorer and Adapter)...")
    hidden_dim = model.config.hidden_size
    num_layers = len(model.vit.encoder.layer)
    model_dtype = next(model.parameters()).dtype

    scorer = ScoringModel(hidden_dim, config.SCORER_HIDDEN_DIM, num_layers, model_dtype).to(device)
    adapter = DynamicAdapter(hidden_dim, config.ADAPTER_BOTTLENECK_DIM, num_layers, model_dtype).to(device)

    # Wrap the trainable models with DDP
    scorer = DDP(scorer, device_ids=[rank])
    adapter = DDP(adapter, device_ids=[rank])

    # 3. Initialize Optimizer
    optimizer = AdamW(
        list(scorer.parameters()) + list(adapter.parameters()),
        lr=config.LEARNING_RATE * world_size # Scale learning rate by world size
    )

    # 4. Resume from Checkpoint if specified
    start_epoch = 0
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "aster_deit_checkpoint.pt")
    if args.resume:
        if os.path.exists(checkpoint_path):
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            # Load state dicts for DDP models using .module
            scorer.module.load_state_dict(checkpoint['scorer_state_dict'])
            adapter.module.load_state_dict(checkpoint['adapter_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if rank == 0: print(f"Resumed from epoch {start_epoch}.")
        elif rank == 0:
            print("WARNING: --resume specified, but checkpoint not found.")

    # 5. Setup Dataset and DataLoader with DistributedSampler
    if rank == 0: print("Setting up dataset and dataloader...")
    try:
        train_dataset = ImageNetDataset(config.TRAIN_DATA_PATH, image_transform)
    except FileNotFoundError:
        if rank == 0: print(f"ERROR: Training data not found at {config.TRAIN_DATA_PATH}")
        return

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, sampler=train_sampler,
        num_workers=4, pin_memory=True, shuffle=False # shuffle must be False with sampler
    )

    # 6. Setup and Run Trainer
    if rank == 0: print("Setting up ASTER trainer...")
    aster_trainer = ASTERTrainerDeiT(
        model, image_transform, scorer, adapter, optimizer, config, rank, world_size
    )

    if rank == 0: print(f"Starting training from epoch {start_epoch + 1}...")
    aster_trainer.train(train_dataloader, train_sampler, start_epoch=start_epoch)

    if rank == 0: print("Training finished successfully!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASTER-DeiT using Distributed Data Parallel.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint.")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Distributed training requires at least 2 GPUs. Running on a single GPU.")
        if world_size == 1:
            print("Please use train_deit.py for single-GPU training.")
        exit()
    else:
        print(f"Found {world_size} GPUs. Starting distributed training...")

        mp.spawn(main_process, args=(world_size, args), nprocs=world_size, join=True)
