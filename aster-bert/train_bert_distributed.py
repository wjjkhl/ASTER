# /aster-bert/train_bert_distributed.py

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
import os
import argparse
from datasets import load_dataset
import random
import numpy as np

import config_bert as config
from model_utils_bert import load_model_and_tokenizer
from components import ScoringModel, DynamicAdapter
from bert_trainer import ASTERTrainerBERT

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def setup(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()


def main_process(rank, world_size, args):
    """ The main function for each process. """
    print(f"Running DDP training on rank {rank}.")
    setup(rank, world_size)
    set_seed(config.SEED) # Set seed in each process
    device = torch.device(f'cuda:{rank}')

    if rank == 0: print("Loading base model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    model.to(device)
    model.eval()

    if rank == 0: print(f"Loading and processing dataset...")
    # Dataset loading should be done once if possible, but it's safe this way.
    raw_datasets = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)

    def preprocess_function(examples):
        return tokenizer(examples[config.DATASET_TEXT_COLUMN], padding="max_length", truncation=True,
                         max_length=config.MAX_SEQ_LENGTH)

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, num_proc=os.cpu_count())
    tokenized_datasets = tokenized_datasets.remove_columns([c for c in ['sentence', 'idx'] if c in tokenized_datasets['train'].column_names])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    train_dataset = tokenized_datasets["train"]

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=train_sampler, num_workers=4,
                                  pin_memory=True, shuffle=False)
    if rank == 0: print("Distributed Sampler and DataLoader created.")

    if rank == 0: print("Initializing ASTER components...")
    hidden_dim = model.config.hidden_size
    # [MODIFIED] Get num_layers from the DistilBERT transformer structure.
    num_layers = len(model.distilbert.transformer.layer)
    model_dtype = next(model.parameters()).dtype

    scorer = ScoringModel(hidden_dim, config.SCORER_HIDDEN_DIM, num_layers, model_dtype).to(device)
    adapter = DynamicAdapter(hidden_dim, config.ADAPTER_BOTTLENECK_DIM, num_layers, model_dtype).to(device)

    scorer = DDP(scorer, device_ids=[rank])
    adapter = DDP(adapter, device_ids=[rank])

    optimizer = AdamW(list(scorer.parameters()) + list(adapter.parameters()), lr=config.LEARNING_RATE * world_size)

    start_epoch = 0
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "aster_bert_checkpoint.pt")
    if args.resume and os.path.exists(checkpoint_path):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        scorer.module.load_state_dict(checkpoint['scorer_state_dict'])
        adapter.module.load_state_dict(checkpoint['adapter_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if rank == 0: print(f"Resumed successfully from epoch {start_epoch}.")
    elif args.resume and rank == 0:
        print(f"WARNING: --resume specified, but no checkpoint found at {checkpoint_path}.")

    if rank == 0: print("Setting up ASTER trainer...")
    aster_trainer = ASTERTrainerBERT(model, scorer, adapter, optimizer, rank, world_size)

    if rank == 0: print(f"Starting distributed training from epoch {start_epoch + 1}...")
    aster_trainer.train(train_dataloader, train_sampler, start_epoch=start_epoch)

    if rank == 0: print("Training finished successfully!")
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASTER-BERT using Distributed Data Parallel.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint.")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Distributed training requires at least 2 GPUs. Please use train_bert.py for single-GPU training.")
        exit()
    else:
        print(f"Found {world_size} GPUs. Starting distributed training...")
        mp.spawn(main_process, args=(world_size, args), nprocs=world_size, join=True)