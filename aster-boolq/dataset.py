import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from typing import Dict, Any
from config import Config
from datasets import load_dataset


class BoolQDataset(Dataset):
    """Dataset for BoolQ with <hcog> token, loaded from HF mirror."""

    def __init__(self, tokenizer, split: str = "train", max_length=Config.MAX_LENGTH, file_path: str = None):
        """
        Initialize BoolQ dataset

        Args:
            tokenizer: Tokenizer for LLAMA 3.1
            split: 'train' or 'validation'
            max_length: Maximum sequence length
            file_path: deprecated; kept for backward compatibility and ignored
        """
        # 使用HF镜像源
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        # 加载HF数据集
        ds = load_dataset("boolq")
        if split not in ds:
            raise ValueError(f"Invalid split: {split}. Available: {list(ds.keys())}")
        self.data = ds[split]

        self.tokenizer = tokenizer
        self.max_length = max_length
        # 确保分词器有填充令牌
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"已将分词器的eos_token设置为pad_token: {self.tokenizer.pad_token}")

        # 将标签映射添加到类属性
        # 统一到baseline：使用带前导空格的" yes"/" no"作为标签token（LLaMA分词对空格敏感）
        self.label_to_token = {
            True: self.tokenizer.encode(" yes", add_special_tokens=False)[0],
            False: self.tokenizer.encode(" no", add_special_tokens=False)[0]
        }

        # 仅调整日志打印，明确显示 ' yes' 与 ' no' 对应的token id（不改训练逻辑）
        print(f"标签映射: ' yes'={self.label_to_token[True]}, ' no'={self.label_to_token[False]}")
        print(f"Loaded {len(self.data)} samples from HF boolq/{split} via {os.environ.get('HF_ENDPOINT')}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Extract question, passage, and answer
        question = item["question"]
        passage = item["passage"]
        answer = bool(item["answer"]) if isinstance(item["answer"], (bool, int)) else (str(item["answer"]).lower() == "true")

        prompt = f"{passage}\nQuestion: {question}?\nAnswer:"

        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 存储token ID供模型生成使用
        # 再次确保标签token与baseline一致（带前导空格）
        self.label_to_token = {
            True: self.tokenizer.encode(" yes", add_special_tokens=False)[0],  # " yes"的token ID
            False: self.tokenizer.encode(" no", add_special_tokens=False)[0]  # " no"的token ID
        }

        # 关键修改：使用类别索引（0="no", 1="yes"）而不是token ID作为标签
        label_idx = 1 if answer else 0

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label_idx, dtype=torch.long),  # 使用类别索引
            "token_label": torch.tensor(self.label_to_token[answer], dtype=torch.long)  # 保留token ID供参考
        }


def create_dataloaders(student_model, teacher_model, world_size=1, rank=0):
    """Create train and validation dataloaders with distributed support"""
    # 使用学生tokenizer
    tokenizer = student_model.tokenizer
    # 确保分词器有填充令牌
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"在数据加载器中设置了pad_token: {tokenizer.pad_token}")

    # 创建数据集（从HF镜像加载）
    train_dataset = BoolQDataset(tokenizer=tokenizer, split="train")
    val_dataset = BoolQDataset(tokenizer=tokenizer, split="validation")

    # 如果是调试模式，仅对训练集采样，验证集保持完整
    if hasattr(Config, 'DEBUG_MODE') and Config.DEBUG_MODE:
        from torch.utils.data import Subset
        import random

        # 固定随机种子以确保可复现性
        rng = random.Random(getattr(Config, 'SEED', 42))

        # 仅对训练集进行下采样
        train_sample_size = int(len(train_dataset) * Config.DEBUG_SAMPLE_RATIO)
        train_indices = rng.sample(range(len(train_dataset)), train_sample_size)
        train_dataset = Subset(train_dataset, train_indices)

        print(f"调试模式: 采样 {len(train_dataset)} 训练样本 ({Config.DEBUG_SAMPLE_RATIO * 100:.1f}%), 验证集保持完整: {len(val_dataset)}")

    # 创建数据加载器，根据是否为分布式环境决定是否使用DistributedSampler
    # DataLoader 可复现性设置
    def _worker_init_fn(worker_id):
        base_seed = getattr(Config, 'SEED', 42)
        seed = base_seed + worker_id + rank * 1000
        import numpy as _np
        import random as _random
        _np.random.seed(seed)
        _random.seed(seed)
        torch.manual_seed(seed)

    g_train = torch.Generator(device='cpu')
    g_val = torch.Generator(device='cpu')
    base = getattr(Config, 'SEED', 42)
    g_train.manual_seed(base + rank * 2 + 0)
    g_val.manual_seed(base + rank * 2 + 1)

    if world_size > 1:
        # 分布式训练 - 使用DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=_worker_init_fn,
            generator=g_train
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=_worker_init_fn,
            generator=g_val
        )

        print(f"Rank {rank}: 训练数据集大小: {len(train_dataset)}, "
              f"实际批次数: {len(train_loader)}")
    else:
        # 单GPU训练
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=_worker_init_fn,
            generator=g_train
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=_worker_init_fn,
            generator=g_val
        )

    return train_loader, val_loader