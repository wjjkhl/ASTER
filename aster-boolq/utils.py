import os
import logging
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (
    FullStateDictConfig, StateDictType
)
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from config import Config


class TemperatureScheduler:
    """Temperature scheduler for exploration-exploitation balance"""

    def __init__(self, init_temp=Config.INIT_TEMP, min_temp=Config.MIN_TEMP,
                 decay_factor=Config.TEMP_DECAY):
        """
        Initialize temperature scheduler

        Args:
            init_temp: Initial temperature
            min_temp: Minimum temperature
            decay_factor: Decay rate per epoch
        """
        self.init_temp = init_temp
        self.min_temp = min_temp
        self.decay_factor = decay_factor
        self.current_temp = init_temp

    def step(self):
        """Decay temperature"""
        self.current_temp = max(self.min_temp, self.current_temp * self.decay_factor)
        return self.current_temp

    def get_temperature(self):
        """Get current temperature"""
        return self.current_temp

    def reset(self):
        """Reset temperature to initial value"""
        self.current_temp = self.init_temp


class AverageMeter:
    """Compute and store the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class LayerPathTracker:
    """Track and analyze layer skipping paths"""

    def __init__(self, num_layers):
        """
        Initialize layer path tracker

        Args:
            num_layers: Number of model layers
        """
        self.num_layers = num_layers
        self.layer_visits = np.zeros(num_layers + 1)  # +1 for embedding layer
        self.total_samples = 0
        self.path_lengths = []
        self.min_path_met = 0  # 添加记录满足最小路径长度的样本数

    def update(self, layer_path):
        """
        Update statistics with a new layer path

        Args:
            layer_path: List of layer indices
        """
        for layer in layer_path:
            self.layer_visits[layer] += 1

        path_length = len(layer_path)
        self.path_lengths.append(path_length)
        self.total_samples += 1

        # 检查是否满足最小路径长度要求
        if path_length >= Config.MIN_PATH_LENGTH:
            self.min_path_met += 1

    def get_stats(self):
        """Return layer path statistics"""
        if self.total_samples == 0:
            return {
                'layer_visit_rate': np.zeros(self.num_layers + 1),
                'avg_path_length': 0,
                'avg_computation': 0,
                'min_path_met_percent': 0
            }

        layer_visit_rate = self.layer_visits / self.total_samples
        avg_path_length = np.mean(self.path_lengths)
        # Computation savings compared to full model
        avg_computation = avg_path_length / (self.num_layers + 1)
        # 计算满足最小路径长度的百分比
        min_path_met_percent = 100 * self.min_path_met / self.total_samples if self.total_samples > 0 else 0

        return {
            'layer_visit_rate': layer_visit_rate,
            'avg_path_length': avg_path_length,
            'avg_computation': avg_computation,
            'min_path_met_percent': min_path_met_percent
        }

    def reset(self):
        """Reset all statistics"""
        self.layer_visits = np.zeros(self.num_layers + 1)
        self.total_samples = 0
        self.path_lengths = []
        self.min_path_met = 0

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (
    FullStateDictConfig, ShardedStateDictConfig, StateDictType
)


def save_fsdp_checkpoint(model, optimizer, lr_scheduler, epoch, temp_scheduler,
                         best_acc, train_stats, val_stats, config, is_best,
                         filename, best_filename, use_fsdp=False):
    """保存FSDP模型检查点"""

    state = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'temp_scheduler': temp_scheduler,
        'best_acc': best_acc,
        'train_stats': train_stats,
        'val_stats': val_stats,
        'config': config
    }

    if use_fsdp:
        # 仅base_model使用FSDP；可训练模块未FSDP包装，可直接保存简化权重
        target = model.module if hasattr(model, 'module') else model
        state['student_model'] = {
            'dynamic_adapter': target.dynamic_adapter.state_dict(),
            'layer_embedding': target.layer_embedding.state_dict(),
            'scoring_model': target.scoring_model.state_dict(),
        }

        is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        if is_rank0:
            torch.save(state, filename)
            if is_best:
                import shutil
                shutil.copyfile(filename, best_filename)
    else:
        # 非FSDP模式
        target = model.module if hasattr(model, 'module') else model
        state['student_model'] = target.state_dict()
        torch.save(state, filename)
        if is_best:
            import shutil
            shutil.copyfile(filename, best_filename)


def load_fsdp_checkpoint(model, checkpoint_path, device):
    """加载FSDP模型检查点"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if hasattr(model, 'use_fsdp') and model.use_fsdp:
        # 同样使用ShardedStateDictConfig
        sharded_state_dict_config = ShardedStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_state_dict_config):
            model.load_state_dict(checkpoint['student_model'])
    else:
        model.load_state_dict(checkpoint['student_model'])

    return checkpoint


def reduce_tensor(tensor, world_size):
    """将张量在所有进程中平均"""
    if world_size == 1:
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def setup_logger(log_file=None):
    """Set up logger for training"""
    logger = logging.getLogger('dynamic_layer_skipping')
    logger.setLevel(logging.INFO)

    # 清除之前的处理器
    if logger.handlers:
        logger.handlers = []

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # 文件处理器（如果指定）
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_serializable_config(config_class):
    """将Config类转换为可序列化的字典，排除不可序列化的属性"""
    result = {}
    for k, v in vars(config_class).items():
        if not k.startswith('__'):
            if isinstance(v, torch.device):
                result[k] = str(v)  # 转换设备为字符串
            elif callable(v):
                continue  # 跳过方法和可调用对象
            else:
                try:
                    # 简单测试是否可序列化
                    import pickle
                    pickle.dumps(v)
                    result[k] = v
                except:
                    result[k] = str(v)  # 回退：转换为字符串
    return result


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    """Save checkpoint"""
    torch.save(state, filename)
    if is_best:
        import shutil
        shutil.copyfile(filename, best_filename)


def plot_layer_visit_rate(visit_rates, output_path):
    """Plot layer visit rate"""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(visit_rates)), visit_rates)

    # Add percentage labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{visit_rates[i] * 100:.1f}%', ha='center', va='bottom')

    plt.xlabel('Layer Index')
    plt.ylabel('Visit Rate')
    plt.title('Layer Visit Rate')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_path)
    plt.close()


def plot_path_length_distribution(path_lengths, num_layers, output_path):
    """Plot distribution of path lengths"""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(path_lengths)), path_lengths)

    # Add count labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{int(height)}', ha='center', va='bottom')

    plt.xlabel('Path Length')
    plt.ylabel('Count')
    plt.title('Distribution of Path Lengths')
    plt.xticks(range(len(path_lengths)))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_path)
    plt.close()


def synchronize_devices(teacher_device, student_device):
    """同步设备，避免GPU内存冲突"""
    # 确保PyTorch CUDA在两个设备上都初始化
    torch.cuda.set_device(teacher_device.index)
    torch.cuda.empty_cache()  # 清空教师设备的缓存

    torch.cuda.set_device(student_device.index)
    torch.cuda.empty_cache()  # 清空学生设备的缓存

    print(f"设备已同步: {teacher_device} 和 {student_device}")

    # 返回设备内存状态
    memory_info = {
        "teacher": torch.cuda.get_device_properties(teacher_device).total_memory,
        "student": torch.cuda.get_device_properties(student_device).total_memory,
        "teacher_free": torch.cuda.memory_reserved(teacher_device.index),
        "student_free": torch.cuda.memory_reserved(student_device.index)
    }

    return memory_info


def move_batch_to_device(batch, device):
    """Move batch to specified device"""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}