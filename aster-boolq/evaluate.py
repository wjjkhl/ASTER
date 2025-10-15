import os
import random
import torch
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import pandas as pd

from models import StudentLLAMA
from dataset import BoolQDataset
from utils import (
    AverageMeter, LayerPathTracker, setup_logger,
    plot_layer_visit_rate, plot_path_length_distribution,
    move_batch_to_device, reduce_tensor
)
from config import Config
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def _set_global_seed(seed: int):
    """设置全局随机种子以确保可复现性"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_simplified_checkpoint(model, checkpoint_path, device):
    """加载使用save_simple_checkpoint保存的简化检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 检查是否为简化格式的检查点
    if isinstance(checkpoint.get('student_model', {}), dict) and 'dynamic_adapter' in checkpoint['student_model']:
        # 这是简化检查点，分别加载每个组件
        model.dynamic_adapter.load_state_dict(checkpoint['student_model']['dynamic_adapter'])
        model.layer_embedding.load_state_dict(checkpoint['student_model']['layer_embedding'])
        model.scoring_model.load_state_dict(checkpoint['student_model']['scoring_model'])

        print(f"已加载简化检查点 (epoch {checkpoint.get('epoch', 'unknown')})")
        return checkpoint

    # 这是常规检查点，尝试直接加载
    if hasattr(model, 'use_fsdp') and model.use_fsdp:
        # 使用FSDP加载方法
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.api import StateDictType, ShardedStateDictConfig
        sharded_state_dict_config = ShardedStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_state_dict_config):
            model.load_state_dict(checkpoint['student_model'])
    else:
        # 常规加载
        model.load_state_dict(checkpoint['student_model'])

    print(f"已加载常规检查点 (epoch {checkpoint.get('epoch', 'unknown')})")
    return checkpoint


def evaluate(model, val_loader, device, logger, true_skipping=True,
             max_skip_distance=None, temperature=0.1, rank=0, world_size=1, enforce_min_path=False):
    """Evaluate student model performance with dynamic layer skipping"""
    # 评估前设置随机种子，确保同一权重可复现
    try:
        _set_global_seed(getattr(Config, 'SEED', 42))
    except Exception:
        pass
    model.eval()

    # For tracking metrics
    correct = 0
    total = 0
    # 统计预测为yes/no的数量（0=no，1=yes）
    yes_count = 0
    no_count = 0
    layer_tracker = LayerPathTracker(model.num_layers)

    # 添加路径长度指标
    path_lengths_meter = AverageMeter('Path Length', ':.2f')

    # 添加最小路径长度指标
    min_path_length_met = 0  # 跟踪满足最小路径长度的样本数

    # For detailed analysis
    path_lengths = []
    detailed_paths = []
    layer_times = {i: [] for i in range(model.num_layers + 1)}

    # 创建进度条（只在主进程显示）
    if rank == 0:
        pbar = tqdm(val_loader, desc="Evaluating")
    else:
        pbar = val_loader

    with torch.no_grad():
        for batch in pbar:
            # Move batch to device
            batch = move_batch_to_device(batch, device)
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"]  # 这是类别索引(0/1)，不是token ID

            # Student forward pass with dynamic layer selection
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                temperature=temperature,
                training=False,
                true_skipping=true_skipping,
                max_skip_distance=max_skip_distance,
                enforce_min_path=enforce_min_path  # 新增参数，控制是否强制最小路径长度
            )

            # Get predictions (获取生成概率最高的token - yes或no)
            yes_no_logits = outputs['logits']
            _, preds = torch.max(yes_no_logits, 1)

            # Update metrics
            batch_size = labels.size(0)
            total += batch_size
            batch_correct = (preds == labels).sum().item()
            correct += batch_correct
            # 累计预测分布
            yes_count += (preds == 1).sum().item()
            no_count += (preds == 0).sum().item()

            # 在分布式环境中同步指标
            if world_size > 1:
                # 收集正确预测数和总样本数
                batch_correct_tensor = torch.tensor(batch_correct, device=device)
                batch_size_tensor = torch.tensor(batch_size, device=device)

                dist.all_reduce(batch_correct_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(batch_size_tensor, op=dist.ReduceOp.SUM)

            # Track layer path
            layer_path = outputs['layer_path']
            layer_tracker.update(layer_path)
            current_path_length = len(layer_path)
            path_lengths.extend([current_path_length] * input_ids.size(0))

            # 更新路径长度统计
            path_lengths_meter.update(current_path_length, batch_size)

            # 检查是否满足最小路径长度
            if current_path_length >= Config.MIN_PATH_LENGTH:
                min_path_length_met += batch_size

            # Record detailed path information
            if 'detailed_path' in outputs:
                detailed_paths.append(outputs['detailed_path'])

            # 更新进度条（只在主进程）
            if rank == 0:
                pbar.set_postfix({
                    'acc': f"{correct / total * 100:.2f}%",
                    'path_len': f"{path_lengths_meter.avg:.2f}",
                    'min_met': f"{min_path_length_met / total * 100:.2f}%",
                    'enforce': f"{enforce_min_path}"
                })

    # 在分布式环境中同步总结果
    if world_size > 1:
        total_tensor = torch.tensor(total, device=device)
        correct_tensor = torch.tensor(correct, device=device)
        min_path_met_tensor = torch.tensor(min_path_length_met, device=device)
        yes_tensor = torch.tensor(yes_count, device=device)
        no_tensor = torch.tensor(no_count, device=device)

        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(min_path_met_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(yes_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(no_tensor, op=dist.ReduceOp.SUM)

        total = total_tensor.item()
        correct = correct_tensor.item()
        min_path_length_met = min_path_met_tensor.item()
        yes_count = yes_tensor.item()
        no_count = no_tensor.item()

    # Calculate overall statistics
    accuracy = 100 * correct / total if total > 0 else 0
    path_stats = layer_tracker.get_stats()
    min_path_met_percent = 100 * min_path_length_met / total if total > 0 else 0

    # 同步path_stats在所有进程间的数据
    if world_size > 1:
        # 所有进程将路径统计数据发送到主进程
        if rank == 0:
            all_layer_visits = [torch.zeros_like(torch.tensor(path_stats["layer_visit_rate"], device=device))
                                for _ in range(world_size)]
            dist.gather(torch.tensor(path_stats["layer_visit_rate"], device=device), all_layer_visits)

            # 合并数据
            path_stats["layer_visit_rate"] = np.mean([visits.cpu().numpy() for visits in all_layer_visits], axis=0)
        else:
            dist.gather(torch.tensor(path_stats["layer_visit_rate"], device=device), None)

    # Get latest layer skipping stats
    skip_stats = model.get_skip_stats()

    # 只在主进程显示结果
    if rank == 0:
        mode_str = "强制最小路径长度模式" if enforce_min_path else "常规模式"
        logger.info(f'评估结果 ({mode_str}):')
        logger.info(f'准确率: {accuracy:.2f}%')
        if total > 0:
            yes_ratio = 100.0 * yes_count / total
            no_ratio = 100.0 * no_count / total
            logger.info(f'预测分布: yes={yes_ratio:.2f}%, no={no_ratio:.2f}%')
        logger.info(f'平均路径长度: {path_stats["avg_path_length"]:.2f} 层')
        logger.info(f'计算量: {path_stats["avg_computation"] * 100:.2f}% 相比完整模型')
        logger.info(f'满足最小路径长度 ({Config.MIN_PATH_LENGTH}) 的样本: {min_path_met_percent:.2f}%')

        # Output skipping details
        if true_skipping:
            skipped = skip_stats.get("skipped_layers", [])
            visited = skip_stats.get("visited_layers", [])
            logger.info(f'层跳跃详情 - 跳过: {len(skipped)}, 访问: {len(visited)}')

            if detailed_paths:
                logger.info(f'样本路径: {detailed_paths[-1]}')

    # Create path length histogram
    path_length_counts = np.bincount(path_lengths, minlength=model.num_layers + 2)

    return {
        'accuracy': accuracy,
        'path_stats': path_stats,
        'layer_visit_rate': path_stats['layer_visit_rate'],
        'path_length_counts': path_length_counts,
        'detailed_paths': detailed_paths[:5],  # Save first 5 path details
        'min_path_length_met_percent': min_path_met_percent,  # 添加最小路径长度满足比例
        'enforce_min_path': enforce_min_path,  # 记录是否强制最小路径长度
        'yes_count': yes_count,
        'no_count': no_count,
        'yes_ratio': (100.0 * yes_count / total) if total > 0 else 0.0,
        'no_ratio': (100.0 * no_count / total) if total > 0 else 0.0
    }


def main():
    """Main evaluation function"""
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Dynamic Layer Skipping Model")
    parser.add_argument('--checkpoint_path', type=str,
                        default='/cephfs/shared/wjj/ICLR/LLAMAtest/dynamic_llama_output/checkpoint_epoch50.pth.tar',
                        help='Path to trained student model checkpoint')
    parser.add_argument('--true_skipping', action='store_true', default=True,
                        help='Enable true layer skipping during evaluation')
    parser.add_argument('--max_skip_distance', type=int, default=None,
                        help='Maximum allowed skip distance; if None, derive from checkpoint epoch')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for layer selection (lower = more greedy)')
    parser.add_argument('--output_dir', type=str, default='./evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--enforce_min_path', action='store_true', default=False,
                        help='Enforce minimum path length during validation')
    args = parser.parse_args()

    # Initialize distributed environment
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        # 初始化进程组
        if world_size > 1:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
            print(f"进程 {rank}/{world_size} 使用 GPU: {local_rank}")
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    # 设备设置
    device = torch.device(f"cuda:{local_rank}")

    # Create output directory (只在主进程创建)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # Set up logger
    if rank == 0:
        logger = setup_logger(os.path.join(args.output_dir, 'evaluation.log'))
        logger.info(f'Arguments: {args}')
        logger.info(f'Distributed: {world_size} processes, current rank: {rank}')
        logger.info(f'最小路径长度设置为: {Config.MIN_PATH_LENGTH}')
        logger.info(f'是否强制最小路径长度: {args.enforce_min_path}')
    else:
        logger = setup_logger()
        logger.setLevel('WARNING')

    # 设置随机种子（主入口）
    try:
        _set_global_seed(getattr(Config, 'SEED', 42))
    except Exception:
        pass

    # Create student model
    logger.info(f"Creating student model on GPU {local_rank}...")
    student_model = StudentLLAMA(
        base_model_path=Config.MODEL_PATH,
        layer_embed_dim=Config.LAYER_EMBED_DIM,
        adapter_base_size=Config.ADAPTER_BASE_SIZE,
        device=device,
        use_fsdp=(world_size > 1)
    )

    # Load checkpoint if provided - 修改部分
    if args.checkpoint_path:
        logger.info(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = load_simplified_checkpoint(student_model, args.checkpoint_path, device)

        # 显示检查点信息
        if rank == 0:
            if 'epoch' in checkpoint:
                logger.info(f"Checkpoint from epoch {checkpoint['epoch']}")
            if 'best_acc' in checkpoint:
                logger.info(f"Best accuracy during training: {checkpoint['best_acc']:.4f}")

        # 如果未显式指定max_skip_distance，则从checkpoint的epoch推断
        try:
            ckpt_epoch = checkpoint.get('epoch', None)
        except Exception:
            ckpt_epoch = None
        if args.max_skip_distance is None:
            derived = Config.get_max_skip_distance(ckpt_epoch if ckpt_epoch is not None else Config.EPOCHS)
            if rank == 0:
                logger.info(f"Deriving max_skip_distance from checkpoint epoch {ckpt_epoch}: {derived}")
            args.max_skip_distance = derived

        logger.info("Checkpoint loaded successfully")

    # Create val dataset
    logger.info("Creating validation dataset...")
    val_dataset = BoolQDataset(
        tokenizer=student_model.tokenizer,
        split="validation",
        max_length=Config.MAX_LENGTH
    )

    # 创建数据加载器，使用DistributedSampler实现分布式评估
    if world_size > 1:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

        # 为可复现性配置worker种子与生成器
        def _worker_init_fn(worker_id):
            base_seed = getattr(Config, 'SEED', 42)
            seed = base_seed + worker_id + rank * 1000
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        g = torch.Generator(device='cpu')
        g.manual_seed(getattr(Config, 'SEED', 42) + rank)

        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=_worker_init_fn,
            generator=g
        )
    else:
        def _worker_init_fn(worker_id):
            base_seed = getattr(Config, 'SEED', 42)
            seed = base_seed + worker_id
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        g = torch.Generator(device='cpu')
        g.manual_seed(getattr(Config, 'SEED', 42))

        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=_worker_init_fn,
            generator=g
        )

    # 进行两次评估，一次正常模式，一次强制最小路径长度
    if rank == 0:
        logger.info("开始第一次评估 (常规模式)...")

    results_normal = evaluate(
        model=student_model,
        val_loader=val_loader,
        device=device,
        logger=logger,
        true_skipping=args.true_skipping,
        max_skip_distance=args.max_skip_distance,
        temperature=args.temperature,
        rank=rank,
        world_size=world_size,
        enforce_min_path=False  # 常规模式
    )

    if rank == 0:
        logger.info("开始第二次评估 (强制最小路径长度模式)...")

    results_enforced = evaluate(
        model=student_model,
        val_loader=val_loader,
        device=device,
        logger=logger,
        true_skipping=args.true_skipping,
        max_skip_distance=args.max_skip_distance,
        temperature=args.temperature,
        rank=rank,
        world_size=world_size,
        enforce_min_path=True  # 强制最小路径长度
    )

    # 只在主进程绘图和保存结果
    if rank == 0:
        # 分别保存两种模式的结果

        # 常规模式图表
        plot_layer_visit_rate(
            results_normal['layer_visit_rate'],
            os.path.join(args.output_dir, 'layer_visit_rate_normal.png')
        )

        plot_path_length_distribution(
            results_normal['path_length_counts'],
            student_model.num_layers + 1,
            os.path.join(args.output_dir, 'path_length_distribution_normal.png')
        )

        # 强制模式图表
        plot_layer_visit_rate(
            results_enforced['layer_visit_rate'],
            os.path.join(args.output_dir, 'layer_visit_rate_enforced.png')
        )

        plot_path_length_distribution(
            results_enforced['path_length_counts'],
            student_model.num_layers + 1,
            os.path.join(args.output_dir, 'path_length_distribution_enforced.png')
        )

        # 保存结果
        np.save(os.path.join(args.output_dir, 'evaluation_results_normal.npy'), results_normal)
        np.save(os.path.join(args.output_dir, 'evaluation_results_enforced.npy'), results_enforced)

        # 生成性能-效率权衡分析
        logger.info(f'性能-效率权衡分析 (常规模式):')
        logger.info(f'准确率: {results_normal["accuracy"]:.2f}% 使用 '
                    f'{results_normal["path_stats"]["avg_computation"] * 100:.2f}% 计算量')
        logger.info(f'最小路径长度 ({Config.MIN_PATH_LENGTH}) 满足率: '
                    f'{results_normal["min_path_length_met_percent"]:.2f}% 样本')

        logger.info(f'性能-效率权衡分析 (强制最小路径长度模式):')
        logger.info(f'准确率: {results_enforced["accuracy"]:.2f}% 使用 '
                    f'{results_enforced["path_stats"]["avg_computation"] * 100:.2f}% 计算量')
        logger.info(f'最小路径长度 ({Config.MIN_PATH_LENGTH}) 满足率: '
                    f'{results_enforced["min_path_length_met_percent"]:.2f}% 样本')

        # 模式比较
        logger.info(f'模式对比:')
        logger.info(f'准确率差异: {results_enforced["accuracy"] - results_normal["accuracy"]:.2f}%')
        logger.info(
            f'计算量差异: {(results_enforced["path_stats"]["avg_computation"] - results_normal["path_stats"]["avg_computation"]) * 100:.2f}%')
        logger.info(
            f'路径长度差异: {results_enforced["path_stats"]["avg_path_length"] - results_normal["path_stats"]["avg_path_length"]:.2f} 层')

        # 计算帕累托效率
        if results_normal["path_stats"]["avg_computation"] < 1.0:  # 如果有计算节省
            normal_efficiency_gain = 1.0 - results_normal["path_stats"]["avg_computation"]
            teacher_acc = 100.0  # 假设完整模型精度为100%，用于相对比较
            normal_accuracy_loss = teacher_acc - results_normal["accuracy"]

            # 每节省1%计算量损失的精度
            if normal_efficiency_gain > 0:
                normal_tradeoff_ratio = normal_accuracy_loss / (normal_efficiency_gain * 100)
                logger.info(f'常规模式: 每节省1%计算量，准确率约降低 {normal_tradeoff_ratio:.4f}%')

        if results_enforced["path_stats"]["avg_computation"] < 1.0:
            enforced_efficiency_gain = 1.0 - results_enforced["path_stats"]["avg_computation"]
            teacher_acc = 100.0
            enforced_accuracy_loss = teacher_acc - results_enforced["accuracy"]

            if enforced_efficiency_gain > 0:
                enforced_tradeoff_ratio = enforced_accuracy_loss / (enforced_efficiency_gain * 100)
                logger.info(f'强制最小路径长度模式: 每节省1%计算量，准确率约降低 {enforced_tradeoff_ratio:.4f}%')

        logger.info(f'评估完成。结果已保存到 {args.output_dir}')

    # 同步所有进程
    if world_size > 1:
        dist.barrier()

    # 返回两种模式的结果
    return results_normal, results_enforced


if __name__ == '__main__':
    main()