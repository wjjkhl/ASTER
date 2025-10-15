#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import logging
import argparse
import datetime
import numpy as np
from tqdm import tqdm

from models import StudentLLAMA, TeacherLLAMA
from dataset import create_dataloaders
from loss import TotalLoss
from config import Config
from utils import LayerPathTracker, AverageMeter, TemperatureScheduler, setup_logger, move_batch_to_device


def save_simple_checkpoint(model, optimizer, epoch, best_acc=0, filename='checkpoint.pt'):
    """简单保存检查点，避免FSDP复杂性"""
    if dist.get_rank() == 0:
        state = {
            'epoch': epoch,
            'best_acc': best_acc,
            'student_model': {
                'dynamic_adapter': model.dynamic_adapter.state_dict(),
                'layer_embedding': model.layer_embedding.state_dict(),
                'scoring_model': model.scoring_model.state_dict()
            },
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, filename)
        print(f"检查点已保存到 {filename}")


def train_one_epoch(student_model, teacher_model, train_loader, criterion, optimizer,
                    scaler, epoch, device, max_skip_distance=2):
    """使用混合精度训练一个epoch"""
    student_model.train()
    teacher_model.eval()

    losses = AverageMeter('Loss', ':.4f')
    accs = AverageMeter('Acc', ':.4f')
    path_lengths = AverageMeter('Path Length', ':.2f')  # 添加路径长度指标
    length_penalties = AverageMeter('Length Penalty', ':.4f')  # 添加路径长度惩罚指标
    layer_tracker = LayerPathTracker(student_model.num_layers)

    # 使用tqdm显示进度条
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    # 梯度累积设置
    accumulated_steps = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        # 移动数据到设备
        batch = move_batch_to_device(batch, device)

        # 使用混合精度训练
        with autocast():
            # 教师模型前向传播
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )

            # 学生模型前向传播
            student_outputs = student_model(
                batch['input_ids'],
                attention_mask=batch['attention_mask'],
                temperature=1.0,
                training=True,
                true_skipping=True,
                max_skip_distance=max_skip_distance
            )

            # 计算损失
            layer_path = student_outputs['layer_path']

            try:
                loss_dict = criterion(
                    student_outputs,
                    teacher_outputs,
                    batch['label'],
                    layer_path,
                    t=batch_idx + 1,
                    T=len(train_loader)
                )

                # 检查NaN损失
                if torch.isnan(loss_dict['total_loss']):
                    print(f"警告: 检测到NaN损失，批次{batch_idx}，跳过该批次")
                    continue

                loss = loss_dict['total_loss'] / Config.GRADIENT_ACCUMULATION_STEPS

                # 使用scaler进行反向传播
                scaler.scale(loss).backward()
            except RuntimeError as e:
                error_msg = str(e)
                if "element 0 of tensors does not require grad" in error_msg or "does not have a grad_fn" in error_msg:
                    print(f"警告: 批次{batch_idx}发生梯度计算错误，跳过该批次: {error_msg}")
                    # 清理梯度并跳过此批次
                    optimizer.zero_grad()
                    continue
                else:
                    # 重新抛出其他类型的错误
                    raise

        # 梯度累积
        accumulated_steps += 1
        if accumulated_steps >= Config.GRADIENT_ACCUMULATION_STEPS:
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in student_model.parameters() if p.requires_grad],
                Config.MAX_GRAD_NORM
            )

            # 更新权重
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            accumulated_steps = 0

        # 更新指标
        losses.update(loss_dict['total_loss'].item(), batch['input_ids'].size(0))
        accs.update(loss_dict['accuracy'], batch['input_ids'].size(0))

        # 更新路径长度统计
        current_path_length = len(layer_path)
        path_lengths.update(current_path_length, batch['input_ids'].size(0))

        # 更新路径长度惩罚指标
        if 'path_length_penalty' in loss_dict:
            length_penalties.update(loss_dict['path_length_penalty'].item(), batch['input_ids'].size(0))

        # 更新层路径统计
        layer_tracker.update(layer_path)

        # 更新进度条信息
        pbar.set_postfix({
            'loss': f"{losses.avg:.4f}",
            'acc': f"{accs.avg:.4f}",
            'len': f"{path_lengths.avg:.2f}",
            'pen': f"{length_penalties.avg:.4f}"  # 显示路径长度惩罚
        })

    # 获取路径统计
    path_stats = layer_tracker.get_stats()

    return {
        'loss': losses.avg,
        'acc': accs.avg,
        'path_stats': path_stats,
        'path_length': path_lengths.avg,
        'length_penalty': length_penalties.avg
    }


def validate(student_model, teacher_model, val_loader, criterion, device, max_skip_distance=2, enforce_min_path=False):
    """评估模型性能，支持强制最小路径长度"""
    student_model.eval()
    teacher_model.eval()

    losses = AverageMeter('Loss', ':.4f')
    accs = AverageMeter('Acc', ':.4f')
    path_lengths = AverageMeter('Path Length', ':.2f')  # 添加路径长度指标
    layer_tracker = LayerPathTracker(student_model.num_layers)

    pbar = tqdm(val_loader, desc="Validation")

    # 记录满足最小路径长度的样本数
    min_path_met = 0
    total_samples = 0

    with torch.no_grad():
        for batch in pbar:
            # 移动数据到设备
            batch = move_batch_to_device(batch, device)
            total_samples += batch['input_ids'].size(0)

            # 教师模型前向传播
            teacher_outputs = teacher_model(
                batch['input_ids'],
                attention_mask=batch['attention_mask']
            )

            # 学生模型前向传播
            student_outputs = student_model(
                batch['input_ids'],
                attention_mask=batch['attention_mask'],
                temperature=0.1,  # 评估时使用较低温度
                training=False,
                true_skipping=True,
                max_skip_distance=max_skip_distance,
                enforce_min_path=enforce_min_path  # 新增参数，控制是否强制最小路径长度
            )

            # 计算损失
            layer_path = student_outputs['layer_path']
            loss_dict = criterion(
                student_outputs,
                teacher_outputs,
                batch['label'],
                layer_path,
                t=1,
                T=1
            )

            # 更新指标
            losses.update(loss_dict['total_loss'].item(), batch['input_ids'].size(0))
            accs.update(loss_dict['accuracy'], batch['input_ids'].size(0))

            # 更新路径长度统计
            current_path_length = len(layer_path)
            path_lengths.update(current_path_length, batch['input_ids'].size(0))

            # 检查是否满足最小路径长度
            if current_path_length >= Config.MIN_PATH_LENGTH:
                min_path_met += batch['input_ids'].size(0)

            # 更新层路径统计
            layer_tracker.update(layer_path)

            # 更新进度条
            min_path_percent = 100 * min_path_met / total_samples if total_samples > 0 else 0
            pbar.set_postfix({
                'val_loss': f"{losses.avg:.4f}",
                'val_acc': f"{accs.avg:.4f}",
                'val_len': f"{path_lengths.avg:.2f}",
                'min_met': f"{min_path_percent:.1f}%"
            })

    # 获取路径统计
    path_stats = layer_tracker.get_stats()

    # 计算满足最小路径长度的百分比
    min_path_met_percent = 100 * min_path_met / total_samples if total_samples > 0 else 0

    return {
        'loss': losses.avg,
        'acc': accs.avg,
        'path_stats': path_stats,
        'path_length': path_lengths.avg,
        'min_path_met_percent': min_path_met_percent
    }


def main():
    """主函数"""
    # 初始化分布式环境
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        # 初始化进程组
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    # 设置设备
    device = torch.device(f"cuda:{local_rank}")

    # 设置日志
    if rank == 0:
        logger = setup_logger(os.path.join(Config.OUTPUT_DIR, 'debug_train.log'))
        logger.info(f"分布式训练: {world_size} 进程, 当前进程: {rank}")
        logger.info(f"最小路径长度设置为: {Config.MIN_PATH_LENGTH}")
        logger.info(f"路径长度惩罚系数设置为: {Config.LENGTH_PENALTY_ALPHA}")
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.WARNING)

    # 创建模型
    if rank == 0:
        logger.info("创建教师模型...")
    teacher_model = TeacherLLAMA(base_model_path=Config.MODEL_PATH, device=device)

    if rank == 0:
        logger.info("创建学生模型...")
    student_model = StudentLLAMA(
        base_model_path=Config.MODEL_PATH,
        layer_embed_dim=Config.LAYER_EMBED_DIM,
        adapter_base_size=Config.ADAPTER_BASE_SIZE,
        device=device,
        use_fsdp=True  # 仍然使用FSDP但简化保存方法
    )

    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(student_model, teacher_model, world_size, rank)

    # 创建损失函数
    criterion = TotalLoss(
        num_layers=student_model.num_layers,
        kd_weight=Config.KD_WEIGHT,
        policy_weight=Config.POLICY_WEIGHT
    )
    criterion = criterion.to(device)

    # 创建优化器
    optimizer = optim.AdamW(
        [p for p in student_model.parameters() if p.requires_grad],
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )

    # 创建混合精度训练的scaler
    scaler = GradScaler()

    # 训练循环
    best_acc = 0.0
    num_epochs = 100  # 调试模式下减少epoch数

    for epoch in range(1, num_epochs + 1):
        if rank == 0:
            logger.info(f"开始第 {epoch}/{num_epochs} 轮训练")

        # 设置最大跳过距离
        max_skip_distance = min(epoch + 1, Config.MAX_DISTANCE)
        if rank == 0:
            logger.info(f"使用最大跳过距离: {max_skip_distance}")

        # 训练一个epoch
        try:
            train_stats = train_one_epoch(
                student_model=student_model,
                teacher_model=teacher_model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                device=device,
                max_skip_distance=max_skip_distance
            )
        except Exception as e:
            if rank == 0:
                logger.error(f"训练过程中发生错误: {str(e)}")

            # 尝试继续验证
            if rank == 0:
                logger.info("尝试继续进行验证...")
            train_stats = {'loss': float('nan'), 'acc': 0.0, 'path_stats': {}, 'path_length': 0.0,
                           'length_penalty': 0.0}

        # 验证 - 分两次进行，一次不强制最小路径长度，一次强制
        val_stats_normal = validate(
            student_model=student_model,
            teacher_model=teacher_model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            max_skip_distance=max_skip_distance,
            enforce_min_path=False  # 不强制最小路径长度
        )

        val_stats_enforced = validate(
            student_model=student_model,
            teacher_model=teacher_model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            max_skip_distance=max_skip_distance,
            enforce_min_path=True  # 强制最小路径长度
        )

        # 记录结果
        if rank == 0:
            logger.info(
                f"Epoch {epoch} - "
                f"Train: loss={train_stats['loss']:.4f}, acc={train_stats['acc']:.4f}, "
                f"path_len={train_stats['path_length']:.2f}, penalty={train_stats.get('length_penalty', 0):.4f} | "
                f"Val(normal): loss={val_stats_normal['loss']:.4f}, acc={val_stats_normal['acc']:.4f}, "
                f"path_len={val_stats_normal['path_length']:.2f}, min_met={val_stats_normal['min_path_met_percent']:.1f}% | "
                f"Val(enforced): loss={val_stats_enforced['loss']:.4f}, acc={val_stats_enforced['acc']:.4f}, "
                f"path_len={val_stats_enforced['path_length']:.2f}, min_met={val_stats_enforced['min_path_met_percent']:.1f}%"
            )

            # 保存检查点 - 使用正常验证结果决定是否为最佳模型
            is_best = val_stats_normal['acc'] > best_acc
            best_acc = max(val_stats_normal['acc'], best_acc)

            save_simple_checkpoint(
                model=student_model,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc,
                filename=os.path.join(Config.OUTPUT_DIR, f'checkpoint_epoch{epoch}.pt')
            )

            if is_best:
                import shutil
                shutil.copyfile(
                    os.path.join(Config.OUTPUT_DIR, f'checkpoint_epoch{epoch}.pt'),
                    os.path.join(Config.OUTPUT_DIR, 'best_model.pt')
                )

    if rank == 0:
        logger.info(f"训练完成！最佳验证准确率: {best_acc:.4f}")

    # 清理分布式环境
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # 运行主函数
    main()