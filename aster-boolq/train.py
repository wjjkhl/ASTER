import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

from models import StudentLLAMA, TeacherLLAMA
from dataset import create_dataloaders
from loss import TotalLoss
from utils import (
    TemperatureScheduler, AverageMeter, LayerPathTracker,
    setup_logger, save_checkpoint, plot_layer_visit_rate,
    plot_path_length_distribution, move_batch_to_device,
    get_serializable_config, reduce_tensor
)
from config import Config
from evaluate import evaluate as eval_inference


def train_epoch(student_model, teacher_model, train_loader, criterion, optimizer, temp_scheduler,
                epoch, logger, layer_tracker, max_skip_distance, rank, world_size, device):
    """训练一个epoch，支持分布式训练"""
    student_model.train()
    teacher_model.eval()

    # 设置分布式采样器的epoch（确保不同进程使用不同的数据）
    if hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)

    # Metrics
    train_loss = AverageMeter('Train Loss', ':.4f')
    train_acc = AverageMeter('Train Acc', ':.4f')
    train_kd_loss = AverageMeter('KD Loss', ':.4f')
    train_policy_loss = AverageMeter('Policy Loss', ':.4f')
    train_rewards = AverageMeter('Rewards', ':.4f')
    train_path_lengths = AverageMeter('Path Length', ':.2f')  # 添加路径长度指标
    train_length_penalties = AverageMeter('Length Penalty', ':.4f')  # 添加路径长度惩罚指标
    layer_tracker.reset()

    # Number of iterations in epoch (for TIDR)
    T = len(train_loader)

    # Determine if true skipping should be used
    use_true_skipping = True  # Enable after 10 epochs

    # Gradient accumulation setup
    optimizer.zero_grad()
    accumulated_steps = 0

    # 在循环开始前添加同步点
    if world_size > 1:
        dist.barrier()

    # 处理tqdm进度条（只在主进程显示）
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    else:
        pbar = train_loader

    # Process each batch
    for batch_idx, batch in enumerate(pbar):
        # 移动数据到设备
        # 注意：在FSDP中，教师和学生使用相同的设备
        batch = move_batch_to_device(batch, device)

        # Current step (for TIDR)
        t = batch_idx + 1

        # Current temperature for exploration-exploitation
        temperature = temp_scheduler.get_temperature()

        # Teacher forward pass (always processes all layers)
        with torch.no_grad():
            teacher_outputs = teacher_model(
                batch['input_ids'],
                attention_mask=batch['attention_mask']
            )

        # Student forward pass with layer skipping
        student_outputs = student_model(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            temperature=temperature,
            training=True,
            true_skipping=use_true_skipping,
            max_skip_distance=max_skip_distance
        )

        # Calculate loss
        layer_path = student_outputs['layer_path']
        loss_dict = criterion(
            student_outputs,
            teacher_outputs,
            batch['label'],
            layer_path,
            t=t,
            T=T
        )

        # 检查NaN损失
        if torch.isnan(loss_dict['total_loss']):
            if rank == 0:
                logger.warning(f"检测到NaN损失，批次{batch_idx}，跳过该批次")
            continue

        # Scale loss for gradient accumulation
        scaled_loss = loss_dict['total_loss'] / Config.GRADIENT_ACCUMULATION_STEPS

        # Backward pass
        scaled_loss.backward()

        # Accumulate gradients
        accumulated_steps += 1

        # Update weights if accumulated enough steps
        if accumulated_steps >= Config.GRADIENT_ACCUMULATION_STEPS:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in student_model.parameters() if p.requires_grad],
                Config.MAX_GRAD_NORM
            )

            # Update weights
            optimizer.step()
            optimizer.zero_grad()
            accumulated_steps = 0

        # 在分布式环境中，同步指标
        if world_size > 1:
            loss_item = reduce_tensor(loss_dict['total_loss'].detach(), world_size)
            # 避免从tensor再次构造tensor导致的UserWarning，直接用张量或将标量转张量
            acc_src = loss_dict['accuracy']
            acc_tensor = acc_src if isinstance(acc_src, torch.Tensor) else torch.tensor(acc_src, device=device)
            acc_item = reduce_tensor(acc_tensor.detach(), world_size)
            kd_loss_item = reduce_tensor(loss_dict['kd_loss'].detach(), world_size)

            if 'policy_loss' in loss_dict:
                policy_loss_item = reduce_tensor(loss_dict['policy_loss'].detach(), world_size)
            else:
                policy_loss_item = torch.tensor(0.0, device=device)

            if 'path_length_penalty' in loss_dict:
                length_penalty_item = reduce_tensor(loss_dict['path_length_penalty'].detach(), world_size)
            else:
                length_penalty_item = torch.tensor(0.0, device=device)
        else:
            loss_item = loss_dict['total_loss'].item()
            acc_item = loss_dict['accuracy']
            kd_loss_item = loss_dict['kd_loss'].item()

            if 'policy_loss' in loss_dict:
                policy_loss_item = loss_dict['policy_loss'].item()
            else:
                policy_loss_item = 0.0

            if 'path_length_penalty' in loss_dict:
                length_penalty_item = loss_dict['path_length_penalty'].item()
            else:
                length_penalty_item = 0.0

        # Update metrics
        train_loss.update(loss_item, batch['input_ids'].size(0))
        train_acc.update(acc_item, batch['input_ids'].size(0))
        train_kd_loss.update(kd_loss_item, batch['input_ids'].size(0))
        train_policy_loss.update(policy_loss_item, batch['input_ids'].size(0))
        train_length_penalties.update(length_penalty_item, batch['input_ids'].size(0))

        # 更新路径长度统计
        current_path_length = len(layer_path)
        train_path_lengths.update(current_path_length, batch['input_ids'].size(0))

        if 'rewards' in loss_dict and loss_dict['rewards']:
            avg_reward = sum(loss_dict['rewards']) / len(loss_dict['rewards'])
            train_rewards.update(avg_reward, batch['input_ids'].size(0))

        # Track layer paths
        layer_tracker.update(layer_path)

        # Log progress (只在主进程)
        if rank == 0 and (batch_idx + 1) % 10 == 0:
            # Get skipping stats
            skip_stats = student_model.get_skip_stats()
            skipped = skip_stats.get("skipped_layers", [])
            visited = skip_stats.get("visited_layers", [])

            logger.info(f'Epoch: [{epoch}][{batch_idx + 1}/{len(train_loader)}] '
                        f'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                        f'Acc: {train_acc.val:.4f} ({train_acc.avg:.4f}) '
                        f'KD: {train_kd_loss.val:.4f} '
                        f'Policy: {train_policy_loss.val:.4f} '
                        f'Penalty: {train_length_penalties.val:.4f} '
                        f'PathLen: {train_path_lengths.val:.2f} '
                        f'Reward: {train_rewards.val:.4f} '
                        f'Temp: {temperature:.4f}')

            # Print layer skipping details
            if use_true_skipping:
                logger.info(f'Skipping details - Skipped: {len(skipped)}, Visited: {len(visited)}')
                logger.info(f'Path: {layer_path}')

        # 更新进度条（仅主进程）
        if rank == 0:
            pbar.set_postfix({
                'loss': f"{train_loss.avg:.4f}",
                'acc': f"{train_acc.avg:.4f}",
                'len': f"{train_path_lengths.avg:.2f}",
                'pen': f"{train_length_penalties.avg:.4f}"
            })

    if world_size > 1:
        dist.barrier()

    # Process any remaining accumulated gradients
    if accumulated_steps > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in student_model.parameters() if p.requires_grad],
            Config.MAX_GRAD_NORM
        )
        optimizer.step()
        optimizer.zero_grad()

    # Get layer path statistics
    path_stats = layer_tracker.get_stats()

    return {
        'loss': train_loss.avg,
        'acc': train_acc.avg,
        'path_stats': path_stats,
        'kd_loss': train_kd_loss.avg,
        'policy_loss': train_policy_loss.avg,
        'rewards': train_rewards.avg,
        'path_length': train_path_lengths.avg,
        'length_penalty': train_length_penalties.avg
    }


def _set_global_seed(seed: int):
    """设置全局随机种子以确保可复现性"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(student_model, teacher_model, rank, world_size, device, resume_checkpoint_path: Optional[str] = None):
    """训练学生模型，支持分布式训练"""
    # 设置全局随机种子（训练阶段）
    _set_global_seed(getattr(Config, 'SEED', 42))
    # Create logger (只在主进程详细记录)
    if rank == 0:
        logger = setup_logger(os.path.join(Config.OUTPUT_DIR, 'train.log'))
        logger.info(f"训练配置: {vars(Config)}")
        logger.info(f"分布式训练: {world_size} 个进程，当前进程: {rank}")
        logger.info(f"最小路径长度设置为: {Config.MIN_PATH_LENGTH}")
        logger.info(f"路径长度惩罚系数设置为: {Config.LENGTH_PENALTY_ALPHA}")
    else:
        logger = setup_logger()
        logger.setLevel('WARNING')

    # Create dataloaders with distributed support
    train_loader, val_loader = create_dataloaders(student_model, teacher_model, world_size, rank)

    # Create layer trackers
    train_layer_tracker = LayerPathTracker(student_model.num_layers)
    val_layer_tracker = LayerPathTracker(student_model.num_layers)

    # Create criterion
    criterion = TotalLoss(
        num_layers=student_model.num_layers,
        kd_weight=Config.KD_WEIGHT,
        policy_weight=Config.POLICY_WEIGHT
    )

    # Move criterion to device
    criterion.to(device)

    # Create optimizer for trainable parameters only
    optimizer = optim.AdamW(
        [p for p in student_model.parameters() if p.requires_grad],
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )

    # Create temperature scheduler
    temp_scheduler = TemperatureScheduler(
        init_temp=Config.INIT_TEMP,
        min_temp=Config.MIN_TEMP,
        decay_factor=Config.TEMP_DECAY
    )

    # Create learning rate scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=(rank == 0),  # 只在主进程显示详细信息
        min_lr=1e-6
    )

    # Resume support: load optimizer/lr scheduler/temp state and start epoch
    start_epoch = 1
    best_acc = 0.0
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        try:
            ckpt = torch.load(resume_checkpoint_path, map_location='cpu')
            if rank == 0:
                logger.info(f"从检查点恢复优化器与调度器: {resume_checkpoint_path}")
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'lr_scheduler' in ckpt:
                try:
                    lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
                except Exception:
                    # 不同版本/结构兼容失败时，忽略恢复lr调度器
                    if rank == 0:
                        logger.warning("学习率调度器状态恢复失败，使用新调度器")
            if 'temp_scheduler' in ckpt and isinstance(ckpt['temp_scheduler'], (int, float)):
                temp_scheduler.current_temp = float(ckpt['temp_scheduler'])
            if 'epoch' in ckpt:
                start_epoch = int(ckpt['epoch']) + 1
            if 'best_acc' in ckpt:
                best_acc = float(ckpt['best_acc'])
        except Exception as e:
            if rank == 0:
                logger.warning(f"恢复检查点训练状态失败: {e}，将从头训练优化器/调度器")
    # Training loop
    for epoch in range(start_epoch, Config.EPOCHS + 1):
        if rank == 0:
            logger.info(f"启动epoch {epoch}/{Config.EPOCHS}")

        # Get max skip distance for current epoch
        max_skip_distance = Config.get_max_skip_distance(epoch)
        if rank == 0:
            logger.info(f"使用最大跳过距离: {max_skip_distance}")

        # Train
        train_stats = train_epoch(
            student_model=student_model,
            teacher_model=teacher_model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            temp_scheduler=temp_scheduler,
            epoch=epoch,
            logger=logger,
            layer_tracker=train_layer_tracker,
            max_skip_distance=max_skip_distance,
            rank=rank,
            world_size=world_size,
            device=device
        )

        # Optional evaluation using unified inference evaluator
        performed_eval = False
        val_stats = None
        if getattr(Config, 'EVAL_EVERY', 0) and (epoch % Config.EVAL_EVERY == 0):
            if hasattr(val_loader.sampler, 'set_epoch'):
                val_loader.sampler.set_epoch(epoch)
            if rank == 0:
                logger.info("执行统一推理评估以替代按轮验证...")
            eval_results = eval_inference(
                model=student_model,
                val_loader=val_loader,
                device=device,
                logger=logger,
                true_skipping=True,
                max_skip_distance=max_skip_distance,
                temperature=getattr(Config, 'EVAL_TEMPERATURE', 0.1),
                rank=rank,
                world_size=world_size,
                enforce_min_path=getattr(Config, 'EVAL_ENFORCE_MIN_PATH', False)
            )
            # 将评估结果映射到训练日志需要的键
            val_stats = {
                'loss': float('nan'),  # 推理评估不计算loss
                'acc': eval_results['accuracy'] / 100.0,
                'path_stats': eval_results['path_stats'],
                'kd_loss': float('nan'),
                'path_length': eval_results['path_stats']['avg_path_length'],
                'min_path_met_percent': eval_results['min_path_length_met_percent'],
                'path_length_counts': eval_results.get('path_length_counts')
            }
            performed_eval = True

        # Step temperature scheduler
        temp_scheduler.step()

        # Step learning rate scheduler（若未评估，则退化为使用训练准确率驱动）
        if performed_eval and val_stats is not None:
            lr_scheduler.step(val_stats['acc'])
        else:
            lr_scheduler.step(train_stats['acc'])

        # 只在主进程记录和保存结果
        if rank == 0:
            # Log results
            if performed_eval and val_stats is not None:
                logger.info(f"Epoch {epoch} 结果 - "
                            f"训练损失: {train_stats['loss']:.4f}, "
                            f"训练准确率: {train_stats['acc']:.4f}, "
                            f"训练路径长度: {train_stats['path_length']:.2f}, "
                            f"训练长度惩罚: {train_stats.get('length_penalty', 0):.4f}, "
                            f"评估准确率: {val_stats['acc']:.4f}, "
                            f"评估路径长度: {val_stats['path_length']:.2f}, "
                            f"评估最小路径满足: {val_stats['min_path_met_percent']:.2f}%")
            else:
                logger.info(f"Epoch {epoch} 结果 - "
                            f"训练损失: {train_stats['loss']:.4f}, "
                            f"训练准确率: {train_stats['acc']:.4f}, "
                            f"训练路径长度: {train_stats['path_length']:.2f}, "
                            f"训练长度惩罚: {train_stats.get('length_penalty', 0):.4f}, "
                            f"（本轮未执行评估）")

            if performed_eval and val_stats is not None:
                logger.info(f"训练路径 - "
                            f"长度: {train_stats['path_stats']['avg_path_length']:.2f}, "
                            f"计算量: {train_stats['path_stats']['avg_computation'] * 100:.2f}%, "
                            f"评估路径 - "
                            f"长度: {val_stats['path_stats']['avg_path_length']:.2f}, "
                            f"计算量: {val_stats['path_stats']['avg_computation'] * 100:.2f}%")

            # Check if best model
            if performed_eval and val_stats is not None:
                is_best = val_stats['acc'] > best_acc
                best_acc = max(val_stats['acc'], best_acc)
            else:
                is_best = False

            # 使用FSDP状态字典
            from utils import save_fsdp_checkpoint
            save_fsdp_checkpoint(
                model=student_model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                temp_scheduler=temp_scheduler.current_temp,
                best_acc=best_acc,
                train_stats=train_stats,
                val_stats=val_stats if val_stats is not None else {},
                config=get_serializable_config(Config),
                is_best=is_best,
                filename=os.path.join(Config.OUTPUT_DIR, f'checkpoint_epoch{epoch}.pth.tar'),
                best_filename=os.path.join(Config.OUTPUT_DIR, 'model_best.pth.tar'),
                use_fsdp=student_model.use_fsdp
            )

            # Plot layer visit rate
            if performed_eval and val_stats is not None:
                plot_path = os.path.join(Config.OUTPUT_DIR, f'epoch{epoch}_layer_visit_rate.png')
                plot_layer_visit_rate(val_stats['path_stats']['layer_visit_rate'], plot_path)

            # Plot path length distribution
            if performed_eval and val_stats is not None and val_stats.get('path_length_counts') is not None:
                plot_path = os.path.join(Config.OUTPUT_DIR, f'epoch{epoch}_path_length_dist.png')
                plot_path_length_distribution(val_stats['path_length_counts'], student_model.num_layers + 1, plot_path)

    if rank == 0:
        logger.info(f"训练完成。最佳验证准确率: {best_acc:.4f}")

    # 确保所有进程同步完成
    if world_size > 1:
        dist.barrier()

    return best_acc