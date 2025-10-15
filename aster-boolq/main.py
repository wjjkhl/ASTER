import datetime
import os
import random
import numpy as np
import torch
import argparse
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload, BackwardPrefetch, ShardingStrategy
)
from torch.nn.parallel import DistributedDataParallel as DDP
# 修改models.py中的导入部分
import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload, BackwardPrefetch, ShardingStrategy
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy  # 使用transformer_auto_wrap_policy替代default_auto_wrap_policy
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from models import StudentLLAMA, TeacherLLAMA
from train import train
from evaluate import evaluate
from dataset import create_dataloaders
from utils import setup_logger, synchronize_devices
from config import Config


# 在main.py中找到init_distributed函数，修改如下(约在第30行附近):
def init_distributed():
    """初始化分布式训练环境"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("未检测到分布式环境变量，将使用单GPU模式")
        rank = 0
        world_size = 1
        local_rank = 0

    # 初始化进程组（如果多于1个进程）
    if world_size > 1:
        # 设置NCCL超时时间(默认30分钟可能太长)和调试信息
        os.environ["NCCL_BLOCKING_WAIT"] = "1"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["NCCL_DEBUG"] = "INFO"  # 获取更多诊断信息
        # 初始化进程组
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(minutes=5),  # 缩短超时时间
            init_method="env://"
        )
        torch.cuda.set_device(local_rank)
        print(f"进程 {rank}/{world_size} 使用 GPU: {local_rank}")

    return rank, world_size, local_rank


def main(logger=None, parser=None):
    """主函数用于训练和评估动态层跳跃"""
    # 初始化分布式环境
    rank, world_size, local_rank = init_distributed()

    # 全局随机种子（训练/评估统一可复现）
    try:
        seed = getattr(Config, 'SEED', 42)
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="LLAMA 3.1的动态层跳跃")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],help='模式：训练或评估')
    parser.add_argument('--debug', action='store_true', help='在调试模式下运行（使用部分数据）')
    parser.add_argument('--checkpoint_path', type=str,
                        default='/cephfs/shared/wjj/ICLR/LLAMAtest/dynamic_llama_output/checkpoint_epoch.pth.tar',
                        help='用于评估或恢复训练的检查点路径')
    args = parser.parse_args()

    # 设置日志记录器（只在主进程记录详细日志）
    if rank == 0:
        logger = setup_logger(os.path.join(Config.OUTPUT_DIR, 'main.log'))
        logger.info(f"在{args.mode}模式下运行，使用 {world_size} 个GPU")
    else:
        logger = setup_logger()
        logger.setLevel('WARNING')

    # 设置设备为local_rank
    device = torch.device(f"cuda:{local_rank}")

    # 创建教师模型
    logger.info(f"在GPU {local_rank}上创建教师模型...")
    teacher_model = TeacherLLAMA(
        base_model_path=Config.MODEL_PATH,
        device=device
    )

    # 确保教师模型分词器有填充令牌
    if teacher_model.tokenizer.pad_token is None:
        teacher_model.tokenizer.pad_token = teacher_model.tokenizer.eos_token
        logger.info(f"在main中设置了教师模型pad_token: {teacher_model.tokenizer.pad_token}")

    # 创建学生模型
    logger.info(f"在GPU {local_rank}上创建学生模型...")
    student_model = StudentLLAMA(
        base_model_path=Config.MODEL_PATH,
        layer_embed_dim=Config.LAYER_EMBED_DIM,
        adapter_base_size=Config.ADAPTER_BASE_SIZE,
        device=device,
        use_fsdp=False  # 关闭FSDP，改用DDP同步小模块梯度
    )

    # 在多卡时使用DDP包裹学生模型（基座被冻结，不增加通信；find_unused_parameters=True 以兼容未跳层批次）
    if world_size > 1:
        student_model = DDP(student_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    raw_student = student_model.module if hasattr(student_model, 'module') else student_model

    # 确保学生模型分词器有填充令牌（注意DDP包裹）
    if raw_student.tokenizer.pad_token is None:
        raw_student.tokenizer.pad_token = raw_student.tokenizer.eos_token
        logger.info(f"在main中设置了学生模型pad_token: {raw_student.tokenizer.pad_token}")

    # 如果提供了检查点，则加载
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        logger.info(f"从{args.checkpoint_path}加载检查点")
        # 使用FSDP/简化检查点加载方法（注意DDP包裹）
        if world_size > 1:
            from utils import load_fsdp_checkpoint
            load_fsdp_checkpoint(raw_student, args.checkpoint_path, device)
            if rank == 0:
                checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
                # 记录检查点信息
                if 'epoch' in checkpoint:
                    logger.info(f"检查点来自第{checkpoint['epoch']}轮")
                if 'best_acc' in checkpoint:
                    logger.info(f"最佳准确率: {checkpoint['best_acc']:.4f}")
        else:
            checkpoint = torch.load(args.checkpoint_path, map_location=device)
            student_model.load_state_dict(checkpoint['student_model'])
            # 记录检查点信息
            if 'epoch' in checkpoint:
                logger.info(f"检查点来自第{checkpoint['epoch']}轮")
            if 'best_acc' in checkpoint:
                logger.info(f"最佳准确率: {checkpoint['best_acc']:.4f}")

    # 按指定模式运行
    if args.mode == 'train':
        logger.info("开始训练...")
        # 清空GPU缓存以获取更多可用内存
        torch.cuda.empty_cache()
        # 开始训练
        # 将checkpoint路径传递给train以支持断点续训
        resume_path = args.checkpoint_path if args.checkpoint_path and os.path.exists(args.checkpoint_path) else None
        best_acc = train(raw_student, teacher_model, rank, world_size, device, resume_checkpoint_path=resume_path)
        if rank == 0:
            logger.info(f"训练完成。最佳准确率: {best_acc:.4f}")

    elif args.mode == 'eval':
        logger.info("开始评估...")
        # 清空GPU缓存以获取更多可用内存
        torch.cuda.empty_cache()
        # 创建数据加载器（注意DDP包裹）
        _, val_loader = create_dataloaders(raw_student, teacher_model, world_size, rank)
        # 评估
        # 若提供检查点且存在，从检查点epoch推导max_skip_distance以保持一致
        derived_max_skip = Config.get_max_skip_distance(Config.EPOCHS)
        if args.checkpoint_path and os.path.exists(args.checkpoint_path):
            try:
                ckpt = torch.load(args.checkpoint_path, map_location='cpu')
                ckpt_epoch = ckpt.get('epoch', None)
            except Exception:
                ckpt_epoch = None
            derived_max_skip = Config.get_max_skip_distance(ckpt_epoch if ckpt_epoch is not None else Config.EPOCHS)
            if rank == 0:
                logger.info(f"Eval deriving max_skip_distance from checkpoint epoch {ckpt_epoch}: {derived_max_skip}")

        results = evaluate(
            model=raw_student,
            val_loader=val_loader,
            device=device,
            logger=logger,
            true_skipping=True,
            max_skip_distance=derived_max_skip,
            temperature=0.1
        )
        if rank == 0:
            logger.info(f"评估完成。准确率: {results['accuracy']:.2f}%")

    logger.info("所有操作成功完成")

    # 清理分布式环境
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()