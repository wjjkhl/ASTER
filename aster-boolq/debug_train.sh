#!/bin/bash
# 使用部分数据进行调试训练的脚本 - 分布式版本

# 设置环境变量
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL优化环境变量 - 使用TORCH_前缀替换旧参数
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_BUFFSIZE=2097152  # 减小buffer大小
export NCCL_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=4

# 减少超时时间，避免卡死
export NCCL_TIMEOUT=30  # 30秒超时更合理

# 清理NCCL共享内存文件
rm -rf /dev/shm/nccl-*

# 启动分布式训练
torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    --nnodes=1 \
    --node_rank=0 \
    --rdzv_id=123456 \
    --rdzv_backend=c10d \
    main.py --mode train --debug