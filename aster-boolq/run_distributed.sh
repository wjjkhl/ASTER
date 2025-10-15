#!/bin/bash
# 使用torchrun启动分布式训练

# 设置环境变量
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2  # 固定使用GPU 0,1,2
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

# 使用torchrun启动
torchrun \
    --nproc_per_node=3 \
    --master_port=29501 \
    main.py --mode train "$@"