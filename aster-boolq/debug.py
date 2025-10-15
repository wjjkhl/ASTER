#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import shutil
import time
import sys


def run_distributed_training():
    """运行分布式训练，使用优化的NCCL参数"""

    # 设置基本环境变量
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    # 设置优化的NCCL环境变量
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "0"
    os.environ["NCCL_BUFFSIZE"] = "2097152"  # 减小buffer大小
    os.environ["NCCL_NTHREADS"] = "4"
    os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
    os.environ["NCCL_SOCKET_NTHREADS"] = "4"

    # 设置较短的超时时间，避免卡死
    os.environ["NCCL_TIMEOUT"] = "30"  # 30秒超时

    print("已设置环境变量:")
    for var in ["OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "TORCH_NCCL_BLOCKING_WAIT",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING", "NCCL_DEBUG", "NCCL_SOCKET_IFNAME",
                "NCCL_IB_DISABLE", "NCCL_P2P_DISABLE", "NCCL_BUFFSIZE", "NCCL_NTHREADS",
                "NCCL_NSOCKS_PERTHREAD", "NCCL_SOCKET_NTHREADS", "NCCL_TIMEOUT"]:
        print(f"{var} = {os.environ.get(var)}")

    # 清理NCCL共享内存文件
    nccl_files = "/dev/shm/nccl-*"
    print(f"清理NCCL文件: {nccl_files}")
    os.system(f"rm -rf {nccl_files}")

    # 使用新创建的debug_train.py替代main.py
    torchrun_cmd = [
        "torchrun",
        "--nproc_per_node=4",
        "--master_port=29501",
        "--nnodes=1",
        "--node_rank=0",
        "--rdzv_id=123456",
        "--rdzv_backend=c10d",
        "/cephfs/shared/wjj/LLAMAtest/debug_train.py"  # 使用新的训练脚本
    ]

    # 打印将要执行的命令
    print("开始执行命令: " + " ".join(torchrun_cmd))

    # 执行torchrun命令
    try:
        start_time = time.time()
        process = subprocess.Popen(
            torchrun_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # 实时打印输出
        for line in process.stdout:
            print(line.strip())

        # 等待进程结束
        return_code = process.wait()
        elapsed_time = time.time() - start_time

        if return_code == 0:
            print(f"分布式训练成功完成! 用时: {elapsed_time:.2f}秒")
            return True
        else:
            print(f"分布式训练失败，返回码: {return_code}, 用时: {elapsed_time:.2f}秒")
            return False

    except KeyboardInterrupt:
        print("训练被用户中断")
        process.terminate()
        process.wait(timeout=10)
        if process.poll() is None:
            process.kill()
        return False
    except Exception as e:
        print(f"执行训练时发生错误: {str(e)}")
        if 'process' in locals():
            process.terminate()
            if process.poll() is None:
                process.kill()
        return False


if __name__ == "__main__":
    # 确保目录存在
    from config import Config

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    print(f"==== 分布式训练开始 - {time.strftime('%Y-%m-%d %H:%M:%S')} ====")
    success = run_distributed_training()
    print(f"==== 分布式训练结束 - {time.strftime('%Y-%m-%d %H:%M:%S')} ====")

    # 返回适当的退出代码
    sys.exit(0 if success else 1)