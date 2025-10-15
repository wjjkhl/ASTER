#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import time
import logging
import argparse
from datetime import datetime


# 设置日志记录
def setup_logger(log_dir="./logs"):
    """设置日志记录器"""
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志文件名，包含时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"distributed_training_{timestamp}.log")

    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def check_environment():
    """检查环境设置"""
    # 检查CUDA可用性
    try:
        gpu_count = int(subprocess.check_output(
            "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l",
            shell=True
        ).decode().strip())

        return {
            "gpu_count": gpu_count,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "未设置")
        }
    except Exception as e:
        return {
            "error": f"检查GPU失败: {str(e)}",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "未设置")
        }


def run_training(script_path="./run_distributed.sh", timeout=None, args=None):
    """运行分布式训练脚本"""
    logger = logging.getLogger(__name__)

    # 确保脚本存在且可执行
    if not os.path.exists(script_path):
        logger.error(f"训练脚本 {script_path} 不存在!")
        return False

    # 添加执行权限
    os.chmod(script_path, 0o755)

    # 构建命令
    cmd = [script_path]
    if args:
        cmd.extend(args)

    logger.info(f"开始执行训练脚本: {' '.join(cmd)}")
    logger.info(f"环境信息: {check_environment()}")

    # 启动进程并实时输出日志
    start_time = time.time()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # 实时读取并记录输出
        for line in process.stdout:
            line = line.strip()
            logger.info(f"训练输出: {line}")

            # 检查是否有错误关键词
            error_keywords = ["error", "exception", "fail", "错误", "异常", "失败"]
            if any(keyword in line.lower() for keyword in error_keywords):
                logger.warning(f"检测到可能的错误: {line}")

        # 等待进程完成
        return_code = process.wait(timeout=timeout)

        if return_code == 0:
            elapsed_time = time.time() - start_time
            logger.info(f"训练成功完成! 用时: {elapsed_time:.2f}秒")
            return True
        else:
            logger.error(f"训练失败，返回码: {return_code}")
            return False

    except subprocess.TimeoutExpired:
        process.kill()
        logger.error(f"训练超时，已强制终止")
        return False
    except Exception as e:
        logger.error(f"执行训练时出错: {str(e)}")
        if 'process' in locals() and process.poll() is None:
            process.kill()
        return False


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="分布式训练启动器")
    parser.add_argument("--script", type=str, default="./run_distributed.sh",
                        help="要执行的分布式训练脚本路径")
    parser.add_argument("--timeout", type=int, default=None,
                        help="训练超时时间（秒），默认无限制")
    parser.add_argument("--visible-devices", type=str, default=None,
                        help="设置CUDA_VISIBLE_DEVICES环境变量")
    parser.add_argument("extra_args", nargs="*",
                        help="传递给训练脚本的额外参数")

    args = parser.parse_args()

    # 设置日志记录器
    logger = setup_logger()
    logger.info("分布式训练启动器启动")

    # 设置可见GPU
    if args.visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_devices
        logger.info(f"设置CUDA_VISIBLE_DEVICES={args.visible_devices}")

    # 运行训练
    success = run_training(
        script_path=args.script,
        timeout=args.timeout,
        args=args.extra_args
    )

    if success:
        logger.info("训练流程成功完成")
    else:
        logger.error("训练流程失败")
        exit(1)


if __name__ == "__main__":
    main()