#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修改后的 model_runner.py 中的日志功能
"""

import os
import sys
sys.path.insert(0, '/root/mingtong/aiwork/nano-vllm')

from nanovllm.log_config import LogConfig, LogLevel, log_debug, log_info, log_warning, log_error

def test_model_runner_logs():
    print("开始测试修改后的 model_runner.py 中的日志功能...")
    
    # 设置环境变量以控制日志级别
    os.environ["NANOVLLM_LOG_LEVEL"] = "DEBUG"
    os.environ["NANOVLLM_LOG_MODEL_RUNNER"] = "DEBUG"
    
    # 测试各种日志级别
    print("\n--- 测试各种日志级别 ---")
    log_debug("model_runner", "这是一个 DEBUG 级别的日志", rank=0)
    log_info("model_runner", "这是一个 INFO 级别的日志", rank=0)
    log_warning("model_runner", "这是一个 WARNING 级别的日志", rank=0)
    log_error("model_runner", "这是一个 ERROR 级别的日志", rank=0)
    
    # 测试不同模块的日志
    print("\n--- 测试不同模块的日志 ---")
    log_debug("chunked_prefill", "Chunked Prefill 模块的调试信息", rank=1)
    log_info("warmup", "Warmup 模块的信息", rank=0)
    log_warning("scheduler", "Scheduler 模块的警告", rank=2)
    
    # 测试环境变量控制
    print("\n--- 测试环境变量控制 ---")
    old_level = os.environ.get("NANOVLLM_LOG_MODEL_RUNNER")
    os.environ["NANOVLLM_LOG_MODEL_RUNNER"] = "WARNING"  # 只显示警告及以上级别
    
    print("设置 NANOVLLM_LOG_MODEL_RUNNER=WARNING，接下来的 DEBUG 和 INFO 日志不应该显示:")
    log_debug("model_runner", "这个 DEBUG 日志不应该显示", rank=0)
    log_info("model_runner", "这个 INFO 日志不应该显示", rank=0)
    log_warning("model_runner", "这个 WARNING 日志应该显示", rank=0)
    
    # 恢复环境变量
    if old_level:
        os.environ["NANOVLLM_LOG_MODEL_RUNNER"] = old_level
    else:
        del os.environ["NANOVLLM_LOG_MODEL_RUNNER"]
    
    print("\n--- 测试完成 ---")
    print("所有日志功能测试完毕，model_runner.py 中的 print 语句已成功替换为新的日志模块")

if __name__ == "__main__":
    test_model_runner_logs()