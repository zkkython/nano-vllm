#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修改后的 log_config.py 文件，验证其使用 logging 模块的功能
"""
from nanovllm.log_config import (
    LogLevel,
    get_log_config,
    log_debug,
    log_info,
    log_warning,
    log_error,
    should_log,
)


def test_log_config():
    print("开始测试修改后的 log_config.py 文件...")

    # 测试1: 基本日志配置
    print("\n--- 测试1: 基本日志配置 ---")
    log_config = get_log_config()
    print(f"全局日志级别: {log_config.global_level}")

    # 测试2: 不同级别的日志输出
    print("\n--- 测试2: 不同级别的日志输出 ---")
    log_config.debug("test_module", "这是一个 DEBUG 级别的日志")
    log_config.info("test_module", "这是一个 INFO 级别的日志")
    log_config.warning("test_module", "这是一个 WARNING 级别的日志")
    log_config.error("test_module", "这是一个 ERROR 级别的日志")

    # 测试3: 使用便捷函数
    print("\n--- 测试3: 使用便捷函数 ---")
    log_debug("test_module", "通过便捷函数输出 DEBUG 日志")
    log_info("test_module", "通过便捷函数输出 INFO 日志")
    log_warning("test_module", "通过便捷函数输出 WARNING 日志")
    log_error("test_module", "通过便捷函数输出 ERROR 日志")

    # 测试4: 带 rank 信息的日志
    print("\n--- 测试4: 带 rank 信息的日志 ---")
    log_config.info("test_module", "带 rank 信息的日志", rank=0)


    # 测试6: should_log 函数
    print("\n--- 测试6: should_log 函数 ---")
    print(f"是否应该输出 debug 日志: {should_log('test_module', LogLevel.DEBUG)}")
    print(f"是否应该输出 info 日志: {should_log('test_module', LogLevel.INFO)}")
    print(f"是否应该输出 warning 日志: {should_log('test_module', LogLevel.WARNING)}")
    print(f"是否应该输出 error 日志: {should_log('test_module', LogLevel.ERROR)}")

    print("\n--- 测试完成 ---")


if __name__ == "__main__":
    test_log_config()
