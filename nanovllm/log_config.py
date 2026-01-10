"""
日志配置模块

控制各个模块的日志输出级别，方便调试和生产环境切换。
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LogLevel(Enum):
    """日志级别"""

    NONE = 0  # 不输出任何日志
    ERROR = 1  # 只输出错误
    WARNING = 2  # 输出警告和错误
    INFO = 3  # 输出信息、警告和错误
    DEBUG = 4  # 输出所有日志，包括调试信息


@dataclass
class LogConfig:
    """
    全局日志配置

    可以通过环境变量或代码配置来控制各个模块的日志级别。

    环境变量：
        NANOVLLM_LOG_LEVEL: 全局日志级别 (NONE/ERROR/WARNING/INFO/DEBUG)
        NANOVLLM_LOG_CHUNKED_PREFILL: Chunked Prefill 模块日志级别
        NANOVLLM_LOG_SCHEDULER: Scheduler 模块日志级别
        NANOVLLM_LOG_MODEL_RUNNER: ModelRunner 模块日志级别
        NANOVLLM_LOG_BLOCK_MANAGER: BlockManager 模块日志级别
        NANOVLLM_LOG_WARMUP: Warmup 模块日志级别

    使用示例：
        # 方式1：通过环境变量（推荐生产环境）
        export NANOVLLM_LOG_LEVEL=ERROR  # 生产环境只输出错误
        export NANOVLLM_LOG_CHUNKED_PREFILL=DEBUG  # 调试 Chunked Prefill

        # 方式2：通过代码配置（推荐开发环境）
        log_config = LogConfig(
            global_level=LogLevel.ERROR,
            chunked_prefill=LogLevel.DEBUG
        )
        llm = LLM(model_path, log_config=log_config)
    """

    # 全局日志级别（默认从环境变量读取，否则为 WARNING）
    global_level: LogLevel = None

    # 各模块的日志级别（None 表示使用 global_level）
    chunked_prefill: Optional[LogLevel] = None
    scheduler: Optional[LogLevel] = None
    model_runner: Optional[LogLevel] = None
    block_manager: Optional[LogLevel] = None
    warmup: Optional[LogLevel] = None
    attention: Optional[LogLevel] = None

    def __post_init__(self):
        # 如果没有设置 global_level，从环境变量读取
        if self.global_level is None:
            env_level = os.getenv("NANOVLLM_LOG_LEVEL", "WARNING").upper()
            self.global_level = self._parse_level(env_level, LogLevel.WARNING)

        # 从环境变量读取各模块的日志级别
        self._load_from_env()

    def _parse_level(self, level_str: str, default: LogLevel) -> LogLevel:
        """解析日志级别字符串"""
        try:
            return LogLevel[level_str.upper()]
        except (KeyError, AttributeError):
            return default

    def _load_from_env(self):
        """从环境变量加载配置"""
        env_mapping = {
            "chunked_prefill": "NANOVLLM_LOG_CHUNKED_PREFILL",
            "scheduler": "NANOVLLM_LOG_SCHEDULER",
            "model_runner": "NANOVLLM_LOG_MODEL_RUNNER",
            "block_manager": "NANOVLLM_LOG_BLOCK_MANAGER",
            "warmup": "NANOVLLM_LOG_WARMUP",
            "attention": "NANOVLLM_LOG_ATTENTION",
        }

        for attr, env_var in env_mapping.items():
            # 只有当属性为 None 时才从环境变量读取
            if getattr(self, attr) is None:
                env_value = os.getenv(env_var)
                if env_value:
                    setattr(self, attr, self._parse_level(env_value, None))

    def get_level(self, module: str) -> LogLevel:
        """
        获取指定模块的日志级别

        Args:
            module: 模块名称 (chunked_prefill, scheduler, model_runner, 等)

        Returns:
            该模块的日志级别
        """
        module_level = getattr(self, module, None)
        return module_level if module_level is not None else self.global_level

    def should_log(self, module: str, level: LogLevel) -> bool:
        """
        判断是否应该输出日志

        Args:
            module: 模块名称
            level: 要输出的日志级别

        Returns:
            True 如果应该输出，False 否则
        """
        module_level = self.get_level(module)
        return level.value <= module_level.value

    def log(self, module: str, level: LogLevel, message: str, **kwargs):
        """
        输出日志

        Args:
            module: 模块名称
            level: 日志级别
            message: 日志消息
            **kwargs: 额外参数（如 flush, rank 等）
        """
        if not self.should_log(module, level):
            return

        # 创建 logger（如果不存在）
        logger_name = f"nanovllm.{module}"
        logger = logging.getLogger(logger_name)

        # 构造日志消息，包含 rank 信息（如果提供）
        rank = kwargs.get("rank")
        if rank is not None:
            formatted_message = f"[rank{rank}][{level.name}][{module}] {message}"
        else:
            formatted_message = f"[{level.name}][{module}] {message}"

        # 根据日志级别调用相应的 logging 方法
        if level == LogLevel.DEBUG:
            logger.debug(formatted_message)
        elif level == LogLevel.INFO:
            logger.info(formatted_message)
        elif level == LogLevel.WARNING:
            logger.warning(formatted_message)
        elif level == LogLevel.ERROR:
            logger.error(formatted_message)

    def debug(self, module: str, message: str, **kwargs):
        """输出 DEBUG 级别日志"""
        self.log(module, LogLevel.DEBUG, message, **kwargs)

    def info(self, module: str, message: str, **kwargs):
        """输出 INFO 级别日志"""
        self.log(module, LogLevel.INFO, message, **kwargs)

    def warning(self, module: str, message: str, **kwargs):
        """输出 WARNING 级别日志"""
        self.log(module, LogLevel.WARNING, message, **kwargs)

    def error(self, module: str, message: str, **kwargs):
        """输出 ERROR 级别日志"""
        self.log(module, LogLevel.ERROR, message, **kwargs)


# 全局单例实例
_global_log_config: Optional[LogConfig] = None


def configure_logging():
    """配置根日志处理器，确保日志能正确输出"""
    root_logger = logging.getLogger("nanovllm")
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.DEBUG)


def get_log_config() -> LogConfig:
    """获取全局日志配置实例"""
    global _global_log_config
    if _global_log_config is None:
        _global_log_config = LogConfig()
        configure_logging()  # 初始化日志配置
    return _global_log_config


def set_log_config(config: LogConfig):
    """设置全局日志配置实例"""
    global _global_log_config
    _global_log_config = config
    configure_logging()  # 确保日志已配置


# 便捷函数
def should_log(module: str, level: LogLevel) -> bool:
    """判断是否应该输出日志"""
    return get_log_config().should_log(module, level)


def log_debug(module: str, message: str, **kwargs):
    """输出 DEBUG 日志"""
    get_log_config().debug(module, message, **kwargs)


def log_info(module: str, message: str, **kwargs):
    """输出 INFO 日志"""
    get_log_config().info(module, message, **kwargs)


def log_warning(module: str, message: str, **kwargs):
    """输出 WARNING 日志"""
    get_log_config().warning(module, message, **kwargs)


def log_error(module: str, message: str, **kwargs):
    """输出 ERROR 日志"""
    get_log_config().error(module, message, **kwargs)
