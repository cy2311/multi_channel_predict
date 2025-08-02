"""日志工具模块

该模块提供了日志配置和管理功能，包括：
- 日志配置
- 多种日志处理器
- 日志格式化
- 性能日志
- 错误追踪
"""

import logging
import logging.handlers
import sys
import os
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime
import json


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
        'RESET': '\033[0m'        # 重置
    }
    
    def format(self, record):
        # 添加颜色
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        
        # 格式化消息
        formatted = super().format(record)
        
        return formatted


class JSONFormatter(logging.Formatter):
    """JSON格式化器"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)


class PerformanceLogger:
    """性能日志记录器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_times = {}
    
    def start_timer(self, name: str):
        """开始计时"""
        self.start_times[name] = time.time()
        self.logger.debug(f"开始计时: {name}")
    
    def end_timer(self, name: str) -> float:
        """结束计时并记录"""
        if name not in self.start_times:
            self.logger.warning(f"未找到计时器: {name}")
            return 0.0
        
        elapsed = time.time() - self.start_times[name]
        del self.start_times[name]
        
        self.logger.info(f"性能统计 - {name}: {elapsed:.4f}秒")
        return elapsed
    
    def log_memory_usage(self, name: str = "当前"):
        """记录内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.logger.info(
                f"内存使用 - {name}: "
                f"RSS={memory_info.rss / 1024 / 1024:.2f}MB, "
                f"VMS={memory_info.vms / 1024 / 1024:.2f}MB"
            )
        except ImportError:
            self.logger.warning("psutil未安装，无法记录内存使用")
    
    def log_gpu_usage(self, name: str = "当前"):
        """记录GPU使用情况"""
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                    cached = torch.cuda.memory_reserved(i) / 1024 / 1024
                    
                    self.logger.info(
                        f"GPU使用 - {name} (GPU {i}): "
                        f"已分配={allocated:.2f}MB, "
                        f"已缓存={cached:.2f}MB"
                    )
        except ImportError:
            self.logger.warning("PyTorch未安装，无法记录GPU使用")


class ContextLogger:
    """上下文日志记录器"""
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        self.logger = logger
        self.context = context
    
    def _add_context(self, record):
        """添加上下文信息"""
        if not hasattr(record, 'extra_fields'):
            record.extra_fields = {}
        record.extra_fields.update(self.context)
        return record
    
    def debug(self, msg, *args, **kwargs):
        record = self.logger.makeRecord(
            self.logger.name, logging.DEBUG, "", 0, msg, args, None
        )
        self._add_context(record)
        self.logger.handle(record)
    
    def info(self, msg, *args, **kwargs):
        record = self.logger.makeRecord(
            self.logger.name, logging.INFO, "", 0, msg, args, None
        )
        self._add_context(record)
        self.logger.handle(record)
    
    def warning(self, msg, *args, **kwargs):
        record = self.logger.makeRecord(
            self.logger.name, logging.WARNING, "", 0, msg, args, None
        )
        self._add_context(record)
        self.logger.handle(record)
    
    def error(self, msg, *args, **kwargs):
        record = self.logger.makeRecord(
            self.logger.name, logging.ERROR, "", 0, msg, args, None
        )
        self._add_context(record)
        self.logger.handle(record)
    
    def critical(self, msg, *args, **kwargs):
        record = self.logger.makeRecord(
            self.logger.name, logging.CRITICAL, "", 0, msg, args, None
        )
        self._add_context(record)
        self.logger.handle(record)


def setup_logging(log_level: Union[str, int] = logging.INFO,
                 log_file: Optional[str] = None,
                 log_format: str = "standard",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 enable_console: bool = True,
                 enable_colors: bool = True) -> logging.Logger:
    """设置日志配置
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
        log_format: 日志格式 (standard, detailed, json)
        max_file_size: 最大文件大小
        backup_count: 备份文件数量
        enable_console: 是否启用控制台输出
        enable_colors: 是否启用颜色
        
    Returns:
        logging.Logger: 配置好的日志器
    """
    # 创建根日志器
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 清除现有处理器
    logger.handlers.clear()
    
    # 设置格式
    if log_format == "standard":
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    elif log_format == "detailed":
        format_str = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(funcName)s - %(message)s"
        )
    elif log_format == "json":
        format_str = None  # JSON格式化器不需要格式字符串
    else:
        format_str = log_format
    
    # 控制台处理器
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        if log_format == "json":
            console_formatter = JSONFormatter()
        elif enable_colors and sys.stdout.isatty():
            console_formatter = ColoredFormatter(format_str)
        else:
            console_formatter = logging.Formatter(format_str)
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        # 确保日志目录存在
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用RotatingFileHandler进行日志轮转
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        
        if log_format == "json":
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(format_str)
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str, 
              log_level: Optional[Union[str, int]] = None,
              context: Optional[Dict[str, Any]] = None) -> Union[logging.Logger, ContextLogger]:
    """获取日志器
    
    Args:
        name: 日志器名称
        log_level: 日志级别
        context: 上下文信息
        
    Returns:
        Union[logging.Logger, ContextLogger]: 日志器
    """
    logger = logging.getLogger(name)
    
    if log_level is not None:
        logger.setLevel(log_level)
    
    if context:
        return ContextLogger(logger, context)
    
    return logger


def log_exception(logger: logging.Logger, 
                 exception: Exception,
                 message: str = "发生异常",
                 include_traceback: bool = True):
    """记录异常
    
    Args:
        logger: 日志器
        exception: 异常对象
        message: 附加消息
        include_traceback: 是否包含堆栈跟踪
    """
    error_msg = f"{message}: {type(exception).__name__}: {str(exception)}"
    
    if include_traceback:
        logger.error(error_msg, exc_info=True)
    else:
        logger.error(error_msg)


def log_function_call(logger: logging.Logger, 
                     func_name: str,
                     args: tuple = (),
                     kwargs: dict = None,
                     log_level: int = logging.DEBUG):
    """记录函数调用
    
    Args:
        logger: 日志器
        func_name: 函数名
        args: 位置参数
        kwargs: 关键字参数
        log_level: 日志级别
    """
    if kwargs is None:
        kwargs = {}
    
    args_str = ", ".join([repr(arg) for arg in args])
    kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
    
    all_args = []
    if args_str:
        all_args.append(args_str)
    if kwargs_str:
        all_args.append(kwargs_str)
    
    params_str = ", ".join(all_args)
    
    logger.log(log_level, f"调用函数: {func_name}({params_str})")


def create_performance_logger(name: str) -> PerformanceLogger:
    """创建性能日志器
    
    Args:
        name: 日志器名称
        
    Returns:
        PerformanceLogger: 性能日志器
    """
    logger = get_logger(name)
    return PerformanceLogger(logger)


class LoggingContext:
    """日志上下文管理器"""
    
    def __init__(self, logger: logging.Logger, 
                 message: str,
                 log_level: int = logging.INFO,
                 log_args: bool = False,
                 log_result: bool = False):
        self.logger = logger
        self.message = message
        self.log_level = log_level
        self.log_args = log_args
        self.log_result = log_result
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(self.log_level, f"开始: {self.message}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.log(
                self.log_level, 
                f"完成: {self.message} (耗时: {elapsed:.4f}秒)"
            )
        else:
            self.logger.error(
                f"失败: {self.message} (耗时: {elapsed:.4f}秒) - "
                f"{exc_type.__name__}: {exc_val}"
            )
        
        return False  # 不抑制异常


def logged_function(logger: Optional[logging.Logger] = None,
                   log_level: int = logging.DEBUG,
                   log_args: bool = False,
                   log_result: bool = False,
                   log_time: bool = True):
    """函数日志装饰器
    
    Args:
        logger: 日志器
        log_level: 日志级别
        log_args: 是否记录参数
        log_result: 是否记录结果
        log_time: 是否记录执行时间
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_logger = logger or get_logger(func.__module__)
            
            # 记录函数调用
            if log_args:
                log_function_call(func_logger, func.__name__, args, kwargs, log_level)
            else:
                func_logger.log(log_level, f"调用函数: {func.__name__}")
            
            start_time = time.time() if log_time else None
            
            try:
                result = func(*args, **kwargs)
                
                # 记录执行时间
                if log_time:
                    elapsed = time.time() - start_time
                    func_logger.log(
                        log_level, 
                        f"函数 {func.__name__} 执行完成 (耗时: {elapsed:.4f}秒)"
                    )
                
                # 记录结果
                if log_result:
                    func_logger.log(log_level, f"函数 {func.__name__} 返回: {repr(result)}")
                
                return result
                
            except Exception as e:
                if log_time:
                    elapsed = time.time() - start_time
                    func_logger.error(
                        f"函数 {func.__name__} 执行失败 (耗时: {elapsed:.4f}秒): {e}"
                    )
                else:
                    func_logger.error(f"函数 {func.__name__} 执行失败: {e}")
                raise
        
        return wrapper
    return decorator


def setup_file_logging(log_dir: str,
                      log_name: str = "decode",
                      log_level: Union[str, int] = logging.INFO,
                      max_file_size: int = 10 * 1024 * 1024,
                      backup_count: int = 5) -> str:
    """设置文件日志
    
    Args:
        log_dir: 日志目录
        log_name: 日志文件名前缀
        log_level: 日志级别
        max_file_size: 最大文件大小
        backup_count: 备份文件数量
        
    Returns:
        str: 日志文件路径
    """
    # 创建日志目录
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{log_name}_{timestamp}.log"
    
    # 设置日志
    setup_logging(
        log_level=log_level,
        log_file=str(log_file),
        log_format="detailed",
        max_file_size=max_file_size,
        backup_count=backup_count
    )
    
    return str(log_file)


def disable_logging(logger_names: Optional[list] = None):
    """禁用日志
    
    Args:
        logger_names: 要禁用的日志器名称列表，None表示禁用所有
    """
    if logger_names is None:
        logging.disable(logging.CRITICAL)
    else:
        for name in logger_names:
            logger = logging.getLogger(name)
            logger.disabled = True


def enable_logging(logger_names: Optional[list] = None):
    """启用日志
    
    Args:
        logger_names: 要启用的日志器名称列表，None表示启用所有
    """
    if logger_names is None:
        logging.disable(logging.NOTSET)
    else:
        for name in logger_names:
            logger = logging.getLogger(name)
            logger.disabled = False