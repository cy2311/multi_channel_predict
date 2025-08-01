"""装饰器模块

该模块提供了各种实用的装饰器，包括：
- 性能监控装饰器
- 缓存装饰器
- 重试装饰器
- 类型检查装饰器
- 异常处理装饰器
"""

import time
import functools
import threading
import warnings
from typing import Any, Callable, Dict, Optional, Union, Type, List
from datetime import datetime, timedelta
import logging
import inspect
import pickle
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)


def timer(func: Callable = None, *, 
         log_level: int = logging.INFO,
         logger_name: Optional[str] = None) -> Callable:
    """计时装饰器
    
    Args:
        func: 被装饰的函数
        log_level: 日志级别
        logger_name: 日志器名称
        
    Returns:
        Callable: 装饰后的函数
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            func_logger = logging.getLogger(logger_name or f.__module__)
            
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                elapsed = time.time() - start_time
                func_logger.log(
                    log_level, 
                    f"函数 {f.__name__} 执行完成，耗时: {elapsed:.4f}秒"
                )
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                func_logger.error(
                    f"函数 {f.__name__} 执行失败，耗时: {elapsed:.4f}秒，错误: {e}"
                )
                raise
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


def retry(max_attempts: int = 3,
         delay: float = 1.0,
         backoff: float = 2.0,
         exceptions: Union[Type[Exception], tuple] = Exception,
         on_retry: Optional[Callable] = None) -> Callable:
    """重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff: 延迟倍数
        exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数
        
    Returns:
        Callable: 装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        logger.error(
                            f"函数 {func.__name__} 重试 {max_attempts} 次后仍然失败: {e}"
                        )
                        raise
                    
                    logger.warning(
                        f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}，"
                        f"{current_delay:.2f}秒后重试"
                    )
                    
                    if on_retry:
                        try:
                            on_retry(attempt + 1, e)
                        except Exception as callback_error:
                            logger.error(f"重试回调函数执行失败: {callback_error}")
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
        
        return wrapper
    return decorator


def cache(maxsize: int = 128,
         ttl: Optional[float] = None,
         typed: bool = False,
         key_func: Optional[Callable] = None) -> Callable:
    """缓存装饰器
    
    Args:
        maxsize: 最大缓存大小
        ttl: 缓存生存时间（秒）
        typed: 是否区分参数类型
        key_func: 自定义键生成函数
        
    Returns:
        Callable: 装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        cache_dict = {}
        cache_times = {}
        cache_lock = threading.RLock()
        
        def make_key(*args, **kwargs):
            if key_func:
                return key_func(*args, **kwargs)
            
            key = args
            if kwargs:
                key += tuple(sorted(kwargs.items()))
            
            if typed:
                key += tuple(type(arg) for arg in args)
                if kwargs:
                    key += tuple(type(val) for val in kwargs.values())
            
            return hash(key)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = make_key(*args, **kwargs)
            current_time = time.time()
            
            with cache_lock:
                # 检查缓存是否存在且未过期
                if key in cache_dict:
                    if ttl is None or (current_time - cache_times[key]) < ttl:
                        return cache_dict[key]
                    else:
                        # 缓存过期，删除
                        del cache_dict[key]
                        del cache_times[key]
                
                # 检查缓存大小限制
                if len(cache_dict) >= maxsize:
                    # 删除最旧的缓存项
                    oldest_key = min(cache_times.keys(), key=lambda k: cache_times[k])
                    del cache_dict[oldest_key]
                    del cache_times[oldest_key]
                
                # 计算结果并缓存
                result = func(*args, **kwargs)
                cache_dict[key] = result
                cache_times[key] = current_time
                
                return result
        
        def cache_info():
            """获取缓存信息"""
            with cache_lock:
                return {
                    'hits': len(cache_dict),
                    'maxsize': maxsize,
                    'currsize': len(cache_dict),
                    'ttl': ttl
                }
        
        def cache_clear():
            """清除缓存"""
            with cache_lock:
                cache_dict.clear()
                cache_times.clear()
        
        wrapper.cache_info = cache_info
        wrapper.cache_clear = cache_clear
        
        return wrapper
    return decorator


def validate_types(**type_hints) -> Callable:
    """类型验证装饰器
    
    Args:
        **type_hints: 参数类型提示
        
    Returns:
        Callable: 装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 获取函数签名
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # 验证类型
            for param_name, expected_type in type_hints.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not isinstance(value, expected_type):
                        raise TypeError(
                            f"参数 {param_name} 期望类型 {expected_type.__name__}，"
                            f"实际类型 {type(value).__name__}"
                        )
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def deprecated(reason: str = "",
              version: Optional[str] = None,
              alternative: Optional[str] = None) -> Callable:
    """弃用警告装饰器
    
    Args:
        reason: 弃用原因
        version: 弃用版本
        alternative: 替代方案
        
    Returns:
        Callable: 装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"函数 {func.__name__} 已弃用"
            
            if version:
                message += f" (自版本 {version})"
            
            if reason:
                message += f": {reason}"
            
            if alternative:
                message += f"。请使用 {alternative} 替代"
            
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def singleton(cls: Type) -> Type:
    """单例装饰器
    
    Args:
        cls: 要装饰的类
        
    Returns:
        Type: 装饰后的类
    """
    instances = {}
    lock = threading.Lock()
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance


def rate_limit(calls: int, period: float) -> Callable:
    """速率限制装饰器
    
    Args:
        calls: 允许的调用次数
        period: 时间周期（秒）
        
    Returns:
        Callable: 装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        call_times = []
        lock = threading.Lock()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            
            with lock:
                # 移除过期的调用记录
                call_times[:] = [t for t in call_times if current_time - t < period]
                
                # 检查是否超过限制
                if len(call_times) >= calls:
                    sleep_time = period - (current_time - call_times[0])
                    if sleep_time > 0:
                        logger.warning(
                            f"函数 {func.__name__} 触发速率限制，等待 {sleep_time:.2f}秒"
                        )
                        time.sleep(sleep_time)
                        current_time = time.time()
                        call_times[:] = [t for t in call_times if current_time - t < period]
                
                # 记录当前调用
                call_times.append(current_time)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def exception_handler(exceptions: Union[Type[Exception], tuple] = Exception,
                     default_return: Any = None,
                     log_exception: bool = True,
                     reraise: bool = False) -> Callable:
    """异常处理装饰器
    
    Args:
        exceptions: 要捕获的异常类型
        default_return: 异常时的默认返回值
        log_exception: 是否记录异常
        reraise: 是否重新抛出异常
        
    Returns:
        Callable: 装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if log_exception:
                    logger.exception(f"函数 {func.__name__} 执行时发生异常: {e}")
                
                if reraise:
                    raise
                
                return default_return
        
        return wrapper
    return decorator


def thread_safe(func: Callable) -> Callable:
    """线程安全装饰器
    
    Args:
        func: 被装饰的函数
        
    Returns:
        Callable: 装饰后的函数
    """
    lock = threading.RLock()
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)
    
    return wrapper


def disk_cache(cache_dir: str,
              expire_time: Optional[float] = None,
              max_size: Optional[int] = None) -> Callable:
    """磁盘缓存装饰器
    
    Args:
        cache_dir: 缓存目录
        expire_time: 过期时间（秒）
        max_size: 最大缓存文件数
        
    Returns:
        Callable: 装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        def make_cache_key(*args, **kwargs):
            """生成缓存键"""
            key_data = (func.__name__, args, tuple(sorted(kwargs.items())))
            key_str = pickle.dumps(key_data)
            return hashlib.md5(key_str).hexdigest()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = make_cache_key(*args, **kwargs)
            cache_file = cache_path / f"{cache_key}.pkl"
            
            # 检查缓存是否存在且未过期
            if cache_file.exists():
                if expire_time is None:
                    # 无过期时间，直接使用缓存
                    try:
                        with open(cache_file, 'rb') as f:
                            return pickle.load(f)
                    except Exception as e:
                        logger.warning(f"读取缓存文件失败: {e}")
                else:
                    # 检查过期时间
                    file_time = cache_file.stat().st_mtime
                    if time.time() - file_time < expire_time:
                        try:
                            with open(cache_file, 'rb') as f:
                                return pickle.load(f)
                        except Exception as e:
                            logger.warning(f"读取缓存文件失败: {e}")
            
            # 计算结果
            result = func(*args, **kwargs)
            
            # 保存到缓存
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                
                # 检查缓存文件数量限制
                if max_size is not None:
                    cache_files = list(cache_path.glob("*.pkl"))
                    if len(cache_files) > max_size:
                        # 删除最旧的文件
                        cache_files.sort(key=lambda f: f.stat().st_mtime)
                        for old_file in cache_files[:-max_size]:
                            old_file.unlink()
                
            except Exception as e:
                logger.warning(f"保存缓存文件失败: {e}")
            
            return result
        
        def clear_cache():
            """清除缓存"""
            try:
                for cache_file in cache_path.glob("*.pkl"):
                    cache_file.unlink()
                logger.info(f"已清除 {func.__name__} 的磁盘缓存")
            except Exception as e:
                logger.error(f"清除缓存失败: {e}")
        
        wrapper.clear_cache = clear_cache
        
        return wrapper
    return decorator


def profile(sort_by: str = 'cumulative',
           lines_to_print: int = 10,
           strip_dirs: bool = True) -> Callable:
    """性能分析装饰器
    
    Args:
        sort_by: 排序方式
        lines_to_print: 打印行数
        strip_dirs: 是否去除目录路径
        
    Returns:
        Callable: 装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                import cProfile
                import pstats
                import io
                
                profiler = cProfile.Profile()
                profiler.enable()
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    profiler.disable()
                
                # 生成报告
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s)
                
                if strip_dirs:
                    ps.strip_dirs()
                
                ps.sort_stats(sort_by)
                ps.print_stats(lines_to_print)
                
                logger.info(f"函数 {func.__name__} 性能分析报告:\n{s.getvalue()}")
                
                return result
                
            except ImportError:
                logger.warning("cProfile不可用，跳过性能分析")
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def conditional(condition: Union[bool, Callable]) -> Callable:
    """条件执行装饰器
    
    Args:
        condition: 执行条件（布尔值或返回布尔值的函数）
        
    Returns:
        Callable: 装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 检查条件
            if callable(condition):
                should_execute = condition(*args, **kwargs)
            else:
                should_execute = condition
            
            if should_execute:
                return func(*args, **kwargs)
            else:
                logger.debug(f"函数 {func.__name__} 因条件不满足而跳过执行")
                return None
        
        return wrapper
    return decorator


def async_to_sync(func: Callable) -> Callable:
    """异步转同步装饰器
    
    Args:
        func: 异步函数
        
    Returns:
        Callable: 同步函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import asyncio
            
            # 尝试获取当前事件循环
            try:
                loop = asyncio.get_running_loop()
                # 如果已有运行中的循环，使用线程池执行
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, func(*args, **kwargs))
                    return future.result()
            except RuntimeError:
                # 没有运行中的循环，直接运行
                return asyncio.run(func(*args, **kwargs))
                
        except ImportError:
            logger.error("asyncio不可用，无法执行异步函数")
            raise
    
    return wrapper


def memoize_property(func: Callable) -> property:
    """属性缓存装饰器
    
    Args:
        func: 属性方法
        
    Returns:
        property: 缓存属性
    """
    attr_name = f'_cached_{func.__name__}'
    
    @functools.wraps(func)
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    
    return property(wrapper)