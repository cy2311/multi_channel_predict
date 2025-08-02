"""工具模块

提供各种实用工具和辅助功能，包括：
- 配置管理
- 数据处理
- 日志工具
- 数学工具
- 可视化工具
- 监控工具
- 装饰器
- 工厂模式
- IO工具
"""

# 配置相关
from .config import (
    TrainingConfig,
    DataConfig,
    ModelConfig,
    OptimizationConfig,
    InferenceConfig,
    EvaluationConfig,
    Config,
    load_config,
    save_config,
    merge_configs,
    validate_config
)

# 数据处理相关
from .data_utils import (
    load_hdf5_data,
    save_hdf5_data,
    normalize_data,
    augment_data,
    split_dataset,
    balance_dataset,
    calculate_dataset_stats,
    convert_data_format,
    validate_data_format,
    create_data_splits,
    apply_transforms
)

# 数学工具
from .math_utils import (
    gaussian_2d,
    gaussian_3d,
    calculate_fwhm,
    fit_gaussian,
    calculate_snr,
    apply_noise,
    normalize_array,
    standardize_array,
    clip_array,
    smooth_array,
    interpolate_array,
    calculate_statistics,
    calculate_skewness,
    calculate_kurtosis,
    find_peaks,
    calculate_correlation,
    calculate_mutual_information,
    fit_polynomial,
    moving_average,
    calculate_gradient,
    calculate_laplacian
)

# 日志工具
from .logging_utils import (
    get_logger,
    log_exception,
    ColoredFormatter,
    JSONFormatter,
    ContextLogger,
    PerformanceLogger
)

# 可视化工具
from .visualization import (
    setup_matplotlib,
    plot_training_curves,
    plot_loss_curves,
    plot_metrics,
    plot_predictions,
    plot_emitters,
    plot_comparison,
    create_animation,
    save_plot,
    plot_confusion_matrix,
    plot_roc_curve
)

# 监控工具
from .monitoring import (
    SystemMonitor,
    GPUMonitor,
    TrainingMonitor,
    PerformanceProfiler,
    MonitoringManager
)

# 装饰器
from .decorators import (
    timer,
    retry,
    cache,
    validate_types,
    deprecated,
    singleton,
    rate_limit,
    exception_handler,
    thread_safe
)

# 工厂模式
from .factories import (
    create_model,
    create_loss_function,
    create_optimizer,
    create_scheduler,
    get_all_available_components,
    print_available_components
)

# IO工具
from .io_utils import (
    create_directory,
    backup_file,
    compress_file,
    extract_file,
    copy_files,
    move_files,
    delete_files,
    get_file_size,
    get_directory_size,
    find_files,
    read_json,
    write_json,
    read_yaml,
    write_yaml,
    format_file_size,
    get_file_info,
    clean_directory
)

# 常量
from .constants import *

__all__ = [
    # 配置相关
    'TrainingConfig',
    'DataConfig', 
    'ModelConfig',
    'OptimizationConfig',
    'InferenceConfig',
    'EvaluationConfig',
    'Config',
    'load_config',
    'save_config',
    'merge_configs',
    'validate_config',
    
    # 数据处理相关
    'load_h5_file',
    'save_h5_file',
    'load_tiff_stack',
    'save_tiff_stack',
    'normalize_image',
    'denormalize_image',
    'apply_gaussian_filter',
    'apply_median_filter',
    'crop_image',
    'pad_image',
    'resize_image',
    'augment_image',
    'split_dataset',
    'create_data_loader',
    'validate_data_format',
    'convert_coordinates',
    'filter_emitters',
    'merge_emitters',
    'calculate_density',
    'generate_grid_coordinates',
    
    # 数学工具
    'gaussian_2d',
    'gaussian_3d',
    'calculate_fwhm',
    'fit_gaussian',
    'calculate_snr',
    'calculate_crlb',
    'apply_nms',
    'calculate_iou',
    'calculate_distance_matrix',
    'find_local_maxima',
    'calculate_precision_recall',
    'calculate_f1_score',
    'calculate_jaccard_index',
    'smooth_array',
    'interpolate_array',
    'calculate_statistics',
    
    # 日志工具
    'get_logger',
    'log_exception',
    'ColoredFormatter',
    'JSONFormatter',
    'ContextLogger',
    'PerformanceLogger',
    
    # 可视化工具
    'plot_training_curves',
    'plot_loss_curves',
    'plot_metrics',
    'plot_predictions',
    'plot_emitters',
    'plot_density_map',
    'plot_precision_recall',
    'plot_roc_curve',
    'save_plot',
    'create_animation',
    'plot_3d_scatter',
    'plot_heatmap',
    'plot_histogram',
    'create_subplot_grid',
    
    # 监控工具
    'SystemMonitor',
    'GPUMonitor',
    'TrainingMonitor',
    'MemoryTracker',
    'TimeProfiler',
    
    # 装饰器
    'timer',
    'memory_monitor',
    'gpu_monitor',
    'retry',
    'cache_result',
    'validate_input',
    'log_calls',
    'profile_memory',
    'profile_time',
    
    # 工厂模式
    'create_model',
    'create_loss_function',
    'create_optimizer',
    'create_scheduler',
    'create_dataset',
    'create_data_loader',
    'create_evaluator',
    'create_visualizer',
    
    # IO工具
    'ensure_dir',
    'copy_file',
    'move_file',
    'delete_file',
    'list_files',
    'get_file_size',
    'get_file_info',
    'compress_file',
    'decompress_file',
    'backup_file',
    'create_symlink',
    'watch_directory',
    'safe_save',
    'atomic_write'
]