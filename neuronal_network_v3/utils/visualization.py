"""可视化工具模块

该模块提供了各种可视化功能，包括：
- 训练曲线绘制
- 指标可视化
- 预测结果展示
- 发射器位置可视化
- 比较图表
- 动画创建
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from mpl_toolkits.mplot3d import Axes3D
import cv2

logger = logging.getLogger(__name__)

# 设置matplotlib样式
plt.style.use('default')
sns.set_palette("husl")


def setup_matplotlib(style: str = "default", 
                    font_size: int = 12,
                    figure_size: Tuple[int, int] = (10, 6),
                    dpi: int = 100) -> None:
    """设置matplotlib参数
    
    Args:
        style: 绘图样式
        font_size: 字体大小
        figure_size: 图形大小
        dpi: 分辨率
    """
    plt.style.use(style)
    plt.rcParams.update({
        'font.size': font_size,
        'figure.figsize': figure_size,
        'figure.dpi': dpi,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    logger.info(f"Matplotlib设置完成: style={style}, font_size={font_size}")


def plot_training_curves(train_losses: List[float],
                        val_losses: List[float],
                        train_metrics: Optional[Dict[str, List[float]]] = None,
                        val_metrics: Optional[Dict[str, List[float]]] = None,
                        save_path: Optional[Union[str, Path]] = None,
                        show: bool = True) -> plt.Figure:
    """绘制训练曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        train_metrics: 训练指标字典
        val_metrics: 验证指标字典
        save_path: 保存路径
        show: 是否显示图形
        
    Returns:
        plt.Figure: 图形对象
    """
    # 计算子图数量
    num_plots = 1  # 损失曲线
    if train_metrics and val_metrics:
        num_plots += len(train_metrics)
    
    # 创建子图
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]
    
    # 绘制损失曲线
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('训练和验证损失')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绘制指标曲线
    if train_metrics and val_metrics:
        for i, metric_name in enumerate(train_metrics.keys()):
            if metric_name in val_metrics:
                ax = axes[i + 1]
                ax.plot(epochs, train_metrics[metric_name], 'b-', 
                       label=f'训练{metric_name}', linewidth=2)
                ax.plot(epochs, val_metrics[metric_name], 'r-', 
                       label=f'验证{metric_name}', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name}曲线')
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    if show:
        plt.show()
    
    return fig


def plot_loss_curves(losses: Dict[str, List[float]],
                    save_path: Optional[Union[str, Path]] = None,
                    show: bool = True) -> plt.Figure:
    """绘制多个损失曲线
    
    Args:
        losses: 损失字典 {loss_name: [values]}
        save_path: 保存路径
        show: 是否显示图形
        
    Returns:
        plt.Figure: 图形对象
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for loss_name, loss_values in losses.items():
        epochs = range(1, len(loss_values) + 1)
        ax.plot(epochs, loss_values, label=loss_name, linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('损失函数曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    if show:
        plt.show()
    
    return fig


def plot_metrics(metrics: Dict[str, float],
                metric_type: str = "bar",
                save_path: Optional[Union[str, Path]] = None,
                show: bool = True) -> plt.Figure:
    """绘制指标图表
    
    Args:
        metrics: 指标字典
        metric_type: 图表类型 (bar, radar)
        save_path: 保存路径
        show: 是否显示图形
        
    Returns:
        plt.Figure: 图形对象
    """
    if metric_type == "bar":
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax.bar(metric_names, metric_values, alpha=0.7)
        
        # 添加数值标签
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel('值')
        ax.set_title('评估指标')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        
    elif metric_type == "radar":
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False)
        metric_values += metric_values[:1]  # 闭合图形
        angles = np.concatenate((angles, [angles[0]]))
        
        ax.plot(angles, metric_values, 'o-', linewidth=2)
        ax.fill(angles, metric_values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_title('评估指标雷达图')
        
    else:
        raise ValueError(f"不支持的图表类型: {metric_type}")
    
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    if show:
        plt.show()
    
    return fig


def plot_predictions(image: np.ndarray,
                    predictions: np.ndarray,
                    ground_truth: Optional[np.ndarray] = None,
                    channel_names: Optional[List[str]] = None,
                    save_path: Optional[Union[str, Path]] = None,
                    show: bool = True) -> plt.Figure:
    """绘制预测结果
    
    Args:
        image: 输入图像
        predictions: 预测结果 (C, H, W)
        ground_truth: 真实标签 (C, H, W)
        channel_names: 通道名称列表
        save_path: 保存路径
        show: 是否显示图形
        
    Returns:
        plt.Figure: 图形对象
    """
    num_channels = predictions.shape[0]
    if channel_names is None:
        channel_names = [f'Channel {i}' for i in range(num_channels)]
    
    # 计算子图布局
    cols = min(4, num_channels + 1)  # +1 for input image
    rows = (num_channels + 1 + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    # 绘制输入图像
    im0 = axes[0].imshow(image, cmap='gray')
    axes[0].set_title('输入图像')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 绘制预测结果
    for i in range(num_channels):
        ax = axes[i + 1]
        im = ax.imshow(predictions[i], cmap='viridis')
        ax.set_title(f'预测: {channel_names[i]}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 如果有真实标签，绘制比较
    if ground_truth is not None and len(axes) > num_channels + 1:
        for i in range(min(num_channels, len(axes) - num_channels - 1)):
            ax = axes[num_channels + 1 + i]
            im = ax.imshow(ground_truth[i], cmap='viridis')
            ax.set_title(f'真实: {channel_names[i]}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 隐藏多余的子图
    for i in range(len(axes)):
        if i >= num_channels + 1 + (num_channels if ground_truth is not None else 0):
            axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    if show:
        plt.show()
    
    return fig


def plot_emitters(image: np.ndarray,
                 emitters: np.ndarray,
                 predictions: Optional[np.ndarray] = None,
                 pixel_size: float = 100.0,
                 save_path: Optional[Union[str, Path]] = None,
                 show: bool = True) -> plt.Figure:
    """绘制发射器位置
    
    Args:
        image: 背景图像
        emitters: 发射器位置 (N, 3) [x, y, z] 或 (N, 2) [x, y]
        predictions: 预测的发射器位置
        pixel_size: 像素大小（nm）
        save_path: 保存路径
        show: 是否显示图形
        
    Returns:
        plt.Figure: 图形对象
    """
    is_3d = emitters.shape[1] >= 3
    
    if is_3d:
        fig = plt.figure(figsize=(15, 5))
        
        # 2D投影
        ax1 = fig.add_subplot(131)
        ax1.imshow(image, cmap='gray', alpha=0.7)
        ax1.scatter(emitters[:, 0], emitters[:, 1], c='red', s=50, alpha=0.8, label='真实')
        if predictions is not None:
            ax1.scatter(predictions[:, 0], predictions[:, 1], c='blue', s=30, alpha=0.8, label='预测')
        ax1.set_title('XY投影')
        ax1.set_xlabel('X (pixel)')
        ax1.set_ylabel('Y (pixel)')
        ax1.legend()
        
        # Z分布
        ax2 = fig.add_subplot(132)
        ax2.hist(emitters[:, 2], bins=20, alpha=0.7, color='red', label='真实')
        if predictions is not None and predictions.shape[1] >= 3:
            ax2.hist(predictions[:, 2], bins=20, alpha=0.7, color='blue', label='预测')
        ax2.set_title('Z分布')
        ax2.set_xlabel('Z (nm)')
        ax2.set_ylabel('计数')
        ax2.legend()
        
        # 3D散点图
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(emitters[:, 0] * pixel_size, emitters[:, 1] * pixel_size, 
                   emitters[:, 2], c='red', s=50, alpha=0.8, label='真实')
        if predictions is not None and predictions.shape[1] >= 3:
            ax3.scatter(predictions[:, 0] * pixel_size, predictions[:, 1] * pixel_size, 
                       predictions[:, 2], c='blue', s=30, alpha=0.8, label='预测')
        ax3.set_title('3D视图')
        ax3.set_xlabel('X (nm)')
        ax3.set_ylabel('Y (nm)')
        ax3.set_zlabel('Z (nm)')
        ax3.legend()
        
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image, cmap='gray', alpha=0.7)
        ax.scatter(emitters[:, 0], emitters[:, 1], c='red', s=50, alpha=0.8, label='真实')
        if predictions is not None:
            ax.scatter(predictions[:, 0], predictions[:, 1], c='blue', s=30, alpha=0.8, label='预测')
        ax.set_title('发射器位置')
        ax.set_xlabel('X (pixel)')
        ax.set_ylabel('Y (pixel)')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    if show:
        plt.show()
    
    return fig


def plot_comparison(data1: np.ndarray,
                   data2: np.ndarray,
                   labels: Tuple[str, str] = ('数据1', '数据2'),
                   plot_type: str = "scatter",
                   save_path: Optional[Union[str, Path]] = None,
                   show: bool = True) -> plt.Figure:
    """绘制比较图
    
    Args:
        data1: 第一组数据
        data2: 第二组数据
        labels: 数据标签
        plot_type: 图表类型 (scatter, histogram, box)
        save_path: 保存路径
        show: 是否显示图形
        
    Returns:
        plt.Figure: 图形对象
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if plot_type == "scatter":
        # 散点图比较
        min_val = min(np.min(data1), np.min(data2))
        max_val = max(np.max(data1), np.max(data2))
        
        axes[0].scatter(data1, data2, alpha=0.6)
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
        axes[0].set_xlabel(labels[0])
        axes[0].set_ylabel(labels[1])
        axes[0].set_title('散点图比较')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 残差图
        residuals = data2 - data1
        axes[1].scatter(data1, residuals, alpha=0.6)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel(labels[0])
        axes[1].set_ylabel('残差')
        axes[1].set_title('残差图')
        axes[1].grid(True, alpha=0.3)
        
    elif plot_type == "histogram":
        # 直方图比较
        axes[0].hist(data1, bins=30, alpha=0.7, label=labels[0])
        axes[0].hist(data2, bins=30, alpha=0.7, label=labels[1])
        axes[0].set_xlabel('值')
        axes[0].set_ylabel('频率')
        axes[0].set_title('分布比较')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 累积分布
        sorted_data1 = np.sort(data1)
        sorted_data2 = np.sort(data2)
        y1 = np.arange(1, len(sorted_data1) + 1) / len(sorted_data1)
        y2 = np.arange(1, len(sorted_data2) + 1) / len(sorted_data2)
        
        axes[1].plot(sorted_data1, y1, label=labels[0])
        axes[1].plot(sorted_data2, y2, label=labels[1])
        axes[1].set_xlabel('值')
        axes[1].set_ylabel('累积概率')
        axes[1].set_title('累积分布函数')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
    elif plot_type == "box":
        # 箱线图比较
        axes[0].boxplot([data1, data2], labels=labels)
        axes[0].set_title('箱线图比较')
        axes[0].grid(True, alpha=0.3)
        
        # 小提琴图
        axes[1].violinplot([data1, data2], positions=[1, 2])
        axes[1].set_xticks([1, 2])
        axes[1].set_xticklabels(labels)
        axes[1].set_title('小提琴图比较')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    if show:
        plt.show()
    
    return fig


def create_animation(images: List[np.ndarray],
                    interval: int = 100,
                    save_path: Optional[Union[str, Path]] = None,
                    show: bool = True) -> FuncAnimation:
    """创建动画
    
    Args:
        images: 图像列表
        interval: 帧间隔（毫秒）
        save_path: 保存路径
        show: 是否显示动画
        
    Returns:
        FuncAnimation: 动画对象
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 初始化图像
    im = ax.imshow(images[0], cmap='viridis')
    ax.set_title('帧 0')
    ax.axis('off')
    
    def update(frame):
        im.set_array(images[frame])
        ax.set_title(f'帧 {frame}')
        return [im]
    
    anim = FuncAnimation(fig, update, frames=len(images), 
                        interval=interval, blit=True, repeat=True)
    
    if save_path:
        save_path = Path(save_path)
        if save_path.suffix == '.gif':
            anim.save(save_path, writer='pillow', fps=1000//interval)
        elif save_path.suffix == '.mp4':
            anim.save(save_path, writer='ffmpeg', fps=1000//interval)
        logger.info(f"动画保存完成: {save_path}")
    
    if show:
        plt.show()
    
    return anim


def save_plot(fig: plt.Figure, 
             save_path: Union[str, Path],
             dpi: int = 300,
             bbox_inches: str = 'tight') -> None:
    """保存图形
    
    Args:
        fig: 图形对象
        save_path: 保存路径
        dpi: 分辨率
        bbox_inches: 边界框设置
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, 
               facecolor='white', edgecolor='none')
    logger.info(f"图形保存完成: {save_path}")


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         normalize: bool = True,
                         save_path: Optional[Union[str, Path]] = None,
                         show: bool = True) -> plt.Figure:
    """绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        normalize: 是否归一化
        save_path: 保存路径
        show: 是否显示图形
        
    Returns:
        plt.Figure: 图形对象
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    ax.set_title('混淆矩阵')
    
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    if show:
        plt.show()
    
    return fig


def plot_roc_curve(y_true: np.ndarray,
                  y_scores: np.ndarray,
                  save_path: Optional[Union[str, Path]] = None,
                  show: bool = True) -> plt.Figure:
    """绘制ROC曲线
    
    Args:
        y_true: 真实标签
        y_scores: 预测分数
        save_path: 保存路径
        show: 是否显示图形
        
    Returns:
        plt.Figure: 图形对象
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
           label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假正率')
    ax.set_ylabel('真正率')
    ax.set_title('ROC曲线')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_plot(fig, save_path)
    
    if show:
        plt.show()
    
    return fig