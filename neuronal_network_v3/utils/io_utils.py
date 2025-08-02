"""I/O工具模块

该模块提供了文件和目录操作相关的工具函数，包括：
- 文件和目录管理
- 数据文件读写
- 压缩和解压缩
- 文件搜索和统计
"""

import os
import json
import yaml
import shutil
import zipfile
import tarfile
import gzip
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Generator
import logging
import glob
from datetime import datetime

logger = logging.getLogger(__name__)


def create_directory(dir_path: Union[str, Path], 
                    exist_ok: bool = True,
                    parents: bool = True) -> Path:
    """创建目录
    
    Args:
        dir_path: 目录路径
        exist_ok: 如果目录已存在是否报错
        parents: 是否创建父目录
        
    Returns:
        Path: 目录路径对象
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=parents, exist_ok=exist_ok)
    logger.info(f"目录创建完成: {dir_path}")
    return dir_path


def backup_file(file_path: Union[str, Path], 
               backup_dir: Optional[Union[str, Path]] = None,
               add_timestamp: bool = True) -> Path:
    """备份文件
    
    Args:
        file_path: 源文件路径
        backup_dir: 备份目录，None表示在原目录
        add_timestamp: 是否添加时间戳
        
    Returns:
        Path: 备份文件路径
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 确定备份目录
    if backup_dir is None:
        backup_dir = file_path.parent
    else:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成备份文件名
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
    else:
        backup_name = f"{file_path.stem}_backup{file_path.suffix}"
    
    backup_path = backup_dir / backup_name
    
    # 复制文件
    shutil.copy2(file_path, backup_path)
    logger.info(f"文件备份完成: {file_path} -> {backup_path}")
    
    return backup_path


def compress_file(file_path: Union[str, Path],
                 output_path: Optional[Union[str, Path]] = None,
                 compression_type: str = "gzip") -> Path:
    """压缩文件
    
    Args:
        file_path: 源文件路径
        output_path: 输出路径
        compression_type: 压缩类型 (gzip, zip, tar)
        
    Returns:
        Path: 压缩文件路径
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    if output_path is None:
        if compression_type == "gzip":
            output_path = file_path.with_suffix(file_path.suffix + ".gz")
        elif compression_type == "zip":
            output_path = file_path.with_suffix(".zip")
        elif compression_type == "tar":
            output_path = file_path.with_suffix(".tar.gz")
    else:
        output_path = Path(output_path)
    
    if compression_type == "gzip":
        with open(file_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
    elif compression_type == "zip":
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(file_path, file_path.name)
            
    elif compression_type == "tar":
        with tarfile.open(output_path, 'w:gz') as tar:
            tar.add(file_path, arcname=file_path.name)
    
    else:
        raise ValueError(f"不支持的压缩类型: {compression_type}")
    
    logger.info(f"文件压缩完成: {file_path} -> {output_path}")
    return output_path


def extract_file(archive_path: Union[str, Path],
                extract_dir: Optional[Union[str, Path]] = None) -> Path:
    """解压文件
    
    Args:
        archive_path: 压缩文件路径
        extract_dir: 解压目录
        
    Returns:
        Path: 解压目录路径
    """
    archive_path = Path(archive_path)
    
    if not archive_path.exists():
        raise FileNotFoundError(f"压缩文件不存在: {archive_path}")
    
    if extract_dir is None:
        extract_dir = archive_path.parent / archive_path.stem
    else:
        extract_dir = Path(extract_dir)
    
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    # 根据文件扩展名选择解压方法
    if archive_path.suffix == '.gz' and archive_path.stem.endswith('.tar'):
        # tar.gz文件
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
    elif archive_path.suffix == '.zip':
        # zip文件
        with zipfile.ZipFile(archive_path, 'r') as zipf:
            zipf.extractall(extract_dir)
    elif archive_path.suffix == '.gz':
        # gzip文件
        output_file = extract_dir / archive_path.stem
        with gzip.open(archive_path, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        raise ValueError(f"不支持的压缩格式: {archive_path.suffix}")
    
    logger.info(f"文件解压完成: {archive_path} -> {extract_dir}")
    return extract_dir


def copy_files(src_pattern: str,
              dst_dir: Union[str, Path],
              preserve_structure: bool = False) -> List[Path]:
    """复制文件
    
    Args:
        src_pattern: 源文件模式（支持通配符）
        dst_dir: 目标目录
        preserve_structure: 是否保持目录结构
        
    Returns:
        List[Path]: 复制的文件列表
    """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    src_files = glob.glob(src_pattern, recursive=True)
    copied_files = []
    
    for src_file in src_files:
        src_path = Path(src_file)
        
        if preserve_structure:
            # 保持相对路径结构
            rel_path = src_path.relative_to(Path(src_pattern).parent)
            dst_path = dst_dir / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # 直接复制到目标目录
            dst_path = dst_dir / src_path.name
        
        shutil.copy2(src_path, dst_path)
        copied_files.append(dst_path)
        logger.debug(f"文件复制: {src_path} -> {dst_path}")
    
    logger.info(f"文件复制完成: {len(copied_files)}个文件")
    return copied_files


def move_files(src_pattern: str,
              dst_dir: Union[str, Path],
              preserve_structure: bool = False) -> List[Path]:
    """移动文件
    
    Args:
        src_pattern: 源文件模式（支持通配符）
        dst_dir: 目标目录
        preserve_structure: 是否保持目录结构
        
    Returns:
        List[Path]: 移动的文件列表
    """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    src_files = glob.glob(src_pattern, recursive=True)
    moved_files = []
    
    for src_file in src_files:
        src_path = Path(src_file)
        
        if preserve_structure:
            # 保持相对路径结构
            rel_path = src_path.relative_to(Path(src_pattern).parent)
            dst_path = dst_dir / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # 直接移动到目标目录
            dst_path = dst_dir / src_path.name
        
        shutil.move(str(src_path), str(dst_path))
        moved_files.append(dst_path)
        logger.debug(f"文件移动: {src_path} -> {dst_path}")
    
    logger.info(f"文件移动完成: {len(moved_files)}个文件")
    return moved_files


def delete_files(file_pattern: str, confirm: bool = True) -> int:
    """删除文件
    
    Args:
        file_pattern: 文件模式（支持通配符）
        confirm: 是否需要确认
        
    Returns:
        int: 删除的文件数量
    """
    files = glob.glob(file_pattern, recursive=True)
    
    if not files:
        logger.info("没有找到匹配的文件")
        return 0
    
    if confirm:
        print(f"将要删除 {len(files)} 个文件:")
        for file in files[:10]:  # 只显示前10个
            print(f"  {file}")
        if len(files) > 10:
            print(f"  ... 还有 {len(files) - 10} 个文件")
        
        response = input("确认删除？(y/N): ")
        if response.lower() != 'y':
            logger.info("删除操作已取消")
            return 0
    
    deleted_count = 0
    for file in files:
        try:
            os.remove(file)
            deleted_count += 1
            logger.debug(f"文件删除: {file}")
        except Exception as e:
            logger.error(f"删除文件失败 {file}: {e}")
    
    logger.info(f"文件删除完成: {deleted_count}个文件")
    return deleted_count


def get_file_size(file_path: Union[str, Path]) -> int:
    """获取文件大小
    
    Args:
        file_path: 文件路径
        
    Returns:
        int: 文件大小（字节）
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    return file_path.stat().st_size


def get_directory_size(dir_path: Union[str, Path]) -> int:
    """获取目录大小
    
    Args:
        dir_path: 目录路径
        
    Returns:
        int: 目录大小（字节）
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {dir_path}")
    
    total_size = 0
    for file_path in dir_path.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    return total_size


def find_files(directory: Union[str, Path],
              pattern: str = "*",
              recursive: bool = True,
              file_type: str = "all") -> List[Path]:
    """查找文件
    
    Args:
        directory: 搜索目录
        pattern: 文件名模式
        recursive: 是否递归搜索
        file_type: 文件类型 (all, file, dir)
        
    Returns:
        List[Path]: 找到的文件列表
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    if recursive:
        search_pattern = directory.rglob(pattern)
    else:
        search_pattern = directory.glob(pattern)
    
    results = []
    for path in search_pattern:
        if file_type == "all":
            results.append(path)
        elif file_type == "file" and path.is_file():
            results.append(path)
        elif file_type == "dir" and path.is_dir():
            results.append(path)
    
    logger.info(f"文件搜索完成: 找到{len(results)}个{file_type}")
    return results


def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """读取JSON文件
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        Dict[str, Any]: JSON数据
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSON文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.debug(f"JSON文件读取完成: {file_path}")
    return data


def write_json(data: Dict[str, Any], 
              file_path: Union[str, Path],
              indent: int = 2,
              ensure_ascii: bool = False) -> None:
    """写入JSON文件
    
    Args:
        data: 要写入的数据
        file_path: JSON文件路径
        indent: 缩进空格数
        ensure_ascii: 是否确保ASCII编码
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
    
    logger.debug(f"JSON文件写入完成: {file_path}")


def read_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """读取YAML文件
    
    Args:
        file_path: YAML文件路径
        
    Returns:
        Dict[str, Any]: YAML数据
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"YAML文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    logger.debug(f"YAML文件读取完成: {file_path}")
    return data


def write_yaml(data: Dict[str, Any], 
              file_path: Union[str, Path],
              default_flow_style: bool = False) -> None:
    """写入YAML文件
    
    Args:
        data: 要写入的数据
        file_path: YAML文件路径
        default_flow_style: 是否使用流式样式
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=default_flow_style, 
                 allow_unicode=True, indent=2)
    
    logger.debug(f"YAML文件写入完成: {file_path}")


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小
    
    Args:
        size_bytes: 文件大小（字节）
        
    Returns:
        str: 格式化的文件大小
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """获取文件信息
    
    Args:
        file_path: 文件路径
        
    Returns:
        Dict[str, Any]: 文件信息
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    stat = file_path.stat()
    
    info = {
        "name": file_path.name,
        "path": str(file_path.absolute()),
        "size": stat.st_size,
        "size_formatted": format_file_size(stat.st_size),
        "created": datetime.fromtimestamp(stat.st_ctime),
        "modified": datetime.fromtimestamp(stat.st_mtime),
        "accessed": datetime.fromtimestamp(stat.st_atime),
        "is_file": file_path.is_file(),
        "is_dir": file_path.is_dir(),
        "suffix": file_path.suffix,
        "stem": file_path.stem
    }
    
    return info


def clean_directory(directory: Union[str, Path],
                   older_than_days: int = 7,
                   pattern: str = "*",
                   dry_run: bool = True) -> int:
    """清理目录中的旧文件
    
    Args:
        directory: 目录路径
        older_than_days: 删除多少天前的文件
        pattern: 文件模式
        dry_run: 是否只是预览，不实际删除
        
    Returns:
        int: 删除的文件数量
    """
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    cutoff_time = datetime.now().timestamp() - (older_than_days * 24 * 3600)
    deleted_count = 0
    
    for file_path in directory.rglob(pattern):
        if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
            if dry_run:
                logger.info(f"[预览] 将删除: {file_path}")
            else:
                try:
                    file_path.unlink()
                    logger.debug(f"删除文件: {file_path}")
                except Exception as e:
                    logger.error(f"删除文件失败 {file_path}: {e}")
                    continue
            deleted_count += 1
    
    action = "将删除" if dry_run else "已删除"
    logger.info(f"目录清理完成: {action}{deleted_count}个文件")
    return deleted_count