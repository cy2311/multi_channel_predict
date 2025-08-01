"""结果解析器模块

用于解析和格式化推理结果，支持多种输出格式。
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import json


@dataclass
class EmitterResult:
    """单个发射器结果
    
    Attributes:
        x, y, z: 3D坐标
        photons: 光子数
        prob: 检测概率
        bg: 背景（可选）
        uncertainty: 不确定性信息（可选）
        frame: 帧号（可选）
        id: 发射器ID（可选）
    """
    x: float
    y: float
    z: float
    photons: float
    prob: float
    bg: Optional[float] = None
    x_sigma: Optional[float] = None
    y_sigma: Optional[float] = None
    z_sigma: Optional[float] = None
    photons_sigma: Optional[float] = None
    frame: Optional[int] = None
    id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'photons': self.photons,
            'prob': self.prob
        }
        
        if self.bg is not None:
            result['bg'] = self.bg
        
        if self.x_sigma is not None:
            result['x_sigma'] = self.x_sigma
            result['y_sigma'] = self.y_sigma
            result['z_sigma'] = self.z_sigma
            result['photons_sigma'] = self.photons_sigma
        
        if self.frame is not None:
            result['frame'] = self.frame
        
        if self.id is not None:
            result['id'] = self.id
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmitterResult':
        """从字典创建"""
        return cls(**data)


@dataclass
class DetectionResult:
    """检测结果
    
    Attributes:
        emitters: 发射器列表
        num_emitters: 发射器数量
        total_photons: 总光子数
        density: 发射器密度
        frame: 帧号（可选）
        metadata: 元数据
    """
    emitters: List[EmitterResult]
    num_emitters: int
    total_photons: float
    density: float
    frame: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'emitters': [emitter.to_dict() for emitter in self.emitters],
            'num_emitters': self.num_emitters,
            'total_photons': self.total_photons,
            'density': self.density
        }
        
        if self.frame is not None:
            result['frame'] = self.frame
        
        if self.metadata is not None:
            result['metadata'] = self.metadata
        
        return result
    
    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        if not self.emitters:
            return pd.DataFrame()
        
        data = [emitter.to_dict() for emitter in self.emitters]
        return pd.DataFrame(data)
    
    def filter_by_photons(self, min_photons: float, max_photons: float = float('inf')) -> 'DetectionResult':
        """按光子数过滤"""
        filtered_emitters = [
            emitter for emitter in self.emitters
            if min_photons <= emitter.photons <= max_photons
        ]
        
        return DetectionResult(
            emitters=filtered_emitters,
            num_emitters=len(filtered_emitters),
            total_photons=sum(e.photons for e in filtered_emitters),
            density=len(filtered_emitters) / (self.density * self.num_emitters) if self.num_emitters > 0 else 0,
            frame=self.frame,
            metadata=self.metadata
        )
    
    def filter_by_prob(self, min_prob: float) -> 'DetectionResult':
        """按概率过滤"""
        filtered_emitters = [
            emitter for emitter in self.emitters
            if emitter.prob >= min_prob
        ]
        
        return DetectionResult(
            emitters=filtered_emitters,
            num_emitters=len(filtered_emitters),
            total_photons=sum(e.photons for e in filtered_emitters),
            density=len(filtered_emitters) / (self.density * self.num_emitters) if self.num_emitters > 0 else 0,
            frame=self.frame,
            metadata=self.metadata
        )


class ResultParser:
    """结果解析器基类"""
    
    def __call__(self, processed_output: Dict[str, Any]) -> Dict[str, Any]:
        """解析处理后的输出
        
        Args:
            processed_output: 后处理器的输出
            
        Returns:
            解析后的结果
        """
        raise NotImplementedError


class StandardResultParser(ResultParser):
    """标准结果解析器
    
    将后处理器的输出转换为标准的结果格式。
    
    Args:
        pixel_size: 像素大小（nm）
        frame_number: 帧号
        add_ids: 是否添加发射器ID
        coordinate_system: 坐标系统 ('pixel', 'nm')
    """
    
    def __init__(self,
                 pixel_size: float = 100.0,
                 frame_number: Optional[int] = None,
                 add_ids: bool = True,
                 coordinate_system: str = 'pixel'):
        
        self.pixel_size = pixel_size
        self.frame_number = frame_number
        self.add_ids = add_ids
        self.coordinate_system = coordinate_system
    
    def __call__(self, processed_output: Dict[str, Any]) -> Dict[str, Any]:
        """解析结果"""
        emitters_dict = processed_output.get('emitters', {})
        
        if not emitters_dict or len(emitters_dict.get('x', [])) == 0:
            return self._empty_result()
        
        # 创建发射器列表
        emitters = self._create_emitters(emitters_dict)
        
        # 计算统计信息
        num_emitters = len(emitters)
        total_photons = sum(e.photons for e in emitters)
        density = processed_output.get('density', 0.0)
        
        # 创建检测结果
        detection_result = DetectionResult(
            emitters=emitters,
            num_emitters=num_emitters,
            total_photons=total_photons,
            density=density,
            frame=self.frame_number,
            metadata=self._create_metadata(processed_output)
        )
        
        return {
            'detection_result': detection_result,
            'emitters_array': self._emitters_to_array(emitters),
            'summary': self._create_summary(detection_result)
        }
    
    def _create_emitters(self, emitters_dict: Dict[str, np.ndarray]) -> List[EmitterResult]:
        """创建发射器列表"""
        num_emitters = len(emitters_dict['x'])
        emitters = []
        
        for i in range(num_emitters):
            # 基本属性
            x = float(emitters_dict['x'][i])
            y = float(emitters_dict['y'][i])
            z = float(emitters_dict['z'][i])
            photons = float(emitters_dict['photons'][i])
            prob = float(emitters_dict['prob'][i])
            
            # 坐标转换
            if self.coordinate_system == 'nm':
                x *= self.pixel_size
                y *= self.pixel_size
                z *= self.pixel_size
            
            # 可选属性
            bg = float(emitters_dict['bg'][i]) if 'bg' in emitters_dict else None
            
            x_sigma = float(emitters_dict['x_sigma'][i]) if 'x_sigma' in emitters_dict else None
            y_sigma = float(emitters_dict['y_sigma'][i]) if 'y_sigma' in emitters_dict else None
            z_sigma = float(emitters_dict['z_sigma'][i]) if 'z_sigma' in emitters_dict else None
            photons_sigma = float(emitters_dict['photons_sigma'][i]) if 'photons_sigma' in emitters_dict else None
            
            # 不确定性坐标转换
            if self.coordinate_system == 'nm' and x_sigma is not None:
                x_sigma *= self.pixel_size
                y_sigma *= self.pixel_size
                z_sigma *= self.pixel_size
            
            emitter = EmitterResult(
                x=x, y=y, z=z,
                photons=photons, prob=prob,
                bg=bg,
                x_sigma=x_sigma, y_sigma=y_sigma, z_sigma=z_sigma,
                photons_sigma=photons_sigma,
                frame=self.frame_number,
                id=i if self.add_ids else None
            )
            
            emitters.append(emitter)
        
        return emitters
    
    def _emitters_to_array(self, emitters: List[EmitterResult]) -> np.ndarray:
        """将发射器转换为数组格式"""
        if not emitters:
            return np.empty((0, 5))
        
        # 基本列：x, y, z, photons, prob
        data = []
        for emitter in emitters:
            row = [emitter.x, emitter.y, emitter.z, emitter.photons, emitter.prob]
            
            # 添加背景
            if emitter.bg is not None:
                row.append(emitter.bg)
            
            # 添加不确定性
            if emitter.x_sigma is not None:
                row.extend([emitter.x_sigma, emitter.y_sigma, emitter.z_sigma, emitter.photons_sigma])
            
            # 添加帧号和ID
            if emitter.frame is not None:
                row.append(emitter.frame)
            if emitter.id is not None:
                row.append(emitter.id)
            
            data.append(row)
        
        return np.array(data)
    
    def _create_metadata(self, processed_output: Dict[str, Any]) -> Dict[str, Any]:
        """创建元数据"""
        metadata = {
            'pixel_size': self.pixel_size,
            'coordinate_system': self.coordinate_system,
            'parser_type': 'StandardResultParser'
        }
        
        # 添加处理参数
        if 'adaptive_threshold' in processed_output:
            metadata['adaptive_threshold'] = processed_output['adaptive_threshold']
        if 'adaptive_min_distance' in processed_output:
            metadata['adaptive_min_distance'] = processed_output['adaptive_min_distance']
        
        return metadata
    
    def _create_summary(self, detection_result: DetectionResult) -> Dict[str, Any]:
        """创建摘要信息"""
        if not detection_result.emitters:
            return {
                'num_emitters': 0,
                'total_photons': 0.0,
                'avg_photons': 0.0,
                'std_photons': 0.0,
                'density': 0.0
            }
        
        photons = [e.photons for e in detection_result.emitters]
        probs = [e.prob for e in detection_result.emitters]
        
        return {
            'num_emitters': detection_result.num_emitters,
            'total_photons': detection_result.total_photons,
            'avg_photons': np.mean(photons),
            'std_photons': np.std(photons),
            'min_photons': np.min(photons),
            'max_photons': np.max(photons),
            'avg_prob': np.mean(probs),
            'min_prob': np.min(probs),
            'max_prob': np.max(probs),
            'density': detection_result.density
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        detection_result = DetectionResult(
            emitters=[],
            num_emitters=0,
            total_photons=0.0,
            density=0.0,
            frame=self.frame_number,
            metadata=self._create_metadata({})
        )
        
        return {
            'detection_result': detection_result,
            'emitters_array': np.empty((0, 5)),
            'summary': self._create_summary(detection_result)
        }


class BatchResultParser(ResultParser):
    """批量结果解析器
    
    用于处理批量推理结果。
    
    Args:
        base_parser: 基础解析器
        combine_results: 是否合并结果
        frame_offset: 帧偏移量
    """
    
    def __init__(self,
                 base_parser: ResultParser,
                 combine_results: bool = True,
                 frame_offset: int = 0):
        
        self.base_parser = base_parser
        self.combine_results = combine_results
        self.frame_offset = frame_offset
    
    def parse_batch(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """解析批量结果
        
        Args:
            batch_results: 批量推理结果列表
            
        Returns:
            解析后的批量结果
        """
        parsed_results = []
        
        for i, result in enumerate(batch_results):
            # 设置帧号
            if hasattr(self.base_parser, 'frame_number'):
                self.base_parser.frame_number = self.frame_offset + i
            
            # 解析单个结果
            parsed_result = self.base_parser(result)
            parsed_results.append(parsed_result)
        
        if self.combine_results:
            return self._combine_results(parsed_results)
        else:
            return {'batch_results': parsed_results}
    
    def _combine_results(self, parsed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并结果"""
        all_emitters = []
        total_photons = 0.0
        total_emitters = 0
        
        for result in parsed_results:
            detection_result = result['detection_result']
            all_emitters.extend(detection_result.emitters)
            total_photons += detection_result.total_photons
            total_emitters += detection_result.num_emitters
        
        # 重新分配ID
        for i, emitter in enumerate(all_emitters):
            emitter.id = i
        
        # 创建合并的检测结果
        combined_result = DetectionResult(
            emitters=all_emitters,
            num_emitters=total_emitters,
            total_photons=total_photons,
            density=total_emitters / len(parsed_results) if parsed_results else 0.0,
            metadata={'num_frames': len(parsed_results)}
        )
        
        return {
            'detection_result': combined_result,
            'emitters_array': self._emitters_to_array(all_emitters),
            'summary': self._create_batch_summary(parsed_results),
            'frame_results': parsed_results
        }
    
    def _emitters_to_array(self, emitters: List[EmitterResult]) -> np.ndarray:
        """将发射器转换为数组"""
        if not emitters:
            return np.empty((0, 6))  # x, y, z, photons, prob, frame
        
        data = []
        for emitter in emitters:
            row = [emitter.x, emitter.y, emitter.z, emitter.photons, emitter.prob]
            if emitter.frame is not None:
                row.append(emitter.frame)
            data.append(row)
        
        return np.array(data)
    
    def _create_batch_summary(self, parsed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """创建批量摘要"""
        if not parsed_results:
            return {}
        
        summaries = [result['summary'] for result in parsed_results]
        
        return {
            'total_frames': len(parsed_results),
            'total_emitters': sum(s['num_emitters'] for s in summaries),
            'total_photons': sum(s['total_photons'] for s in summaries),
            'avg_emitters_per_frame': np.mean([s['num_emitters'] for s in summaries]),
            'std_emitters_per_frame': np.std([s['num_emitters'] for s in summaries]),
            'avg_photons_per_frame': np.mean([s['total_photons'] for s in summaries]),
            'avg_density': np.mean([s['density'] for s in summaries])
        }


class ExportResultParser(ResultParser):
    """导出结果解析器
    
    支持多种导出格式的结果解析器。
    
    Args:
        base_parser: 基础解析器
        export_format: 导出格式 ('csv', 'hdf5', 'json', 'mat')
        output_path: 输出路径
    """
    
    def __init__(self,
                 base_parser: ResultParser,
                 export_format: str = 'csv',
                 output_path: Optional[Union[str, Path]] = None):
        
        self.base_parser = base_parser
        self.export_format = export_format
        self.output_path = Path(output_path) if output_path else None
    
    def __call__(self, processed_output: Dict[str, Any]) -> Dict[str, Any]:
        """解析并导出结果"""
        # 使用基础解析器解析
        result = self.base_parser(processed_output)
        
        # 导出结果
        if self.output_path is not None:
            self._export_result(result)
        
        return result
    
    def _export_result(self, result: Dict[str, Any]):
        """导出结果"""
        detection_result = result['detection_result']
        
        if self.export_format == 'csv':
            self._export_csv(detection_result)
        elif self.export_format == 'hdf5':
            self._export_hdf5(result)
        elif self.export_format == 'json':
            self._export_json(result)
        elif self.export_format == 'mat':
            self._export_mat(result)
        else:
            raise ValueError(f"Unsupported export format: {self.export_format}")
    
    def _export_csv(self, detection_result: DetectionResult):
        """导出为CSV"""
        df = detection_result.to_dataframe()
        output_file = self.output_path.with_suffix('.csv')
        df.to_csv(output_file, index=False)
    
    def _export_hdf5(self, result: Dict[str, Any]):
        """导出为HDF5"""
        import h5py
        
        output_file = self.output_path.with_suffix('.h5')
        
        with h5py.File(output_file, 'w') as f:
            # 保存发射器数组
            emitters_array = result['emitters_array']
            if len(emitters_array) > 0:
                f.create_dataset('emitters', data=emitters_array)
            
            # 保存摘要
            summary_group = f.create_group('summary')
            for key, value in result['summary'].items():
                summary_group.attrs[key] = value
            
            # 保存元数据
            metadata = result['detection_result'].metadata
            if metadata:
                metadata_group = f.create_group('metadata')
                for key, value in metadata.items():
                    metadata_group.attrs[key] = value
    
    def _export_json(self, result: Dict[str, Any]):
        """导出为JSON"""
        output_file = self.output_path.with_suffix('.json')
        
        # 转换为可序列化的格式
        json_data = {
            'detection_result': result['detection_result'].to_dict(),
            'summary': result['summary']
        }
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=self._json_serializer)
    
    def _export_mat(self, result: Dict[str, Any]):
        """导出为MATLAB格式"""
        from scipy.io import savemat
        
        output_file = self.output_path.with_suffix('.mat')
        
        # 准备MATLAB数据
        mat_data = {
            'emitters': result['emitters_array'],
            'summary': result['summary']
        }
        
        savemat(output_file, mat_data)
    
    def _json_serializer(self, obj):
        """JSON序列化器"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return str(obj)