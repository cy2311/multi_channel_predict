"""配置管理模块
统一管理系统参数和路径
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """配置管理类"""
    
    def __init__(self, config_file: str = None):
        """初始化配置
        
        Args:
            config_file: 配置文件路径，如果为None则使用默认配置
        """
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                self.config = json.load(f)
            print(f"从文件加载配置: {config_file}")
        else:
            self.config = self._get_default_config()
            print("使用默认配置")
        
        # 验证配置
        self._validate_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'optical': {
                'wavelength': 670e-9,  # 波长 (m)
                'NA': 1.4,  # 数值孔径
                'n': 1.518,  # 折射率
                'pixel_size': 65e-9,  # 像素大小 (m)
                'magnification': 100,  # 放大倍数
                'zernike_modes': 15  # Zernike模式数量
            },
            'camera': {
                'QE': 0.9,  # 量子效率
                'EMGain': 30.0,  # EM增益
                'read_noise_e': 1.0,  # 读取噪声 (电子)
                'offset': 100.0,  # 基线偏移
                'A2D': 1.0,  # A/D转换因子
                'max_adu': 65535  # 最大ADU值
            },
            'simulation': {
                'num_frames': 50,  # 帧数
                'num_emitters': 500,  # 发射器数量
                'image_size': [256, 256],  # 图像尺寸
                'upsampling_factor': 4,  # 上采样因子
                'seed': 42,  # 随机种子
                'emitter_intensity_range': [1000, 5000],  # 发射器强度范围
                'emitter_lifetime_range': [1, 10],  # 发射器生命周期范围
                'background_level': 10.0  # 背景水平
            },
            'paths': {
                'zmap_file': '../phase_retrieval_tiff2h5/result/result.h5',
                'output_dir': './result',
                'visualization_dir': './result/visualization',
                'tiff_dir': './result/tiff'
            }
        }
    
    def _validate_config(self):
        """验证配置的有效性"""
        required_sections = ['optical', 'camera', 'simulation', 'paths']
        
        for section in required_sections:
            if section not in self.config:
                print(f"警告: 配置中缺少 '{section}' 部分，使用默认值")
                default_config = self._get_default_config()
                self.config[section] = default_config[section]
        
        # 验证路径
        zmap_file = Path(self.config['paths']['zmap_file'])
        if not zmap_file.exists():
            print(f"警告: Zmap文件不存在: {zmap_file}")
    
    def get_optical_params(self) -> Dict[str, Any]:
        """获取光学参数"""
        return self.config['optical'].copy()
    
    def get_camera_params(self) -> Dict[str, Any]:
        """获取相机参数"""
        return self.config['camera'].copy()
    
    def get_simulation_params(self) -> Dict[str, Any]:
        """获取模拟参数"""
        return self.config['simulation'].copy()
    
    def get_paths(self) -> Dict[str, Path]:
        """获取路径配置"""
        paths = {}
        for key, value in self.config['paths'].items():
            paths[key] = Path(value)
        return paths
    
    def get_param(self, section: str, key: str, default=None):
        """获取特定参数
        
        Args:
            section: 配置部分名称
            key: 参数键名
            default: 默认值
            
        Returns:
            参数值
        """
        return self.config.get(section, {}).get(key, default)
    
    def set_param(self, section: str, key: str, value):
        """设置参数
        
        Args:
            section: 配置部分名称
            key: 参数键名
            value: 参数值
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def ensure_output_dirs(self):
        """确保输出目录存在"""
        paths = self.get_paths()
        
        # 创建主要输出目录
        for dir_key in ['output_dir', 'visualization_dir', 'tiff_dir']:
            if dir_key in paths:
                paths[dir_key].mkdir(parents=True, exist_ok=True)
                print(f"确保目录存在: {paths[dir_key]}")
    
    def save_config(self, output_file: Path):
        """保存当前配置到文件
        
        Args:
            output_file: 输出文件路径
        """
        # 转换Path对象为字符串以便JSON序列化
        config_to_save = self.config.copy()
        if 'paths' in config_to_save:
            for key, value in config_to_save['paths'].items():
                if isinstance(value, Path):
                    config_to_save['paths'][key] = str(value)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"配置保存到: {output_file}")
    
    def update_from_args(self, args):
        """从命令行参数更新配置
        
        Args:
            args: argparse解析的参数
        """
        if hasattr(args, 'zmap_file') and args.zmap_file:
            self.config['paths']['zmap_file'] = args.zmap_file
        
        if hasattr(args, 'output_dir') and args.output_dir:
            self.config['paths']['output_dir'] = args.output_dir
            # 同时更新子目录
            base_dir = Path(args.output_dir)
            self.config['paths']['visualization_dir'] = str(base_dir / 'visualization')
            self.config['paths']['tiff_dir'] = str(base_dir / 'tiff')
        
        if hasattr(args, 'num_frames') and args.num_frames:
            self.config['simulation']['num_frames'] = args.num_frames
        
        if hasattr(args, 'num_emitters') and args.num_emitters:
            self.config['simulation']['num_emitters'] = args.num_emitters
    
    def print_summary(self):
        """打印配置摘要"""
        print("\n配置摘要:")
        print("-" * 40)
        
        # 光学参数
        optical = self.get_optical_params()
        print(f"波长: {optical['wavelength']*1e9:.0f} nm")
        print(f"数值孔径: {optical['NA']}")
        print(f"像素大小: {optical['pixel_size']*1e9:.0f} nm")
        
        # 模拟参数
        sim = self.get_simulation_params()
        print(f"帧数: {sim['num_frames']}")
        print(f"发射器数量: {sim['num_emitters']}")
        print(f"图像尺寸: {sim['image_size']}")
        
        # 路径
        paths = self.get_paths()
        print(f"Zmap文件: {paths['zmap_file']}")
        print(f"输出目录: {paths['output_dir']}")
        print("-" * 40)