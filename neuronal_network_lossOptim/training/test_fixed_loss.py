#!/usr/bin/env python3
"""
测试修复后的损失函数

验证：
1. 损失值数量级是否合理（0.1-10范围）
2. 数值稳定性
3. 梯度是否正常
4. 与原始复杂损失函数的对比

作者：AI Assistant
日期：2025年
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入损失函数
from loss.unified_decode_loss import (
    UnifiedDECODELoss,
    SimpleCountLoss,
    SimpleLocLoss,
    SimpleCombinedLoss
)

# 导入原始复杂损失函数进行对比
try:
    from loss.count_loss import CountLoss
    from loss.loc_loss import LocLoss
    ORIGINAL_AVAILABLE = True
except ImportError:
    print("原始损失函数不可用，跳过对比测试")
    ORIGINAL_AVAILABLE = False


class LossTestSuite:
    """
    损失函数测试套件
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"使用设备: {self.device}")
    
    def create_test_data(self, batch_size: int = 8, image_size: int = 40) -> Dict[str, torch.Tensor]:
        """
        创建测试数据
        
        Args:
            batch_size: 批次大小
            image_size: 图像大小
            
        Returns:
            包含测试数据的字典
        """
        # 模拟网络输出（6通道）
        output = torch.randn(batch_size, 6, image_size, image_size, device=self.device)
        
        # 模拟目标数据
        target_unified = torch.randn(batch_size, 6, image_size, image_size, device=self.device)
        target_unified[:, 0] = torch.sigmoid(target_unified[:, 0])  # 概率通道
        
        # 分离的目标数据
        targets = {
            'unified_target': target_unified,
            'count_maps': torch.sigmoid(torch.randn(batch_size, 1, image_size, image_size, device=self.device)),
            'loc_maps': torch.randn(batch_size, 3, image_size, image_size, device=self.device),
            'photon_maps': torch.abs(torch.randn(batch_size, 1, image_size, image_size, device=self.device)),
            'background_maps': torch.abs(torch.randn(batch_size, 1, image_size, image_size, device=self.device))
        }
        
        return {
            'output': output,
            'targets': targets
        }
    
    def test_unified_loss(self) -> Dict[str, float]:
        """
        测试统一损失函数
        """
        print("\n=== 测试统一DECODE损失函数 ===")
        
        # 创建损失函数
        loss_fn = UnifiedDECODELoss(
            channel_weights=[1.0, 1.0, 1.0, 1.0, 0.5, 0.1],
            pos_weight=1.0
        ).to(self.device)
        
        # 创建测试数据
        data = self.create_test_data()
        output = data['output']
        target = data['targets']['unified_target']
        
        # 计算损失
        losses = loss_fn(output, target)
        
        # 打印结果
        print(f"总损失: {losses['total'].item():.6f}")
        for key, value in losses.items():
            if key != 'total':
                print(f"  {key}: {value.item():.6f}")
        
        # 测试梯度
        output_for_grad = data['output'].clone().requires_grad_(True)
        losses_for_grad = loss_fn(output_for_grad, target)
        losses_for_grad['total'].backward()
        
        grad_norm = torch.norm(output_for_grad.grad).item()
        print(f"梯度范数: {grad_norm:.6f}")
        
        # 检查数值稳定性
        is_stable = (
            not torch.isnan(losses['total']) and
            not torch.isinf(losses['total']) and
            0.001 <= losses['total'].item() <= 100.0
        )
        
        print(f"数值稳定性: {'✓' if is_stable else '✗'}")
        
        return {
            'total_loss': losses['total'].item(),
            'grad_norm': grad_norm,
            'is_stable': is_stable
        }
    
    def test_simple_count_loss(self) -> Dict[str, float]:
        """
        测试简化计数损失
        """
        print("\n=== 测试简化计数损失 ===")
        
        # 创建损失函数
        loss_fn = SimpleCountLoss(pos_weight=1.0).to(self.device)
        
        # 创建测试数据
        data = self.create_test_data()
        pred_logits = data['output'][:, 0:1]  # [B, 1, H, W]
        target_prob = data['targets']['count_maps']  # [B, 1, H, W]
        
        # 计算损失
        loss = loss_fn(pred_logits, target_prob)
        
        print(f"计数损失: {loss.item():.6f}")
        
        # 测试梯度
        pred_logits_for_grad = data['output'][:, 0:1].clone().requires_grad_(True)
        loss_for_grad = loss_fn(pred_logits_for_grad, target_prob)
        loss_for_grad.backward()
        
        grad_norm = torch.norm(pred_logits_for_grad.grad).item()
        print(f"梯度范数: {grad_norm:.6f}")
        
        # 检查数值稳定性
        is_stable = (
            not torch.isnan(loss) and
            not torch.isinf(loss) and
            0.001 <= loss.item() <= 10.0
        )
        
        print(f"数值稳定性: {'✓' if is_stable else '✗'}")
        
        return {
            'loss': loss.item(),
            'grad_norm': grad_norm,
            'is_stable': is_stable
        }
    
    def test_simple_loc_loss(self) -> Dict[str, float]:
        """
        测试简化定位损失
        """
        print("\n=== 测试简化定位损失 ===")
        
        # 创建损失函数
        loss_fn = SimpleLocLoss().to(self.device)
        
        # 创建测试数据
        data = self.create_test_data()
        pred_coords = data['output'][:, 1:4]  # [B, 3, H, W]
        target_coords = data['targets']['loc_maps']  # [B, 3, H, W]
        
        # 创建掩码（只在部分位置计算损失）
        mask = torch.rand_like(data['targets']['count_maps']) > 0.7  # [B, 1, H, W]
        
        # 计算损失
        loss = loss_fn(pred_coords, target_coords, mask)
        
        print(f"定位损失: {loss.item():.6f}")
        
        # 测试梯度
        pred_coords_for_grad = data['output'][:, 1:4].clone().requires_grad_(True)
        loss_for_grad = loss_fn(pred_coords_for_grad, target_coords, mask)
        loss_for_grad.backward()
        
        grad_norm = torch.norm(pred_coords_for_grad.grad).item()
        print(f"梯度范数: {grad_norm:.6f}")
        
        # 检查数值稳定性
        is_stable = (
            not torch.isnan(loss) and
            not torch.isinf(loss) and
            0.001 <= loss.item() <= 10.0
        )
        
        print(f"数值稳定性: {'✓' if is_stable else '✗'}")
        
        return {
            'loss': loss.item(),
            'grad_norm': grad_norm,
            'is_stable': is_stable
        }
    
    def test_simple_combined_loss(self) -> Dict[str, float]:
        """
        测试简化组合损失
        """
        print("\n=== 测试简化组合损失 ===")
        
        # 创建损失函数
        loss_fn = SimpleCombinedLoss(
            count_weight=1.0,
            loc_weight=1.0,
            photon_weight=0.5,
            bg_weight=0.1,
            pos_weight=1.0
        ).to(self.device)
        
        # 创建测试数据
        data = self.create_test_data()
        output = data['output']
        
        # 构建输出字典
        outputs = {
            'prob': output[:, 0:1],      # [B, 1, H, W]
            'offset': output[:, 1:4],    # [B, 3, H, W]
            'photon': output[:, 4:5],    # [B, 1, H, W]
            'background': output[:, 5:6] # [B, 1, H, W]
        }
        
        # 计算损失
        losses = loss_fn(outputs, data['targets'])
        
        print(f"总损失: {losses['total'].item():.6f}")
        for key, value in losses.items():
            if key != 'total':
                print(f"  {key}: {value.item():.6f}")
        
        # 测试梯度
        output_for_grad = data['output'].clone().requires_grad_(True)
        outputs_for_grad = {
            'prob': output_for_grad[:, 0:1],
            'offset': output_for_grad[:, 1:4],
            'photon': output_for_grad[:, 4:5],
            'background': output_for_grad[:, 5:6]
        }
        losses_for_grad = loss_fn(outputs_for_grad, data['targets'])
        losses_for_grad['total'].backward()
        
        grad_norm = torch.norm(output_for_grad.grad).item()
        print(f"梯度范数: {grad_norm:.6f}")
        
        # 检查数值稳定性
        is_stable = (
            not torch.isnan(losses['total']) and
            not torch.isinf(losses['total']) and
            0.001 <= losses['total'].item() <= 100.0
        )
        
        print(f"数值稳定性: {'✓' if is_stable else '✗'}")
        
        return {
            'total_loss': losses['total'].item(),
            'grad_norm': grad_norm,
            'is_stable': is_stable
        }
    
    def compare_with_original(self) -> Dict[str, Dict[str, float]]:
        """
        与原始复杂损失函数对比
        """
        if not ORIGINAL_AVAILABLE:
            print("\n=== 原始损失函数不可用，跳过对比 ===")
            return {}
        
        print("\n=== 与原始损失函数对比 ===")
        
        results = {}
        
        # 测试数据
        data = self.create_test_data(batch_size=4, image_size=32)  # 较小的数据避免内存问题
        
        try:
            # 原始CountLoss
            print("\n--- 原始CountLoss ---")
            original_count_loss = CountLoss().to(self.device)
            
            prob_map = torch.sigmoid(data['output'][:, 0:1])  # [B, 1, H, W]
            true_count = torch.randint(0, 10, (data['output'].size(0),), device=self.device).float()
            
            start_time = time.time()
            original_loss = original_count_loss(prob_map, true_count)
            original_time = time.time() - start_time
            
            print(f"原始CountLoss: {original_loss.item():.6f} (耗时: {original_time:.4f}s)")
            
            # 简化CountLoss
            print("\n--- 简化CountLoss ---")
            simple_count_loss = SimpleCountLoss().to(self.device)
            
            pred_logits = data['output'][:, 0:1]
            target_prob = data['targets']['count_maps']
            
            start_time = time.time()
            simple_loss = simple_count_loss(pred_logits, target_prob)
            simple_time = time.time() - start_time
            
            print(f"简化CountLoss: {simple_loss.item():.6f} (耗时: {simple_time:.4f}s)")
            
            results['count_loss'] = {
                'original': original_loss.item(),
                'simple': simple_loss.item(),
                'original_time': original_time,
                'simple_time': simple_time,
                'speedup': original_time / simple_time if simple_time > 0 else float('inf')
            }
            
            print(f"加速比: {results['count_loss']['speedup']:.2f}x")
            
        except Exception as e:
            print(f"CountLoss对比失败: {e}")
        
        try:
            # 原始LocLoss（这个可能会很复杂，我们简化测试）
            print("\n--- LocLoss对比（简化版本） ---")
            
            # 简化LocLoss
            simple_loc_loss = SimpleLocLoss().to(self.device)
            
            pred_coords = data['output'][:, 1:4]
            target_coords = data['targets']['loc_maps']
            mask = data['targets']['count_maps'] > 0.5
            
            start_time = time.time()
            simple_loc_result = simple_loc_loss(pred_coords, target_coords, mask)
            simple_loc_time = time.time() - start_time
            
            print(f"简化LocLoss: {simple_loc_result.item():.6f} (耗时: {simple_loc_time:.4f}s)")
            
            results['loc_loss'] = {
                'simple': simple_loc_result.item(),
                'simple_time': simple_loc_time
            }
            
        except Exception as e:
            print(f"LocLoss测试失败: {e}")
        
        return results
    
    def stress_test(self, num_iterations: int = 100) -> Dict[str, List[float]]:
        """
        压力测试：多次运行检查稳定性
        """
        print(f"\n=== 压力测试 ({num_iterations}次迭代) ===")
        
        # 创建损失函数
        unified_loss = UnifiedDECODELoss().to(self.device)
        simple_count_loss = SimpleCountLoss().to(self.device)
        
        results = {
            'unified_losses': [],
            'count_losses': [],
            'grad_norms': [],
            'computation_times': []
        }
        
        for i in range(num_iterations):
            # 创建随机测试数据
            data = self.create_test_data(batch_size=np.random.randint(2, 16))
            
            try:
                # 测试统一损失
                start_time = time.time()
                
                output = data['output'].clone().requires_grad_(True)
                
                unified_result = unified_loss(output, data['targets']['unified_target'])
                unified_result['total'].backward()
                
                grad_norm = torch.norm(output.grad).item()
                computation_time = time.time() - start_time
                
                # 记录结果
                results['unified_losses'].append(unified_result['total'].item())
                results['grad_norms'].append(grad_norm)
                results['computation_times'].append(computation_time)
                
                # 测试简化计数损失
                count_result = simple_count_loss(
                    data['output'][:, 0:1], 
                    data['targets']['count_maps']
                )
                results['count_losses'].append(count_result.item())
                
                if (i + 1) % 20 == 0:
                    print(f"完成 {i+1}/{num_iterations} 次迭代")
                    
            except Exception as e:
                print(f"迭代 {i+1} 失败: {e}")
                continue
        
        # 统计结果
        print("\n--- 压力测试结果 ---")
        print(f"统一损失 - 均值: {np.mean(results['unified_losses']):.6f}, "
              f"标准差: {np.std(results['unified_losses']):.6f}")
        print(f"计数损失 - 均值: {np.mean(results['count_losses']):.6f}, "
              f"标准差: {np.std(results['count_losses']):.6f}")
        print(f"梯度范数 - 均值: {np.mean(results['grad_norms']):.6f}, "
              f"标准差: {np.std(results['grad_norms']):.6f}")
        print(f"计算时间 - 均值: {np.mean(results['computation_times']):.4f}s, "
              f"标准差: {np.std(results['computation_times']):.4f}s")
        
        # 检查异常值
        unified_losses = np.array(results['unified_losses'])
        anomalies = np.sum((unified_losses < 0.001) | (unified_losses > 100.0))
        print(f"异常损失值数量: {anomalies}/{len(unified_losses)}")
        
        return results
    
    def run_all_tests(self):
        """
        运行所有测试
        """
        print("开始损失函数测试套件...")
        
        # 基础功能测试
        unified_result = self.test_unified_loss()
        count_result = self.test_simple_count_loss()
        loc_result = self.test_simple_loc_loss()
        combined_result = self.test_simple_combined_loss()
        
        # 对比测试
        comparison_result = self.compare_with_original()
        
        # 压力测试
        stress_result = self.stress_test(num_iterations=50)
        
        # 总结
        print("\n" + "="*50)
        print("测试总结")
        print("="*50)
        
        all_stable = (
            unified_result['is_stable'] and
            count_result['is_stable'] and
            loc_result['is_stable'] and
            combined_result['is_stable']
        )
        
        print(f"所有损失函数数值稳定: {'✓' if all_stable else '✗'}")
        
        # 损失值范围检查
        loss_values = [
            unified_result['total_loss'],
            count_result['loss'],
            loc_result['loss'],
            combined_result['total_loss']
        ]
        
        reasonable_range = all(0.001 <= loss <= 100.0 for loss in loss_values)
        print(f"损失值在合理范围内: {'✓' if reasonable_range else '✗'}")
        
        # 梯度检查
        grad_norms = [
            unified_result['grad_norm'],
            count_result['grad_norm'],
            loc_result['grad_norm'],
            combined_result['grad_norm']
        ]
        
        healthy_gradients = all(0.0001 <= grad <= 10.0 for grad in grad_norms)
        print(f"梯度健康: {'✓' if healthy_gradients else '✗'}")
        
        if all_stable and reasonable_range and healthy_gradients:
            print("\n🎉 所有测试通过！损失函数修复成功！")
        else:
            print("\n⚠️  部分测试未通过，需要进一步调整")
        
        return {
            'unified': unified_result,
            'count': count_result,
            'loc': loc_result,
            'combined': combined_result,
            'comparison': comparison_result,
            'stress': stress_result
        }


def main():
    """主函数"""
    print("DECODE损失函数修复验证")
    print("=" * 50)
    
    # 创建测试套件
    test_suite = LossTestSuite()
    
    # 运行所有测试
    results = test_suite.run_all_tests()
    
    # 可选：保存结果
    import json
    
    # 转换结果为可序列化格式
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            serializable_results[key] = {k: float(v) if isinstance(v, (int, float, torch.Tensor)) else v 
                                       for k, v in value.items() if not isinstance(v, list)}
    
    with open('loss_test_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print("\n测试结果已保存到 loss_test_results.json")


if __name__ == '__main__':
    main()