#!/usr/bin/env python3
"""
生成结果摘要图像
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# 图像文件和标题
images = [
    'psf_comparison.png', 
    'psf_comparison_five.png', 
    'zernike_coefficients.png', 
    'zernike_coefficients_random.png', 
    'zernike_phase_maps.png'
]

titles = [
    'PSF对比', 
    '5帧PSF对比', 
    'Zernike系数', 
    '随机发射体Zernike系数', 
    'Zernike相位图'
]

# 创建图像网格
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 加载并显示图像
for i, (img_path, title) in enumerate(zip(images, titles)):
    if i < 5:  # 只使用前5个图像
        row, col = i // 3, i % 3
        img = mpimg.imread(os.path.join('result', img_path))
        axes[row, col].imshow(img)
        axes[row, col].set_title(title)
        axes[row, col].axis('off')

# 隐藏未使用的子图
axes[1, 2].axis('off')

# 调整布局并保存
plt.tight_layout()
plt.savefig(os.path.join('result', 'summary.png'), dpi=300)
print('已生成结果摘要图像：result/summary.png')