"""这个文件用于生成各种通用的颜色
"""
import numpy as np
import matplotlib.pyplot as plt

deep_gray = (0.25, 0.25, 0.25)
mid_gray = (0.5, 0.5, 0.5)

def get_colors(n):
    colors = plt.cm.viridis(np.linspace(0, 1, n))  # 获取n种不同的颜色
    return colors[:, :3]
