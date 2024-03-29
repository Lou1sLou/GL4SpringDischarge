import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 读取CSV文件
data = pd.read_csv('./data/water flow 7.csv')

# 创建一个3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 提取数据集的x、y和z坐标
x = np.arange(data.shape[0])  # 使用np.arange以控制x轴长度
y = np.arange(data.shape[1])  # 使用np.arange以控制y轴长度
x, y = np.meshgrid(x, y)  # 创建网格坐标
z = data.to_numpy()  # 转为NumPy数组作为z坐标

# 设置不同列的颜色
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

for i in range(data.shape[1]):
    ax.bar3d(x.ravel(), y.ravel(), 0, 1, 1, z.ravel(), shade=True, color=colors[i])

# 设置坐标轴标签
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# 设置x和y轴范围以拉长x轴和缩短y轴
ax.set_xlim(0, data.shape[0])
ax.set_ylim(0, data.shape[1])

# 显示图形
plt.show()