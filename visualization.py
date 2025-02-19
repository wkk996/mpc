from parameters import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def set_axes_equal(ax):
    '''
    设置3D坐标轴均等缩放。

    参数：
        ax (mpl_toolkits.mplot3d.Axes3D): 3D坐标轴对象。
    '''
    # 获取当前坐标轴的所有限
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # 计算各轴范围的中点和半径
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # 找到最大的范围
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    # 设置所有轴的范围为中点 ± plot_radius
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

# 加载路径数据
with open("results/data_MPC-DC_N=30.txt", 'rb') as file:
    data = pickle.load(file)
path0 = data['path']
ref = data['ref']

with open("results/data_MPC-CBF_N=30.txt", 'rb') as file:
    data = pickle.load(file)
path1 = data['path']


#with open("results/data_True.txt", 'rb') as file:
#    data = pickle.load(file)
#path2 = data['path']

# 定义障碍物，位置和半径
obstacles = [
    (np.array([4.0, 4.0, 1.0]), 2.0), 
     (np.array([-4.0, -4.0, 1.0]), 2.0),  
    (np.array([4.0, -4.0, 1.0]), 2.0),
    (np.array([-4.0, 4.0, 1.0]), 2.0),
]

# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# 绘制障碍物为球体
for obstacle in obstacles:
    center, radius = obstacle
    # 创建球体的参数
    u, v = np.linspace(0, 2 * np.pi, 50), np.linspace(0, np.pi, 50)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    # 绘制球体表面
    ax.plot_surface(x, y, z, color='gray', alpha=0.8, linewidth=0)

# 绘制参考路径
ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], label="reference", color="blue")
#ax.plot(path0[:, 0], path0[:, 1], path0[:, 2], label="MPC-DC", color="red")
ax.plot(path1[:, 0], path1[:, 1], path1[:, 2], label="MPC-CBF", color="orange")

# 设置轴标签（可选）
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


# 设置均等缩放
set_axes_equal(ax)
# 设置3D坐标轴的比例为1:1:1
ax.set_box_aspect([1,1,1])  # 确保比例尺相同

# 设置图例
ax.legend()

# 显示图形
plt.show()
