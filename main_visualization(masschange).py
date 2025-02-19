import os
import casadi as ca
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D  # 用于自定义图例句柄

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

def plot_sphere(center, radius, color, ax):
    """
    绘制一个球体。

    参数：
        center (np.array): 球心位置，形状为 (3,)
        radius (float): 球体半径
        color (str): 球体颜色
        ax (mpl_toolkits.mplot3d.Axes3D): 3D坐标轴对象
    """
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=1.0, linewidth=0)

# ---------------------------
# 加载不同控制器的路径数据
# ---------------------------
with open("results_masschange/data_test_CBF-MPC_1.2.txt", 'rb') as file:
    data = pickle.load(file)
path0 = data['path']
ref = data['ref']

with open("results_masschange/data_test_Myopic MPC_1.2.txt", 'rb') as file:
    data = pickle.load(file)
path1 = data['path']
ref = data['ref']

with open("results_masschange/data_test_AMPC_1.2.txt", 'rb') as file:
    data = pickle.load(file)
path2 = data['path']

# 从AMPC数据中获取 inside_obstacle_points（如果存在）
inside_obstacle_points = data.get('inside_obstacle_points', [])

with open("results_masschange/data_test_SANMPC_1.2.txt", 'rb') as file:
    data = pickle.load(file)
path3 = data['path']

with open("results_masschange/data_test_Neural MPC_1.2.txt", 'rb') as file:
    data = pickle.load(file)
path4 = data['path']




# ---------------------------
# 定义障碍物，位置和半径
# ---------------------------
obstacles = [
    (np.array([4.0, 4.0, 7.0]), 2.0), 
    (np.array([8.0, 8.0, 2.0]), 2.0), 
    (np.array([6.0, 8.0, 3.0]), 2.0), 
    (np.array([4.0, 4.0, 1.0]), 2.0), 
    (np.array([8.0, 2.0, 2.0]), 2.0), 
    (np.array([2.0, 8.0, 5.0]), 2.0),  
]

# ---------------------------
# 创建3D图形并绘制
# ---------------------------
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# 设置坐标轴背景颜色为白色
ax.set_facecolor('white')

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

# 绘制不同控制器生成的路径
ax.plot(path0[:, 0], path0[:, 1], path0[:, 2], label="CBF-MPC", color="blue")
ax.plot(path1[:, 0], path1[:, 1], path1[:, 2], label="Myopic MPC", color="red")
ax.plot(path2[:, 0], path2[:, 1], path2[:, 2], label="AMPC", color="green")
ax.plot(path3[:, 0], path3[:, 1], path3[:, 2], label="SANMPC", color="orange")
ax.plot(path4[:, 0], path4[:, 1], path4[:, 2], label="Neural MPC", color="purple")

# 获取路径的起点和终点（这里以 SANMPC 的路径为例）
start_path = path0[0]
end_path = path0[-1]

# 绘制起点和终点（使用小球表示）
plot_sphere(start_path, radius=0.5, color='red', ax=ax)     # 起点
plot_sphere(end_path, radius=0.5, color='blue', ax=ax)        # 终点

# 绘制 AMPC 路径的终点小球
plot_sphere(path2[-1], radius=0.5, color='green', ax=ax)

# 绘制 Neural MPC 路径的终点小球
plot_sphere(path4[-1], radius=0.5, color='purple', ax=ax)

# 如果 inside_obstacle_points 不为空，则绘制一个叉叉作为碰撞点标记（颜色 magenta，粗细可通过 linewidths 调整）
if inside_obstacle_points:
    first_point = inside_obstacle_points[0]
    ax.scatter(first_point[0], first_point[1], first_point[2],
               marker='x', color='red', s=80, linewidths=2)
else:
    print("AMPC数据中没有记录进入障碍物内部的状态点。")

    
# ---------------------------
# 添加图例
# ---------------------------
# 自定义图例句柄：起点、终点和碰撞点
# ---------------------------
# 添加图例
# ---------------------------
# 自定义图例句柄：起点、终点和碰撞点
start_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                        markersize=10, label='Start Point')
end_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                      markersize=10, label='Goal Point')
collision_marker = Line2D([0], [0],
                          marker='x',
                          linestyle='none',
                          color='red',
                          markersize=9,
                          markeredgewidth=2,  # 设置叉叉粗细
                          label='Collision Point')
line_CBF = Line2D([0], [0], color='blue', lw=2, label='CBF-MPC')
line_Myopic = Line2D([0], [0], color='red', lw=2, label='Myopic MPC')
line_AMPC = Line2D([0], [0], color='green', lw=2, label='AMPC')
line_SAN = Line2D([0], [0], color='orange', lw=2, label='SANMPC')
line_Neural = Line2D([0], [0], color='purple', lw=2, label='Neural MPC')
# 获取已有图例句柄（如各控制器路径）
legend_handles = [start_marker, end_marker, collision_marker,
                  line_CBF, line_Myopic, line_AMPC,line_SAN,line_Neural]

ax.legend(handles=legend_handles, prop={ 'size': 8})
# ---------------------------
# 设置坐标轴标签和缩放
# ---------------------------
ax.set_xlabel('X-position (m)')
ax.set_ylabel('Y-position (m)')
ax.set_zlabel('Z-position (m)')

# 设置坐标轴的范围为 -1 到 11
ax.set_xlim([-1, 11])
ax.set_ylim([-1, 11])
ax.set_zlim([-1, 11])



set_axes_equal(ax)
ax.set_box_aspect([1,1,1])  # 确保比例尺相同

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False


# 设置视角：elev 表示俯仰角，azim 表示方位角
ax.view_init(elev=30, azim=-45)  # 调整俯仰角为 30°，方位角为 45°
plt.savefig('quadrotor3D22.png', dpi=300, transparent=False, facecolor='white')
# 显示图形
plt.show()
