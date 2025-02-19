import torch
import torch.nn as nn
import numpy as np
import pickle
import time
from quadrotor import Quadrotor      # 无人机模型类
from parameters import *             # 包含仿真参数，如 DT、FILE_NAME 等

# ---------------------------
# 1. 定义神经网络结构
# ---------------------------
# 该网络结构与训练时保持一致，输入为13维状态，输出为4维控制输入
class AMPCPredictor(nn.Module):
    """定义神经网络模型。"""
    def __init__(self, input_dim):
        super(AMPCPredictor, self).__init__()
        self.fc1 = nn.Linear(13, 100)   # 13 输入特征 -> 第一隐藏层 100 个神经元
        self.fc2 = nn.Linear(100, 100)  # 第一隐藏层 100 个神经元 -> 第二隐藏层 100 个神经元
        self.fc3 = nn.Linear(100, 4)    # 第二隐藏层 100 个神经元 -> 输出层 4 个神经元

    def forward(self, x):
        x = nn.functional.leaky_relu(self.fc1(x), negative_slope=0.01)  # 第一隐藏层，泄漏的ReLU激活
        x = nn.functional.leaky_relu(self.fc2(x), negative_slope=0.01)  # 第二隐藏层，泄漏的ReLU激活
        x = self.fc3(x)  # 输出层，无激活函数
        return x

# ---------------------------
# 2. 加载模型与标准化器
# ---------------------------
# 选择设备（GPU/CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的神经网络模型
model = AMPCPredictor(13).to(device)
model.load_state_dict(torch.load('AMPC/ampc_control.pth', map_location=device))
model.eval()  # 切换为评估模式

# 加载保存的标准化器（状态和控制输入的缩放器）
scaler_data = torch.load('AMPC/scalers_ampc.pth', map_location=device)
state_scaler = scaler_data['state_scaler_ampc']
cost_scaler = scaler_data['cost_scaler_ampc']

# ---------------------------
# 3. 定义基于 ampc 的控制器函数
# ---------------------------
def ampc_controller(state):
    """
    利用训练好的神经网络模型预测控制输入
    :param state: numpy数组，13维状态向量
    :return: numpy数组，4维控制输入
    """
    # 对状态进行标准化（注意：必须与训练时使用的 state_scaler 保持一致）
    state_scaled = state_scaler.transform(state.reshape(1, -1))
    # 转换为 tensor，并送入对应设备
    state_tensor = torch.FloatTensor(state_scaled).to(device)
    # 预测控制输入（注意：预测的是标准化后的控制输入）
    with torch.no_grad():
        control_scaled = model(state_tensor)
    control_scaled_np = control_scaled.cpu().numpy()
    # 反标准化，得到实际控制输入
    control = cost_scaler.inverse_transform(control_scaled_np)
    return control.flatten()  # 返回一维数组，共 4 个控制输入

# ---------------------------
# 4. 仿真主程序（使用 ampc 控制器）
# ---------------------------
if __name__ == "__main__":
    # 定义障碍物（障碍物以球体形式描述，格式为 (中心坐标, 半径)）
    obstacles = [
        (np.array([4.0, 4.0, 7.0]), 2.0), 
        (np.array([8.0, 8.0, 2.0]), 2.0), 
        (np.array([6.0, 8.0, 3.0]), 2.0), 
        (np.array([4.0, 4.0, 1.0]), 2.0), 
        (np.array([8.0, 2.0, 2.0]), 2.0), 
        (np.array([2.0, 8.0, 5.0]), 2.0)
    ]
    
    # 初始化无人机模型
    quad = Quadrotor()

    # 用于存储轨迹、计算控制的时间以及记录进入障碍物内部的状态点
    path = []
    times = []
    inside_obstacle_points = []   # 用于记录处于障碍物内部的状态点坐标

    cur_time = 0
    total_time = 2.3  # 仿真总时长（秒）

    # 定义目标点及容差（用于判断是否提前结束仿真）
    target_point = np.array([10.0, 10.0, 10.0])
    tolerance = 0.5  # 米

    while cur_time < total_time:
        # 获取当前无人机状态
        # 假设 quad.get_state() 返回 (position, quaternion, velocity, angular_rate)
        # 分别为 (3,)、(4,)、(3,)、(3,) 维向量，总共 13 维
        state = np.concatenate(quad.get_state())
        
        # 记录计算控制输入所需时间
        start = time.time()
        # 利用 ampc 控制器获得 4 维控制输入
        thrust = ampc_controller(state)
        times.append(time.time() - start)
        
        # 打印当前仿真信息
        print(f"Time: {cur_time:.2f}s")
        pos, quat, vel, a_rate = quad.get_state()
        print("States:")
        print(f"  Position: {pos}")
        print(f"  Quaternion: {quat}")
        print(f"  Velocity: {vel}")
        print(f"  Angular Rate: {a_rate}")
        print("Control Inputs (Thrusts):", thrust)
        print("-" * 50)
        
        # 更新无人机状态（使用相同的系统方程）
        quad.update(thrust, dt=DT)
        path.append(quad.pos)
        
        # 检查当前状态是否位于任一障碍物的球体内
        for center, radius in obstacles:
            if np.linalg.norm(quad.pos - center) <= radius+0.5:
                inside_obstacle_points.append(quad.pos.copy())
                break  # 如果进入任意一个障碍物，则无需继续检查其它障碍物

        cur_time += DT
        
        # 检查是否到达目标点
        distance_to_target = np.linalg.norm(quad.pos - target_point)
        if distance_to_target <= tolerance:
            print(f"目标点已到达 (距离: {distance_to_target:.4f} 米)，提前结束仿真。")
            print(f"到达时间: {cur_time:.2f} 秒")
            break

    # 在仿真结束后打印所有处于障碍物内部的状态点坐标
    if inside_obstacle_points:
        print("\n以下状态点位于障碍物球体内:")
        for point in inside_obstacle_points:
            print(point)
    else:
        print("\n无人机在仿真过程中未进入任何障碍物。")
    
    # 保存仿真数据，包括路径、计算时间以及进入障碍物内部的状态点
    with open(FILE_NAME, 'wb') as file:
        path_array = np.array(path)
        times_array = np.array(times)
        data = {
            'path': path_array,
            'times': times_array,
            'inside_obstacle_points': inside_obstacle_points  # 保存障碍物内的状态点坐标
        }
        print("Max processing time: {:.4f}s".format(times_array.max()))
        print("Min processing time: {:.4f}s".format(times_array.min()))
        print("Mean processing time: {:.4f}s".format(times_array.mean()))
        pickle.dump(data, file)
