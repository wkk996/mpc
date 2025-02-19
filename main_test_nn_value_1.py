import numpy as np
import pickle
from quadrotor import Quadrotor
from controller_nn import Controller_nn
from parameters import get_reference_fixed, FILE_NAME_BASE, N, DT
from utils import unit_quat
from pyDOE import lhs  # 确保已安装 pyDOE 库: pip install pyDOE
from AMPC_test import ampc_controller
import time
import psutil
import torch
import subprocess  # 用于获取nvidia-smi输出

class CostPredictor(torch.nn.Module):
    def __init__(self, input_dim):
        super(CostPredictor, self).__init__()
        self.fc1 = torch.nn.Linear(13, 32)  # 13输入特征, 32个神经元
        self.fc2 = torch.nn.Linear(32, 32)  # 32个神经元
        self.fc3 = torch.nn.Linear(32, 32)  # 32个神经元
        self.fc4 = torch.nn.Linear(32, 1)   # 输出1个值

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # 第一层，tanh激活
        x = torch.tanh(self.fc2(x))  # 第二层，tanh激活
        x = torch.tanh(self.fc3(x))  # 第三层，tanh激活
        x = self.fc4(x)              # 输出层，无激活
        return x


def load_model(model_path='SANMPC_Neural_Model/neural_value_function_model.pth', device='cpu'):
    model = CostPredictor(13)
    model.load_state_dict(torch.load(model_path, map_location=device))  # 加载到指定设备
    model.to(device)  # 确保模型也被移动到指定设备上
    model.eval()  # 设置为评估模式
    return model


def pytorch_nn(state_input, model_path='SANMPC_Neural_Model/neural_value_function_model.pth', cost_scaler=None, device='cpu'):
    model = load_model(model_path, device)
    scalers = torch.load('SANMPC_Neural_Model/scalers_value_fuction.pth', map_location=device)
    state_scaler = scalers['state_scaler']
    cost_scaler = scalers['cost_scaler']
    
    # 标准化输入数据
    state_input = state_input.reshape(1, -1) 
    state_input_scaled = state_scaler.transform(state_input)  # 标准化输入数据
    
    # 将标准化后的数据转换为Tensor，并将其移到相同的设备上（device）
    state_tensor = torch.FloatTensor(state_input_scaled).to(device)  # 将输入张量移到正确的设备
    
    with torch.no_grad():
        output = model(state_tensor)
    
    # 反标准化输出，确保是二维数组
    if cost_scaler:
        output_reshaped = output.cpu().numpy().reshape(-1, 1)  # 将输出移回CPU并转换为二维数组
        output = cost_scaler.inverse_transform(output_reshaped)
    
    return output


def generate_random_quaternion():
    """生成一个随机单位四元数"""
    rand = np.random.normal(0, 1, 4)
    return unit_quat(rand)

def generate_random_state_lhs(obstacles, quad_radius, min_distance=0):
    """
    使用拉丁超立方体采样生成随机的四旋翼状态，并确保其不与任何障碍物重叠。

    :param obstacles: 障碍物列表，每个障碍物定义为 (位置, 半径)
    :param quad_radius: 四旋翼的半径
    :param min_distance: 最小安全距离
    :return: 合法的随机状态向量
    """
    while True:
        lhs_sample = lhs(3, samples=1).flatten()  # 生成3维位置样本
        pos = lhs_sample * 10  # 将样本缩放到 [0, 10] 范围内

        quat = generate_random_quaternion()
        vel = np.random.uniform(-2, 2, 3)
        a_rate = np.random.uniform(-np.pi, np.pi, 3)

        # 检查与所有障碍物的距离
        valid = True
        for (ob_pos, ob_radius) in obstacles:
            distance = np.linalg.norm(pos - ob_pos)
            if distance <= (quad_radius + ob_radius + min_distance):
                valid = False
                break

        if valid:
            return np.concatenate([pos, quat, vel, a_rate])


def check_collision(pos, obstacles, quad_radius):
    """
    检查四旋翼当前位置是否与任何障碍物发生碰撞。

    :param pos: 当前四旋翼的位置
    :param obstacles: 障碍物列表，每个障碍物定义为 (位置, 半径)
    :param quad_radius: 四旋翼的半径
    :return: 如果发生碰撞返回 True,否则返回 False
    """
    for (ob_pos, ob_radius) in obstacles:
        distance = np.linalg.norm(pos - ob_pos)
        if distance < (quad_radius + ob_radius):
            return True
    return False

# 获取GPU内存占用和GPU利用率
def get_gpu_usage():
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated(0)  # 已分配的显存
        reserved_memory = torch.cuda.memory_reserved(0)  # 已预留的显存
        gpu_utilization = torch.cuda.get_device_properties(0).total_memory  # 总显存
        gpu_percent = torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100
        return allocated_memory, reserved_memory, gpu_utilization, gpu_percent
    else:
        print("No GPU available")
        return None, None, None, None

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择GPU或CPU

    obstacles = [
        (np.array([4.0, 4.0, 7.0]), 2.0), 
        (np.array([8.0, 8.0, 2.0]), 2.0), 
        (np.array([6.0, 8.0, 3.0]), 2.0), 
        (np.array([4.0, 4.0, 1.0]), 2.0), 
        (np.array([8.0, 2.0, 2.0]), 2.0), 
        (np.array([2.0, 8.0, 5.0]), 2.0), 
    ]
    
    target_point = np.array([10.0, 10.0, 10.0])
    tolerance = 0.00005  
    quad_radius = 0.5  
    num_samples = 200  

    file_index = 1
    current_file_data = []
    saved_samples_count = 0
    collision_count = 0  # 用于记录碰撞次数

    total_exec_time = 0  # 用于累计总的执行时间
    total_gpu_usage = 0  # 用于累计总的 GPU 使用率变化
    total_psutil_calls = 0  # 用于记录psutil.cpu_percent()的调用次数

    total_allocated_memory = 0  # 累计已分配显存
    total_reserved_memory = 0  # 累计已预留显存
    total_gpu_percent = 0  # 累计GPU使用百分比

    for sample_idx in range(num_samples):
        print(f"\n=== 安全测试样本 {sample_idx + 1}/{num_samples} ===")

        # 生成去除障碍物的空间的随机状态
        x0 = generate_random_state_lhs(obstacles, quad_radius, min_distance=0)

        quad = Quadrotor()
        quad.pos = x0[0:3]
        quad.quat = x0[3:7]
        quad.vel = x0[7:10]
        quad.a_rate = x0[10:13]

        current_state = np.concatenate([
            quad.pos,
            quad.quat,
            quad.vel,
            quad.a_rate
        ])

        x_ref = get_reference_fixed(
            time=0.0,
            x0=current_state,
            n=N,
            dt=DT,
            target=target_point
        )

        # 获取GPU前的状态
        gpu_before_alloc, gpu_before_reserved, gpu_total, gpu_before_percent = get_gpu_usage()
        start_time = time.time()  # 记录开始时间

        pytorch_output = pytorch_nn(current_state, model_path='SANMPC_Neural_Model/neural_value_function_model.pth', device=device)

        end_time = time.time()  # 记录结束时间
        gpu_after_alloc, gpu_after_reserved, gpu_total, gpu_after_percent = get_gpu_usage()  # 获取GPU后的状态

        # 计算本次执行时间和 GPU 使用情况变化
        exec_time = end_time - start_time
        gpu_usage = gpu_after_percent - gpu_before_percent

        # 累计总的执行时间和 GPU 使用情况
        if gpu_usage >= 0:
            total_exec_time += exec_time
            total_gpu_usage += gpu_usage
            total_psutil_calls += 1  # 记录psutil调用次数

            # 累计GPU内存
            total_allocated_memory += gpu_after_alloc
            total_reserved_memory += gpu_after_reserved
            total_gpu_percent += gpu_after_percent

        # 打印当前状态和新的状态
        print(f"样本 {sample_idx + 1}:")
        print("当前状态 x:")
        print(f"  位置: {quad.pos}")
        print(f"  四元数: {quad.quat}")
        print(f"  速度: {quad.vel}")
        print(f"  角速度: {quad.a_rate}")
        print(f"当前cost: {pytorch_output}")

        print("-" * 50)

        # 保存样本数据
        sample_data = {
            'state': current_state.copy(),
            'value': pytorch_output.copy(),
        }

        current_file_data.append(sample_data)
        saved_samples_count += 1  

    if current_file_data:
        file_name = f"{FILE_NAME_BASE}_{file_index}.txt"
        with open(file_name, 'wb') as f:
            pickle.dump(current_file_data, f)
        print(f"已保存 {len(current_file_data)} 条样本到 {file_name}")

    # 计算并输出平均求解时间和平均 GPU 占用情况
    avg_exec_time = total_exec_time / num_samples
    avg_gpu_usage = total_gpu_usage / total_psutil_calls
    avg_allocated_memory = total_allocated_memory / num_samples / 1024 ** 2  # 转换为MB
    avg_reserved_memory = total_reserved_memory / num_samples / 1024 ** 2  # 转换为MB
    avg_gpu_percent = total_gpu_percent / total_psutil_calls

    print(f"\n平均控制器计算时间: {avg_exec_time:.6f}秒")
    print(f"平均GPU占用变化: {avg_gpu_usage:.6f}%")
    print(f"平均已分配显存: {avg_allocated_memory:.2f} MB")
    print(f"平均已预留显存: {avg_reserved_memory:.2f} MB")
    print(f"平均GPU使用百分比: {avg_gpu_percent:.2f}%")
    print(f"使用的设备: {device}")

if __name__ == "__main__":
    main()
