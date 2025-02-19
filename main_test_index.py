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

import numpy as np

def generate_random_state_on_all_surfaces(obstacles, quad_radius, min_distance=0):
    """
    在所有障碍物的表面采样四旋翼的状态，使其与障碍物表面接触，同时确保不与其他障碍物碰撞。

    :param obstacles: 障碍物列表，每个障碍物定义为 (位置, 半径)
    :param quad_radius: 四旋翼的半径
    :param min_distance: 最小安全距离（默认为0，表示刚好接触）
    :return: 合法的随机状态向量
    """
    while True:
        # 随机选择一个障碍物
        ob_idx = np.random.choice(len(obstacles))
        ob_pos, ob_radius = obstacles[ob_idx]

        # 在障碍物的表面生成多个随机点
        theta = np.random.uniform(0, 2 * np.pi)  # 随机选择极角
        phi = np.random.uniform(0, np.pi)  # 随机选择方位角

        # 转换为笛卡尔坐标
        x = ob_pos[0] + (ob_radius + quad_radius) * np.sin(phi) * np.cos(theta)
        y = ob_pos[1] + (ob_radius + quad_radius) * np.sin(phi) * np.sin(theta)
        z = ob_pos[2] + (ob_radius + quad_radius) * np.cos(phi)

        # 检查与其他障碍物的碰撞
        valid = True
        for other_ob_pos, other_ob_radius in obstacles:
            if np.array_equal(ob_pos, other_ob_pos):  # 跳过自己
                continue
            distance = np.linalg.norm(np.array([x, y, z]) - other_ob_pos)
            if distance <= (quad_radius + other_ob_radius + min_distance):
                valid = False
                break
        
        # 如果该位置没有碰撞，退出循环并返回该状态
        if valid:
            quat = generate_random_quaternion()  # 生成随机四元数
            vel = np.random.uniform(-2, 2, 3)  # 随机生成速度
            a_rate = np.random.uniform(-np.pi, np.pi, 3)  # 随机生成角速度
            state = np.concatenate([np.array([x, y, z]), quat, vel, a_rate])  # 组合成状态向量
            return state



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

def main():
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
    total_cpu_usage = 0  # 用于累计总的 CPU 使用率变化
    total_psutil_calls = 0  # 用于记录psutil.cpu_percent()的调用次数

    for sample_idx in range(num_samples):
        print(f"\n=== 安全测试样本 {sample_idx + 1}/{num_samples} ===")

        # 生成去除障碍物的空间的随机状态
        x0 = generate_random_state_lhs(obstacles, quad_radius, min_distance=0)
        # 生成在所有障碍物表面上的随机状态
        #x0 = generate_random_state_on_all_surfaces(obstacles, quad_radius, min_distance=0.05)

        quad = Quadrotor()
        quad.pos = x0[0:3]
        quad.quat = x0[3:7]
        quad.vel = x0[7:10]
        quad.a_rate = x0[10:13]

        controller = Controller_nn(quad, obstacles=obstacles, n_nodes=N, use_nn=False, 
                                   model_path='SANMPC_Neural_Model/neural_value_function_model.pth',
                                   use_sensitivity=False, model_path_sensitivity='SANMPC_Neural_Model/neural_sensitivity_model.pth',
                                   dt=DT, gamma=0.3)

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


        cpu_before = psutil.cpu_percent(interval=0.01)  # 获取前的 CPU 占用率
        start_time = time.time()  # 记录开始时间

# 获取CBF-MPC,SANMPC控制输入
        control_input, cost_over_30, cost_over_last_25, state_step_5 = controller.compute_control_signal(x_ref)
        if cost_over_last_25 is None:
            print(f"样本 {sample_idx + 1}: 跳过该样本。")
            continue

# 获取AMPC控制输入
        #control_input = ampc_controller(current_state)

        end_time = time.time()  # 记录结束时间
        cpu_after = psutil.cpu_percent(interval=0.01)  # 获取后的 CPU 占用率


        # 计算本次执行时间和 CPU 使用情况变化
        exec_time = end_time - start_time
        cpu_usage = cpu_after - cpu_before

        # 累计总的执行时间和 CPU 使用情况
        if cpu_usage >= 0:
            # 累计总的执行时间和 CPU 使用情况
            total_exec_time += exec_time
            total_cpu_usage += cpu_usage
            total_psutil_calls += 1  # 记录psutil调用次数


        # 打印当前状态和新的状态
        print(f"样本 {sample_idx + 1}:")
        print("当前状态 x:")
        print(f"  位置: {quad.pos}")
        print(f"  四元数: {quad.quat}")
        print(f"  速度: {quad.vel}")
        print(f"  角速度: {quad.a_rate}")

        # 施加控制输入并更新状态
        quad.update(control_input, dt=DT)

        # 获取新的状态
        new_state = np.concatenate([
            quad.pos,
            quad.quat,
            quad.vel,
            quad.a_rate
        ])

        print("-" * 50)

        print("新的状态 x' (更新后的状态):")
        print(f"  新位置: {quad.pos}")
        print(f"  新四元数: {quad.quat}")
        print(f"  新速度: {quad.vel}")
        print(f"  新角速度: {quad.a_rate}")

        print("-" * 50)

        # 检查是否发生碰撞
        if check_collision(quad.pos, obstacles, quad_radius):
            collision_count += 1  # 增加碰撞计数
            print(f"碰撞检测: 样本 {sample_idx + 1} 发生了碰撞！")

        # 保存样本数据
        sample_data = {
            'state': current_state.copy(),
            'input': control_input.copy(),
            'new_state': new_state.copy()  
        }

        current_file_data.append(sample_data)
        saved_samples_count += 1  

        print(f"当前成功优化的数量: {saved_samples_count} / {sample_idx + 1}")


    
    if current_file_data:
        file_name = f"{FILE_NAME_BASE}_{file_index}.txt"
        with open(file_name, 'wb') as f:
            pickle.dump(current_file_data, f)
        print(f"已保存 {len(current_file_data)} 条样本到 {file_name}")

    # 计算碰撞率
    collision_rate = collision_count / saved_samples_count
    print(f"\n所有数据已保存。总共成功优化的样本数量: {saved_samples_count} / {num_samples}")
    print(f"总碰撞次数: {collision_count}, 碰撞率: {collision_rate * 100:.2f}%")

    # 计算并输出平均求解时间和平均 CPU 占用变化
    avg_exec_time = total_exec_time / num_samples
    avg_cpu_usage = total_cpu_usage / total_psutil_calls
    print(f"\n平均控制器计算时间: {avg_exec_time:.4f}秒")
    print(f"平均CPU占用变化: {avg_cpu_usage:.2f}%")

if __name__ == "__main__":
    main()
