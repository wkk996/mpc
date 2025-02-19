# data_generator.py

import numpy as np
import pickle
from quadrotor import Quadrotor
from controller import Controller
from controller2 import Controller2
from parameters import get_reference_fixed, FILE_NAME_BASE, N, DT
from utils import unit_quat
from pyDOE import lhs  # 确保已安装 pyDOE 库: pip install pyDOE

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
    num_samples = 100000  

    file_index = 1
    current_file_data = []
    saved_samples_count = 0

    for sample_idx in range(num_samples):
        print(f"\n=== 生成样本 {sample_idx + 1}/{num_samples} ===")
        
        x0 = generate_random_state_lhs(obstacles, quad_radius, min_distance=0.5)

        quad = Quadrotor()
        quad.pos = x0[0:3]
        quad.quat = x0[3:7]
        quad.vel = x0[7:10]
        quad.a_rate = x0[10:13]

        controller = Controller(
            quad=quad, 
            obstacles=obstacles, 
            n_nodes=N, 
            dt=DT, 
            use_rbf=False, 
            gamma=0.3,
            massx=1
        )

        controller2 = Controller2(
            quad=quad, 
            obstacles=obstacles, 
            n_nodes=N, 
            dt=DT, 
            use_rbf=False, 
            gamma=0.3,
            massx=0.99
        )

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


        control_input, cost_over_30, cost_over_last_25, state_step_5 = controller.compute_control_signal(x_ref)
        _, _, cost_25_change, _ = controller2.compute_control_signal(x_ref)


        if cost_over_last_25 is None:
            print(f"样本 {sample_idx + 1}: 优化失败，跳过该样本。")
            continue
        if cost_25_change is None:
            print(f"样本 {sample_idx + 1}: 优化失败，跳过该样本。")
            continue

        # 计算质量对 cost_over_last_25 的灵敏度
        delta_mass = -0.01  

        if cost_25_change is not None:
            mass_sensitivity = (cost_25_change - cost_over_last_25) / delta_mass
        else:
            mass_sensitivity = None  

        sample_data = {
            'state': current_state.copy(),
            'input': control_input.copy(),
            'cost_over_30': cost_over_30,
            'cost_over_last_25': cost_over_last_25,
            'cost_25_change':cost_25_change,
            'state_step_5': state_step_5,
            'mass_sensitivity': mass_sensitivity  
        }

        current_file_data.append(sample_data)
        saved_samples_count += 1  

        print(f"样本 {sample_idx + 1}:")
        print("当前状态 x:")
        print(f"  位置: {quad.pos}")
        print(f"  四元数: {quad.quat}")
        print(f"  速度: {quad.vel}")
        print(f"  角速度: {quad.a_rate}")
        print("控制输入 (推力):", control_input)
        print(f"成本函数值 (30步): {cost_over_30:.4f}")
        print(f"成本函数值 (后25步): {cost_over_last_25:.4f}")
        print(f"成本函数值 (改变质量后25步): {cost_25_change:.4f}")
        print(f"质量灵敏度: {mass_sensitivity}")
        print("-" * 50)

        print(f"当前成功优化的数量: {saved_samples_count} / {sample_idx + 1}")

        if len(current_file_data) >= 10000:
            file_name = f"{FILE_NAME_BASE}_{file_index}.txt"
            with open(file_name, 'wb') as f:
                pickle.dump(current_file_data, f)
            print(f"已保存 {len(current_file_data)} 条样本到 {file_name}")
            current_file_data = []
            file_index += 1
    
    if current_file_data:
        file_name = f"{FILE_NAME_BASE}_{file_index}.txt"
        with open(file_name, 'wb') as f:
            pickle.dump(current_file_data, f)
        print(f"已保存 {len(current_file_data)} 条样本到 {file_name}")

    print(f"\n所有数据已保存。总共成功优化的样本数量: {saved_samples_count} / {num_samples}")

if __name__ == "__main__":
    main()
