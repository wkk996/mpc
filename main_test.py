from quadrotor import Quadrotor
from controller import Controller
from controller_nn import Controller_nn
from parameters import *
import numpy as np
import pickle
import time
import torch
# 定义障碍物

if __name__ == "__main__":
    obstacles = [
        (np.array([4.0, 4.0, 7.0]), 2.0), 
        (np.array([8.0, 8.0, 2.0]), 2.0), 
        (np.array([6.0, 8.0, 3.0]), 2.0), 
        (np.array([4.0, 4.0, 1.0]), 2.0), 
        (np.array([8.0, 2.0, 2.0]), 2.0), 
        (np.array([2.0, 8.0, 5.0]), 2.0),  

  ]

    quad = Quadrotor()
    #controller = Controller(quad, obstacles=obstacles, n_nodes=N, dt=DT, use_rbf=False, gamma=0.3,massx=1.2)
    controller = Controller_nn(quad, obstacles=obstacles, n_nodes=N,use_nn=True, 
                               model_path='SANMPC_Neural_Model/neural_value_function_model.pth' ,
                               use_sensitivity=True, model_path_sensitivity='SANMPC_Neural_Model/neural_sensitivity_model.pth', dt=DT, gamma=0.3)
    path = []
    ref = []
    times = []

    cur_time = 0
    total_time = 50
    iter = 0

    # 定义目标点和容差
    target_point = np.array([10.0, 10.0, 10.0])
    tolerance = 0.5  # 米

    while(total_time > cur_time):
        x0 = np.concatenate(quad.get_state())
        # x_ref = get_reference(cur_time, x0, N, DT)  # 轨迹跟踪
        x_ref = get_reference_fixed(cur_time, x0, N, DT)  # 避障导航
        ref.append(x_ref[:,1])
        start = time.time()
        thrust,cost,_,_ = controller.compute_control_signal(x_ref)
        times.append(time.time() - start)
        
        # 打印当前时间
        print(f"Time: {cur_time:.2f}s")

        # 打印13个状态
        pos, quat, vel, a_rate = quad.get_state()
        print("States:")
        print(f"  Position: {pos}")
        print(f"  Quaternion: {quat}")
        print(f"  Velocity: {vel}")
        print(f"  Angular Rate: {a_rate}")

        # 打印4个控制输入
        print("Control Inputs (Thrusts):", thrust)

        # 打印成本函数值
        if cost is not None:
            print(f"Cost Function Value: {cost:.4f}")
        else:
            print("Cost Function Value: N/A (Optimization Failed)")

        print("-" * 50)  # 分隔线，便于阅读

        # 更新无人机状态
        quad.update(thrust, dt=DT)
        path.append(quad.pos)
        cur_time += DT

        # 检测是否到达目标点
        distance_to_target = np.linalg.norm(quad.pos - target_point)
        if distance_to_target <= tolerance:
            print(f"目标点已到达 (距离: {distance_to_target:.4f} 米)，提前结束仿真。")
            print(f"到达时间: {cur_time:.2f} 秒")
            break

        iter += 1

    with open(FILE_NAME, 'wb') as file:
        path = np.array(path)
        ref = np.array(ref)
        times = np.array(times)
        print("Max processing time: {:.4f}s".format(times.max()))
        print("Min processing time: {:.4f}s".format(times.min()))
        print("Mean processing time: {:.4f}s".format(times.mean()))
        data = dict()
        data['path'] = path
        data['ref'] = ref
        data['times'] = times
        pickle.dump(data, file)
