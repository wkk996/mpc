import numpy as np
from quadrotor import Quadrotor
from controller_nn import Controller_nn
from parameters import get_reference_fixed, FILE_NAME_BASE, N, DT
from utils import unit_quat
import time
import psutil
import torch

class CostPredictor(torch.nn.Module):
    def __init__(self, input_dim):
        super(CostPredictor, self).__init__()
        self.fc1 = torch.nn.Linear(13, 32)  
        self.fc2 = torch.nn.Linear(32, 32)  
        self.fc3 = torch.nn.Linear(32, 32)  
        self.fc4 = torch.nn.Linear(32, 1)  

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  
        x = torch.tanh(self.fc2(x))  
        x = torch.tanh(self.fc3(x))  
        x = self.fc4(x)              
        return x
    

def load_model(model_path='SANMPC_Neural_Model/neural_value_function_model.pth', device='gpu'):
    model = CostPredictor(13)
    model.load_state_dict(torch.load(model_path, map_location=device)) 
    model.eval()  
    return model


def pytorch_nn(state_input, model_path='SANMPC_Neural_Model/neural_value_function_model.pth', cost_scaler=None, device='gpu'):
    model = load_model(model_path, device)
    scalers = torch.load('SANMPC_Neural_Model/scalers_value_fuction.pth', map_location=device)
    state_scaler = scalers['state_scaler']
    cost_scaler = scalers['cost_scaler']
    

    state_input = state_input.reshape(1, -1) 
    state_input_scaled = state_scaler.transform(state_input)  
    

    state_tensor = torch.FloatTensor(state_input_scaled).to(device)
    
    with torch.no_grad():
        output = model(state_tensor)
    
  
    if cost_scaler:
        output_reshaped = output.cpu().numpy().reshape(-1, 1) 
        output = cost_scaler.inverse_transform(output_reshaped)
    
    return output

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_point = np.array([10.0, 10.0, 10.0])
    tolerance = 0.00005  
    quad_radius = 0.5  
    num_samples = 20  

    file_index = 1
    current_file_data = []
    saved_samples_count = 0

    total_exec_time = 0 
    total_cpu_usage = 0  
    total_psutil_calls = 0

    for sample_idx in range(num_samples):
        x0 = np.random.normal(0, 1, 13)

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

        cpu_before = psutil.cpu_percent(interval=0.01)  
        start_time = time.time()  

        pytorch_output = pytorch_nn(current_state, model_path='SANMPC_Neural_Model/neural_value_function_model.pth', device=device)

        end_time = time.time()  
        cpu_after = psutil.cpu_percent(interval=0.01)  


        exec_time = end_time - start_time
        cpu_usage = cpu_after - cpu_before


        if cpu_usage >= 0:

            total_exec_time += exec_time
            total_cpu_usage += cpu_usage
            total_psutil_calls += 1  


        print("-" * 50)


        sample_data = {
            'state': current_state.copy(),
            'value': pytorch_output.copy(),
        }

        current_file_data.append(sample_data)
        saved_samples_count += 1  




    avg_exec_time = total_exec_time / num_samples
    avg_cpu_usage = total_cpu_usage / total_psutil_calls
    print(f"\n{avg_exec_time:.6f}")
    print(f"{avg_cpu_usage:.2f}%")
    print(f"{device}")

if __name__ == "__main__":
    main()
