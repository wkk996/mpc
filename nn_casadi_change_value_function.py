import torch
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler


# 假设已有的 PyTorch 神经网络类
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


# 加载训练好的 PyTorch 模型
def load_model(model_path='SANMPC_Neural_Model/neural_value_function_model.pth'):
    model = CostPredictor(13)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # 设置为评估模式
    return model


# 提取 PyTorch 模型的权重和偏置
def get_model_parameters(model):
    state_dict = model.state_dict()
    
    W1 = state_dict['fc1.weight'].detach().numpy()
    b1 = state_dict['fc1.bias'].detach().numpy()
    W2 = state_dict['fc2.weight'].detach().numpy()
    b2 = state_dict['fc2.bias'].detach().numpy()
    W3 = state_dict['fc3.weight'].detach().numpy()
    b3 = state_dict['fc3.bias'].detach().numpy()
    W4 = state_dict['fc4.weight'].detach().numpy()
    b4 = state_dict['fc4.bias'].detach().numpy()
    
    return W1, b1, W2, b2, W3, b3, W4, b4


# 将 PyTorch 网络转换为 CasADi 符号网络
# 将 PyTorch 网络转换为 CasADi 符号网络
def create_casadi_nn(input_dim, W1, b1, W2, b2, W3, b3, W4, b4, state_scaler, cost_scaler):
    # 创建 CasADi 符号变量（行向量）
    x = ca.MX.sym('x', 1, 13)  # 输入 x 是行向量 (1,13)
    
    # 输入标准化
    x_normalized = (x - state_scaler.mean_.reshape(1, -1)) / state_scaler.scale_.reshape(1, -1)
    # 使用reshape(1, -1)来确保mean_和scale_与x的维度一致，并且可以进行正确的广播

    # 转置权重矩阵以匹配 PyTorch 的线性层计算
    W1_T = W1.T
    W2_T = W2.T
    W3_T = W3.T
    W4_T = W4.T
    
    # 第一层：x @ W1.T + b1
    W1_casadi = ca.MX(W1_T)
    b1_casadi = ca.MX(b1).T
    layer1 = ca.tanh(ca.mtimes(x_normalized, W1_casadi) + b1_casadi)

    # 第二层：layer1 @ W2.T + b2
    W2_casadi = ca.MX(W2_T)
    b2_casadi = ca.MX(b2).T
    layer2 = ca.tanh(ca.mtimes(layer1, W2_casadi) + b2_casadi)

    # 第三层：layer2 @ W3.T + b3
    W3_casadi = ca.MX(W3_T)
    b3_casadi = ca.MX(b3).T
    layer3 = ca.tanh(ca.mtimes(layer2, W3_casadi) + b3_casadi)

    # 输出层：layer3 @ W4.T + b4
    W4_casadi = ca.MX(W4_T)
    b4_casadi = ca.MX(b4).T
    output = ca.mtimes(layer3, W4_casadi) + b4_casadi

    # 反标准化输出 (CasADi表达式)
    output_normalized = output * cost_scaler.scale_.reshape(1, -1) + cost_scaler.mean_.reshape(1, -1)
    # 同样，使用 reshape(1, -1) 确保 cost_scaler 的 mean_ 和 scale_ 能正确广播到输出

    return x, output_normalized



# 将 PyTorch 网络转化为 CasADi 函数
def pytorch_to_casadi(model_path='SANMPC_Neural_Model/neural_value_function_model.pth',state_scaler=None, cost_scaler=None):
    # 1. 加载训练好的模型
    model = load_model(model_path)
    scalers = torch.load('SANMPC_Neural_Model/scalers_value_fuction.pth')
    state_scaler = scalers['state_scaler']
    cost_scaler = scalers['cost_scaler']


    # 2. 提取模型的权重和偏置
    W1, b1, W2, b2, W3, b3, W4, b4 = get_model_parameters(model)
    
    # 3. 将 PyTorch 网络转化为 CasADi 网络，并进行反标准化
    x_casadi, output_casadi = create_casadi_nn(3, W1, b1, W2, b2, W3, b3, W4, b4, state_scaler,cost_scaler)
    
    # 4. 将 CasADi 网络包装为可调用的函数
    nn_function = ca.Function('nn_function', [x_casadi], [output_casadi])
    
    return nn_function



# TEST

# 调用PyTorch NN模型
def pytorch_nn(state_input, model_path='SANMPC_Neural_Model/neural_value_function_model.pth', cost_scaler=None):
    model = load_model(model_path)
    scalers = torch.load('SANMPC_Neural_Model/scalers_value_fuction.pth')
    state_scaler = scalers['state_scaler']
    cost_scaler = scalers['cost_scaler']
    # 标准化输入数据
    state_input = state_input.reshape(1, -1) 
    state_input_scaled = state_scaler.transform(state_input)  # 标准化输入数据
    
    # 将标准化后的数据转换为Tensor
    state_tensor = torch.FloatTensor(state_input_scaled)
    with torch.no_grad():
        output = model(state_tensor)
    
    # 反标准化输出，确保是二维数组
    if cost_scaler:
        output_reshaped = output.numpy().reshape(-1, 1)  # 转换为二维数组
        output = cost_scaler.inverse_transform(output_reshaped)
    
    return output



# 调用CasADi NN模型
# 调用CasADi NN模型
# 调用CasADi NN模型
def casadi_nn(state_input, model_path='SANMPC_Neural_Model/neural_value_function_model.pth', cost_scaler=None):
    nn_function = pytorch_to_casadi(model_path, cost_scaler)
    output_value = nn_function(state_input)  # 已经封装了反标准化
    return output_value



# 测试：分别调用PyTorch NN和CasADi NN模型并进行比较
def test_nn_models():
    model_path = 'SANMPC_Neural_Model/neural_value_function_model.pth'  # 请确保路径正确
    state_input =  np.array([
        0,   # 状态1
        0,   # 状态2
        0,  # 状态3
        1,   # 状态4
        0,   # 状态5
        0,   # 状态6
        0,  # 状态7
        0,   # 状态8
        0,   # 状态9
        0,  # 状态10
        0,   # 状态11
        0,   # 状态12
        0   # 状态13
    ], dtype=np.float32)



    # 调用PyTorch NN模型
    pytorch_output = pytorch_nn(state_input, model_path)
    # 调用CasADi NN模型
    #casadi_output = casadi_nn(state_input, model_path)




    print(f"PyTorch NN Prediction: {pytorch_output}")
    #print(f"CasADi NN Prediction: {casadi_output}")


if __name__ == "__main__":
    test_nn_models()