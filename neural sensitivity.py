import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 数据加载与预处理（包含过滤逻辑）
class QuadDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        # 数据过滤与特征提取
        states = []
        sensitivity = []
        for sample in raw_data:
            mass_sensitivity = sample['mass_sensitivity']
            if mass_sensitivity <= 1000 and mass_sensitivity >= -1000:  # 过滤高成本样本
                states.append(sample['state_step_5'])
                sensitivity.append(mass_sensitivity)

        # 转换为numpy数组
        self.states = np.array(states)
        self.sensitivity = np.array(sensitivity)

        # 数据统计信息
        print(f"\n原始样本数: {len(raw_data)}")
        print(f"过滤后样本数: {len(self.states)}")
        print(f"过滤比例: {(len(raw_data)-len(self.states))/len(raw_data):.2%}")

        # 使用StandardScaler进行标准化
        self.state_scaler = StandardScaler()
        self.sensitivity_scaler = StandardScaler()

        # 标准化states和sensitivity
        self.states = self.state_scaler.fit_transform(self.states)
        self.sensitivity = self.sensitivity_scaler.fit_transform(self.sensitivity.reshape(-1, 1)).flatten()

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.states[idx]),
            torch.FloatTensor([self.sensitivity[idx]])
        )

class SensitivityPredictor(nn.Module):
    """定义神经网络模型。"""
    def __init__(self, input_dim):
        super(SensitivityPredictor, self).__init__()
        self.fc1 = nn.Linear(13, 32)  # 13输入特征, 32个神经元
        self.fc2 = nn.Linear(32, 32)  # 32个神经元
        self.fc3 = nn.Linear(32, 32)  # 32个神经元
        self.fc4 = nn.Linear(32, 1)   # 输出1个值

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # 第一层，tanh激活
        x = torch.tanh(self.fc2(x))  # 第二层，tanh激活
        x = torch.tanh(self.fc3(x))  # 第三层，tanh激活
        x = self.fc4(x)              # 输出层，无激活
        return x


# 3. 训练流程
def main():
    # 加载并预处理数据
    dataset = QuadDataset("results/merged_data_mass.txt")
    
    # 划分训练测试集
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # 创建数据加载器
    batch_size = 200
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 初始化模型
    model = SensitivityPredictor(13).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练参数
    num_epochs = 20
    train_losses = []

    # 4. 训练循环
    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for states, sensitivity in train_loader:
            states = states.to(device)
            sensitivity = sensitivity.to(device)
            
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, sensitivity)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * states.size(0)
        
        avg_loss = epoch_loss / len(train_dataset)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # 5. 保存模型和标准化器
    torch.save(model.state_dict(), 'neural_sensitivity_model.pth')
    torch.save({
        'state_scaler': dataset.state_scaler,
        'sensitivity_scaler': dataset.sensitivity_scaler
    }, 'scalers_sensitivity.pth')

    # 6. 评估模型
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for states, sensitivity in test_loader:
            states = states.to(device)
            sensitivity = sensitivity.to(device)
            
            outputs = model(states)
            test_loss += criterion(outputs, sensitivity).item() * states.size(0)
    
    avg_test_loss = test_loss / len(test_dataset)
    print(f"\nTest MSE: {avg_test_loss:.4f}")
    
    # 7. 可视化结果
    plt.figure(figsize=(12, 5))
    
    # 训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    
    # 预测结果散点图
    plt.subplot(1, 2, 2)
    plt.scatter(sensitivity.cpu().numpy(), outputs.cpu().numpy(), alpha=0.3)
    plt.plot([min(sensitivity.cpu().numpy()), max(sensitivity.cpu().numpy())], 
             [min(sensitivity.cpu().numpy()), max(sensitivity.cpu().numpy())], 'r--')
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("Prediction vs Ground Truth")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
