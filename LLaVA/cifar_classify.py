import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# 定义模型
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(4096, 512)  # 假设每个 tensor 是 1024 维
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)  # 10个分类

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

info_save_index = [0, 6, 12, 18, 24, 30]

# 训练模型
num_epochs = 10

for i, index in enumerate(info_save_index):
    print(f"-----------------index {index}-----------------")
    
    # 初始化模型、损失函数和优化器
    model = SimpleClassifier().to('cuda:0')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 加载标签和输入数据
    labels = torch.load('label_list_cifar100.pth')
    input_data = torch.load('info_probe_list_cifar100.pth')[i]
    
    # 将数据转换为 Tensor
    data_tensor = torch.stack(input_data)  # (N, 1024) 假设每个 tensor 是 1024 维
    labels_tensor = torch.tensor(labels, device='cuda:0')

    # 创建 TensorDataset
    dataset = TensorDataset(data_tensor, labels_tensor)

    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        for inputs, target in train_loader:
            inputs = inputs.float().to('cuda:0')
            target = target.to('cuda:0')
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
        
        # 计算训练集准确度
        model.eval()  # 设置模型为评估模式
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, target in train_loader:
                inputs = inputs.float().to('cuda:0')
                target = target.to('cuda:0')
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

print("训练完成！")