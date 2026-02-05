import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import random

# 1. 模型定义
class SimilarityNetwork(nn.Module):
    def __init__(self):
        super(SimilarityNetwork, self).__init__()
        self.fc1 = nn.Linear(4096 * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x1, x2):
        combined = torch.cat((x1, x2), dim=1)
        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # 输出相似性分数

# 2. 数据准备
# N = 1000  # 示例数据量
# input_l = torch.randn(N, 4096)
# input_L = torch.randn(N, 4096)
only_image=False

if not only_image:
    input_datas=torch.load('../data/ce_training_full_v1.pth')
else:
    input_datas=torch.load('../data/ce_training_image_full_v1.pth')


# print(input_datas[-1])
input_l=torch.stack(input_datas[0]).to(torch.float)
input_L=torch.stack(input_datas[-1]).to(torch.float)
# print(len(input_L), input_L[0].shape)


# 3. 分割数据集
input_l_train, input_l_val, input_L_train, input_L_val = train_test_split(
    input_l, input_L, test_size=0.2, random_state=42
)

# 生成两个索引列表
def generate_index_lists(size):
    index_list_1 = random.sample(range(size), size)
    index_list_2 = random.sample(range(size), size)
    
    # 确保不重复且对应元素不相等
    while any(index_list_1[i] == index_list_2[i] for i in range(size)) or \
          len(set(zip(index_list_1, index_list_2))) < size:
        index_list_2 = random.sample(range(size), size)

    return index_list_1, index_list_2

# 4. 创建数据加载器
train_dataset = TensorDataset(input_l_train, input_L_train)
# print(len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 5. 训练过程
def train_model(model, train_loader, criterion, optimizer, num_epochs=400):
    model.train()
    for epoch in range(num_epochs):
        for batch_input_l, batch_input_L in train_loader:
            # 正样本
            labels = torch.ones(batch_input_l.size(0))  # 正样本标签
            
            # 生成索引列表
            index_list_1, index_list_2 = generate_index_lists(batch_input_l.size(0))
            batch_input_l_neg = batch_input_l[index_list_1]  # 负样本
            batch_input_L_neg = batch_input_L[index_list_2]  # 负样本
            # print(index_list_1, index_list_2)
            # exit()
            # 合并正负样本
            combined_input_l = torch.cat((batch_input_l, batch_input_l_neg))
            combined_input_L = torch.cat((batch_input_L, batch_input_L_neg))
            combined_labels = torch.cat((labels, torch.zeros(batch_input_l.size(0)))).to("cuda:0")  # 负样本标签为0
            
            optimizer.zero_grad()
            # 计算相似性分数
            similarity_scores = model(combined_input_l, combined_input_L)
            loss = criterion(similarity_scores, combined_labels.view(-1, 1).float())  # 计算损失
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 6. 初始化模型、损失函数和优化器
model = SimilarityNetwork().to("cuda:0")
criterion = nn.BCEWithLogitsLoss().to("cuda:0")  # 二元交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
train_model(model, train_loader, criterion, optimizer, num_epochs=400)

# 7. 验证模型
def evaluate_model(model, val_loader):
    model.eval()
    with torch.no_grad():
        total, correct = 0, 0
        for batch_input_l, batch_input_L in val_loader:
            labels = torch.ones(batch_input_l.size(0))  # 正样本标签
            
            # 生成索引列表
            index_list_1, index_list_2 = generate_index_lists(batch_input_l.size(0))
            batch_input_l_neg = batch_input_l[index_list_1]
            batch_input_L_neg = batch_input_L[index_list_2]

            combined_input_l = torch.cat((batch_input_l, batch_input_l_neg))
            combined_input_L = torch.cat((batch_input_L, batch_input_L_neg))
            combined_labels = torch.cat((labels, torch.zeros(batch_input_l.size(0)))).to("cuda:0")  # 负样本标签为0
            
            similarity_scores = model(combined_input_l, combined_input_L)
            
            predicted = (similarity_scores > 0).float()
            total += combined_labels.size(0)
            correct += (predicted.view(-1) == combined_labels).sum().item()
    
    accuracy = correct / total
    print(f'Validation Accuracy: {accuracy:.4f}')

# 创建验证集 DataLoader
val_dataset = TensorDataset(input_l_val, input_L_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 验证模型
evaluate_model(model, val_loader)

if not only_image:
    torch.save(model.state_dict(), './similarity_network_full_v1.pth')
else:
    torch.save(model.state_dict(), './similarity_network_only_image_full_v1.pth')
