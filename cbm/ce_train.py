import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn.functional as F
import random

# info_save_index = [0, 6, 12, 18, 24, 30]
info_save_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]    
# info_save_index=[0, 12, 24]

ce_data_dir="/home/junda/zihan/cbm/data/"

# 训练模型
num_epochs = 50

input_datas = torch.load(ce_data_dir+'ce_training_full.pth')

# if isinstance(input_datas, list):
#     # 将列表转换为 tensor，假设列表的每个元素形状一致
#     input_datas = torch.stack([torch.tensor(data) for data in input_datas])
# # input_datas = input_datas[:, :10, :]
merge_label=None
sample_size = len(input_datas[0])
labels = torch.load(ce_data_dir+'ce_training_label_full.pth')
# labels = labels[random_indices]
for i in range(len(input_datas)):
    random_indices = random.sample(range(len(input_datas[0])), sample_size)
    temp=[input_datas[i][x] for x in random_indices]
    input_datas[i]=temp #input_datas[i][random_indices]
    temp=[labels[x] for x in random_indices]
    if i==0:
        merge_label=temp #labels[random_indices]
    else:
        merge_label+=temp #labels[random_indices]
labels=merge_label

# 暂时取前10000个sample，训练比较快
# input_datas = [data[:500] for data in input_datas]
# labels = labels[:500]
# input_datas = [item for sublist in input_datas for item in sublist]

written_file="ce_result_merge.txt"


vit=False
one_item=False
# 定义模型
class CE(nn.Module):
    def __init__(self):
        super(CE, self).__init__()

        self.fc1 = nn.Linear(4096, 512)  # 假设每个 tensor 是 1024 维
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, len(labels[0]))  # 10个分类
        print(len(labels[0]))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = torch.sigmoid(self.fc3(x))
        x = self.fc3(x)
        return x


if vit==True:
    written_file="vit_"+written_file
    info_save_index=[0]

results={}
inference_result={}

############
# print(f"-----------------index {index}-----------------")

# 初始化模型、损失函数和优化器
model = CE().to('cuda:0')

# criterion = nn.BCELoss()  # 二元交叉熵损失
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

merge_data=None

if not vit:
    for i in range(len(info_save_index)):    
        if i==0:
            merge_data=input_datas[i]
            # merge_label=labels[:]
        else:
            merge_data+=input_datas[i]
            # merge_label+=labels[:]

input_data=merge_data
# labels=merge_label

# 将数据转换为 Tensor
if vit:
    data_tensor = torch.stack(vit_feature_list)
else:
    data_tensor = torch.stack(input_data)  # (N, 1024) 假设每个 tensor 是 1024 维
labels_tensor = torch.tensor(labels, device='cuda:0')

# 创建 TensorDataset
dataset = TensorDataset(data_tensor, labels_tensor)

# # 划分训练集和测试集
# train_size = int(0.9 * len(dataset))
# test_size = len(dataset) - train_size

# # 创建索引
# train_indices = list(range(train_size))
# test_indices = list(range(train_size, len(dataset)))

# # 使用 Subset 创建子集
# train_dataset = Subset(dataset, train_indices)
# test_dataset = Subset(dataset, test_indices)
# 划分训练集和测试集
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

# 随机化数据集并划分
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True) #64
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# vit_feature_list
all_losses=[]
all_f1=[]
all_accuracies = []
classifiy_res_index=[]
gt_res_index=[]
for epoch in range(num_epochs):
    classifiy_res_epoch=[]
    gt_res_epoch=[]
    
    model.train()  # 设置模型为训练模式
    for inputs, target in train_loader:
        inputs = inputs.float().to('cuda:0')
        if not one_item:
            target = target.float().to('cuda:0')
        else:
            target=target.unsqueeze(1).float()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
    
    # 测试模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        total_kl=[]
        for inputs, target in test_loader:
            inputs = inputs.float().to('cuda:0')
            # target = target.int()
            
            predicted = model(inputs)
            
            classifiy_res_epoch+=predicted
            if gt_res_index==[]:
                gt_res_epoch+=target
            
            for j in range(len(target)):
                if torch.all(target[j] == 0):
                    continue
                
                # 计算 softmax 和 log
                # softmax_pred = F.softmax(predicted[j], dim=0)
                # log_softmax_pred = softmax_pred.log()
                log_softmax_pred=torch.log_softmax(predicted[j], dim=0)
                target_float = target[j].float()
                
                
                kl_div = F.kl_div(log_softmax_pred, target_float, reduction='batchmean')
                
                # total_kl.append(F.kl_div(F.softmax(predicted[j]).log(), target[j].float()))
                
                # debug, 检查是否为 NaN
                if torch.isnan(kl_div):
                    print(f"NaN detected at index {j}:")
                    # print(f"Softmax result: {softmax_pred}")
                    print(f"Log Softmax result: {log_softmax_pred}")
                    print(f"Target (float): {target_float}")

                total_kl.append(kl_div)

    
    if one_item:
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
    else:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, kl_div: {sum(total_kl)/len(total_kl):.2f}, middle case: {sum(total_kl):.2f}, {len(total_kl):.2f}')
    # 保存当前epoch的loss和accuracy
    all_losses.append(loss.item())
    if one_item:
        all_accuracies.append(accuracy)
    else:
        all_f1.append(sum(total_kl)/len(total_kl))
    
    classifiy_res_index.append(classifiy_res_epoch)
    if gt_res_index==[]:
        gt_res_index=gt_res_epoch
    
    if (epoch+1)%2==0:
        torch.save(model.state_dict(), f'./result/ckpt/ce_model_merge_epoch{epoch}.pth')
    
    # break
    
# 将当前 index 的结果存入字典
if vit:
    results[0] = {
        'losses': all_losses,
        'accuracies': all_f1,
    }
        
    inference_result[0] = {
        'classifiy_res': classifiy_res_index,
        'gt': gt_res_index
    }
else:
    if one_item:
        results[0] = {
            'losses': all_losses,
            'accuracies': all_accuracies,
        }
        
        inference_result[0] = {
            'classifiy_res': classifiy_res_index,
            'gt': gt_res_index
        }
    else:
        results[0] = {
            'losses': all_losses,
            'accuracies': all_f1,
        }
        
        inference_result[0] = {
            'classifiy_res': classifiy_res_index,
            'gt': gt_res_index
        }
torch.save(model.state_dict(), f'./result/ce_model_merge.pth')
    
with open(written_file, "w") as f:
    if one_item:
        f.write("Index\tEpoch\tLoss\tAccuracy\n")
    else:
        f.write("Index\tEpoch\tLoss\tF1\n")
    for index, metrics in results.items():
        for epoch in range(num_epochs):
            f.write(f"{index}\t{epoch + 1}\t{metrics['losses'][epoch]:.4f}\t{metrics['accuracies'][epoch]:.2f}\n")

print(written_file[:-4]+".pth")
torch.save(inference_result, written_file[:-4]+".pth")

print("训练完成！")