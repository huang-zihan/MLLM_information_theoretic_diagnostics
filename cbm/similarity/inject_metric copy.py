import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import random

# 1. 定义模型类
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

def info_nce_loss(model, data_l, data_L):
    N = data_l.size(0)
    # 计算相似性分数
    # 计算温度参数 tau
    # print(data_l)
    # print(data_L)
    tau = len(data_l[0])**0.5
    # print('tau', tau)
    
    all_NCE=[]
    # 计算正样本相似性
    for i in range(len(data_l)):
        pos_sim = torch.exp(model(data_l[i].unsqueeze(0), data_L[i].unsqueeze(0))/tau)
        neg_sim = sum([torch.exp(model(data_l[i].unsqueeze(0), data_L[j].unsqueeze(0))/tau) for j in range(len(data_l))])
        
        # print(pos_sim, neg_sim, torch.log(pos_sim/neg_sim), model(data_l[i].unsqueeze(0), data_L[i].unsqueeze(0)))
        # print([torch.exp(model(data_l[i].unsqueeze(0), data_L[j].unsqueeze(0))/tau) for j in range(len(data_l))])
        # exit()
        all_NCE.append(torch.log(pos_sim/neg_sim))
    
    # 计算损失
    loss = -sum(all_NCE)/len(all_NCE)  # InfoNCE损失
    return loss

# 2. 加载模型
model = SimilarityNetwork().to("cuda:0")
model.load_state_dict(torch.load('./similarity_network.pth'))
model.eval()  # 设置为评估模式

base_model = SimilarityNetwork().to("cuda:0")
base_model.load_state_dict(torch.load('./similarity_network_only_image.pth'))
base_model.eval()  # 设置为评估模式

# 确保数据是浮点型
input_datas = torch.load('../pope/pope_info_probe_list.pth')

print(input_datas[0])

base_input_datas = torch.load('../pope/image_pope_info_probe_list.pth')

input_l=torch.stack(input_datas[0]).to(torch.float).to("cuda:0")
input_L=torch.stack(input_datas[-1]).to(torch.float).to("cuda:0")

base_input_l=torch.stack(base_input_datas[0]).to(torch.float).to("cuda:0")
base_input_L=torch.stack(base_input_datas[-1]).to(torch.float).to("cuda:0")



# 3. 创建数据集和数据加载器
dataset = TensorDataset(input_l, input_L)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

base_dataset = TensorDataset(base_input_l, base_input_L)
base_dataloader = DataLoader(base_dataset, batch_size=16, shuffle=False)

# 4. 推理函数
def inference(model, dataloader):
    all_scores = []

    with torch.no_grad():
        for i, (batch_data1, batch_data2) in enumerate(dataloader):
            # score = model(batch_data1.to("cuda:0"), batch_data2.to("cuda:0"))
            print(i, "/", len(dataloader), end='\r')
            
            score = info_nce_loss(model, batch_data1.to("cuda:0"), batch_data2.to("cuda:0"))
            all_scores.append(score)
        
    return sum(all_scores)/len(all_scores)

    # return torch.cat(predictions), torch.cat(similarity_scores)

# 5. 执行推理
score = inference(model, dataloader)
base_score = inference(base_model, base_dataloader)
print("taskinfo and base score", score, base_score)
print("taskinfo and base metric", torch.log(torch.tensor(len(input_l), dtype=torch.float32))-score, torch.log(torch.tensor(len(input_l), dtype=torch.float32))-base_score)


print("mutual information", base_score-score)