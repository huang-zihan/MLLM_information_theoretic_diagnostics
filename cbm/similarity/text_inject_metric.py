import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import random

import matplotlib.pyplot as plt
import pandas as pd

# 1. 定义模型类
class SimilarityNetwork(nn.Module):
    def __init__(self):
        super(SimilarityNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

    def forward(self, feature1, feature2):
        feature1 = self.encoder(feature1)
        feature2 = self.encoder(feature2)
        feature1 = torch.tanh(feature1)
        feature2 = torch.tanh(feature2)
        
        logits_feature1 = feature1 @ feature2.t()/feature1[0].shape[-1]
        logits_feature2 = logits_feature1.t()

        # # 计算相似度得分
        return logits_feature1, logits_feature2


def info_nce_loss(model, data_l, data_L):
    N = data_l.size(0)
    # 计算相似性分数
    # 计算温度参数 tau
    tau = len(data_l[0])**0.5
    
    all_NCE=[]
    # 计算正样本相似性, row of similarity1
    similarity_scores1, similarity_scores2 = model(data_l, data_L)
    for i in range(len(data_l)):
        
        pos_sim = torch.exp(similarity_scores1[i][i])/tau
        neg_sim = sum([torch.exp(similarity_scores1[i][j]/tau) for j in range(len(data_l))])
        all_NCE.append(torch.log(pos_sim/neg_sim))

    # 计算损失
    loss = -sum(all_NCE)/len(all_NCE)  # InfoNCE损失
    return loss

index=1
epoch=399

# 2. 加载模型
model = SimilarityNetwork().to("cuda:0")
model.load_state_dict(torch.load(f'./similarity_network_full_v1_combine_epoch{epoch}.pth'))
model.eval()  # 设置为评估模式

base_model = SimilarityNetwork().to("cuda:0")
base_model.load_state_dict(torch.load(f'./similarity_network_only_image_full_v1_combine_epoch{epoch}.pth'))
base_model.eval()  # 设置为评估模式

input_data_name_list={
    "indom": '../pope/pope_info_probe_listindom_v1.pth',
    # "outdom": '../pope/pope_info_probe_listoutdom_v1.pth',
    "insuff_att": '../pope/pope_info_probe_listinsuff_att_v1.pth',
    "amb_desc": '../pope/pope_info_probe_listamb_desc_v1.pth',
}
base_input_data_name_list={
    "indom":'../pope/image_pope_info_probe_listindom_v1.pth',
    # "outdom":'../pope/image_pope_info_probe_listoutdom_v1.pth',
    "insuff_att":'../pope/image_pope_info_probe_listinsuff_att_v1.pth',
    "amb_desc":'../pope/image_pope_info_probe_listamb_desc_v1.pth',
}

for idx, (input_data_name, base_input_data_name) in enumerate(zip(input_data_name_list.values(), base_input_data_name_list.values())):

    input_datas = torch.load(input_data_name)
    base_input_datas = torch.load(base_input_data_name)

    input_L=torch.stack(input_datas[-1]).to(torch.float).to("cuda:0")
    base_input_L=torch.stack(base_input_datas[-1]).to(torch.float).to("cuda:0")

    index_list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31] # 
    # result=[]
    base_scores=[]
    scores=[]
    for index in index_list:
        print(f"-----------------processing layer{index}----------------------")
        input_l=torch.stack(input_datas[index]).to(torch.float).to("cuda:0")
        base_input_l=torch.stack(base_input_datas[index]).to(torch.float).to("cuda:0")


        # 3. 创建数据集和数据加载器
        dataset = TensorDataset(input_l, input_L)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        base_dataset = TensorDataset(base_input_l, base_input_L)
        base_dataloader = DataLoader(base_dataset, batch_size=32, shuffle=False)

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
        # print("taskinfo and base metric", torch.log(torch.tensor(len(input_l), dtype=torch.float32))-score, torch.log(torch.tensor(len(input_l), dtype=torch.float32))-base_score)
        print("mutual information", base_score-score)
        # result.append(base_score.cpu()-score.cpu())
        base_scores.append(-base_score.cpu())
        scores.append(-score.cpu())
        # break

    torch.save({'scores': scores, 'base_scores': base_scores}, f'text/{list(input_data_name_list.keys())[idx]}_scores_data.pth')
    print("save to ", f'text/{list(input_data_name_list.keys())[idx]}_scores_data.pth')
