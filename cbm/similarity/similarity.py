import os
# 设置可见 GPU 为 4
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import random

# 1. 模型定义
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
        # logits_feature1 = feature1 @ feature2.t()
        return logits_feature1, logits_feature2


# index=1
for only_image in [False]:
# only_image=True
    input_datas = torch.load('../data/ce_training_full_v1.pth') if not only_image else torch.load('../data/ce_training_image_full_v1.pth')

    index_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31] 
    select_ratio=0.2
    for i, index in enumerate(index_list):
        rand_index = random.sample(range(len(input_datas[index])), k=int(select_ratio*len(input_datas[index])))
        input_datas[index]=torch.stack(input_datas[index]).to(torch.float)[rand_index]

        if i==0:
            input_L = torch.stack(input_datas[-1])[rand_index].to(torch.float)
        else:
            input_L = torch.cat((input_L, torch.stack(input_datas[-1])[rand_index].to(torch.float)), dim=0)

    # for i, index in enumerate(index_list):
    #     input_datas[index]=torch.stack(input_datas[index]).to(torch.float)
    #     if i==0:
    #         input_L = torch.stack(input_datas[-1]).to(torch.float)
    #     else:
    #         input_L = torch.cat((input_L, torch.stack(input_datas[-1]).to(torch.float)), dim=0)

    input_l = input_datas = torch.stack(input_datas[:32]).to(torch.float).view(-1, 4096)
    
    print(input_l.shape, input_L.shape)
    # exit()
    # for i, index in enumerate(index_list):
        # index_list = random.sample(range(len(input_datas[index])), k=int(select_ratio*len(input_datas[index])))
        # if i==0:
        #     input_l = torch.stack(input_datas[index]).to(torch.float)[index_list]
        #     input_L = torch.stack(input_datas[-1]).to(torch.float)[index_list]
        # else:
        #     input_l = torch.cat((input_l, torch.stack(input_datas[index]).to(torch.float)[index_list]), dim=0)
        #     input_L = torch.cat((input_L, torch.stack(input_datas[-1]).to(torch.float)[index_list]), dim=0)

    # input_l = input_l[index_list].to("cuda:0")
    # input_L = input_L[index_list].to("cuda:0")
    input_l = input_l.to("cuda:0")
    input_L = input_L.to("cuda:0")

    # 4. 创建数据加载器
    train_dataset = TensorDataset(input_l, input_L)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 5. 训练过程
    def train_model(model, train_loader, criterion, optimizer, num_epochs=30, save_per=50):
        model.train()
        for epoch in range(num_epochs):
            for batch_input_l, batch_input_L in train_loader:
                optimizer.zero_grad()
                # calculate similarity
                similarity_scores1, similarity_scores2 = model(batch_input_l, batch_input_L)
                # print(similarity_scores.shape, feature1.shape, feature2.shape) #[32, 32] [32, 64] [bs, dim]

                labels = torch.arange(batch_input_l.size(0), device=batch_input_l.device)
                # print(labels)

                loss1 = criterion(similarity_scores1, labels)
                loss2 = criterion(similarity_scores2, labels)
                loss=loss1+loss2
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            
            if (epoch + 1)%save_per==0:
                if not only_image:
                    torch.save(model.state_dict(), f'./similarity_network_full_v1_combine_epoch{epoch}.pth')
                else:
                    torch.save(model.state_dict(), f'./similarity_network_only_image_full_v1_combine_epoch{epoch}.pth')

    # 6. 初始化模型、损失函数和优化器
    model = SimilarityNetwork().to("cuda:0")
    criterion = nn.CrossEntropyLoss().to("cuda:0")  # 使用交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)


    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs=400)

# # 7. 验证模型
# def evaluate_model(model, val_loader):
#     model.eval()
#     with torch.no_grad():
#         total, correct = 0, 0
#         for batch_input_l, batch_input_L in val_loader:
#             similarity_scores = model(batch_input_l, batch_input_L)
#             predicted = torch.argmax(similarity_scores, dim=1)
#             labels = torch.arange(batch_input_l.size(0), device=batch_input_l.device)

#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     accuracy = correct / total
#     print(f'Validation Accuracy: {accuracy:.4f}')

# # 创建验证集 DataLoader
# val_dataset = TensorDataset(input_l_val, input_L_val)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# # 验证模型
# evaluate_model(model, val_loader)

# if not only_image:
#     torch.save(model.state_dict(), f'./similarity_network_full_v1_layer{index}.pth')
# else:
#     torch.save(model.state_dict(), f'./similarity_network_only_image_full_v1_layer{index}.pth')






############
    # for only_image in [True, False]:
    #     # 2. 数据准备
    #     input_datas = torch.load('../data/ce_training_full_v1.pth') if not only_image else torch.load('../data/ce_training_image_full_v1.pth')

    #     # # 假设 input_datas 是一个包含多个张量的列表
    #     # input_L = torch.stack(input_datas[-1]).to(torch.float).to("cuda:0").unsqueeze(0).expand(32, -1, -1).reshape(-1, 4096)
    #     # input_l = torch.cat([torch.stack(input_datas[i]).to(torch.float).to("cuda:0") for i in range(32)], dim=0)
    #     # 对于 input_L，将其复制 32 次
    #     print(input_l.shape)

    #     # # 3. 分割数据集
    #     # input_l_train, input_l_val, input_L_train, input_L_val = train_test_split(
    #     #     input_l, input_L, test_size=0.2, random_state=42
    #     # )
    #     input_l_train=input_l
    #     input_L_train=input_L

        # # 4. 创建数据加载器
        # train_dataset = TensorDataset(input_l_train, input_L_train)
        # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # # 5. 训练过程
        # def train_model(model, train_loader, criterion, optimizer, num_epochs=200):
        #     model.train()
        #     for epoch in range(num_epochs):
        #         for batch_input_l, batch_input_L in train_loader:
        #             optimizer.zero_grad()
        #             # 计算相似性分数
        #             similarity_scores1, similarity_scores2 = model(batch_input_l, batch_input_L)
        #             # print(similarity_scores.shape, feature1.shape, feature2.shape) #[32, 32] [32, 64] [bs, dim]
        #             # 生成标签，正样本为1
        #             labels = torch.arange(batch_input_l.size(0), device=batch_input_l.device)
        #             # print(labels)
        #             # 计算损失
        #             loss1 = criterion(similarity_scores1, labels)
        #             loss2 = criterion(similarity_scores2, labels)
        #             loss=loss1+loss2
        #             loss.backward()
        #             optimizer.step()

        #         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # # 6. 初始化模型、损失函数和优化器
        # model = SimilarityNetwork().to("cuda:0")
        # criterion = nn.CrossEntropyLoss().to("cuda:0")  # 使用交叉熵损失
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


        # # 训练模型
        # train_model(model, train_loader, criterion, optimizer, num_epochs=200)

        # # # 7. 验证模型
        # # def evaluate_model(model, val_loader):
        # #     model.eval()
        # #     with torch.no_grad():
        # #         total, correct = 0, 0
        # #         for batch_input_l, batch_input_L in val_loader:
        # #             similarity_scores = model(batch_input_l, batch_input_L)
        # #             predicted = torch.argmax(similarity_scores, dim=1)
        # #             labels = torch.arange(batch_input_l.size(0), device=batch_input_l.device)

        # #             total += labels.size(0)
        # #             correct += (predicted == labels).sum().item()

        # #     accuracy = correct / total
        # #     print(f'Validation Accuracy: {accuracy:.4f}')

        # # # 创建验证集 DataLoader
        # # val_dataset = TensorDataset(input_l_val, input_L_val)
        # # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # # # 验证模型
        # # evaluate_model(model, val_loader)

        # if not only_image:
        #     torch.save(model.state_dict(), f'./similarity_network_full_v1_layer{index}.pth')
        # else:
        #     torch.save(model.state_dict(), f'./similarity_network_only_image_full_v1_layer{index}.pth')
