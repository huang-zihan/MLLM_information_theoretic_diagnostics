import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# info_save_index = [0, 6, 12, 18, 24, 30]
info_save_index=[0, 12, 24, 27, 30, 31]
# info_save_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    
# info_save_index=[0, 12, 24]
# 训练模型
num_epochs = 400

item_dict = {'snowboard': 0, 'backpack': 1, 'person': 2, 'car': 3, 'skis': 4, 'dog': 5, 'truck': 6, 'dining table': 7, 'handbag': 8, 'bicycle': 9, 'motorcycle': 10, 'potted plant': 11, 'vase': 12, 'traffic light': 13, 'bus': 14, 'chair': 15, 'bed': 16, 'book': 17, 'spoon': 18, 'cup': 19, 'fork': 20, 'tv': 21, 'toaster': 22, 'microwave': 23, 'bottle': 24, 'bird': 25, 'boat': 26, 'couch': 27, 'sandwich': 28, 'bowl': 29, 'hot dog': 30, 'frisbee': 31, 'knife': 32, 'cake': 33, 'remote': 34, 'baseball glove': 35, 'sports ball': 36, 'baseball bat': 37, 'bench': 38, 'sink': 39, 'toilet': 40, 'teddy bear': 41, 'bear': 42, 'cat': 43, 'mouse': 44, 'laptop': 45, 'toothbrush': 46, 'cow': 47, 'skateboard': 48, 'surfboard': 49, 'cell phone': 50, 'train': 51, 'clock': 52, 'tennis racket': 53, 'suitcase': 54, 'horse': 55, 'banana': 56, 'wine glass': 57, 'refrigerator': 58, 'carrot': 59, 'broccoli': 60, 'tie': 61, 'scissors': 62, 'sheep': 63, 'airplane': 64, 'stop sign': 65, 'fire hydrant': 66, 'keyboard': 67, 'pizza': 68, 'donut': 69, 'kite': 70, 'parking meter': 71, 'giraffe': 72, 'zebra': 73, 'umbrella': 74, 'orange': 75, 'oven': 76, 'elephant': 77, 'apple': 78}


# 加载标签和输入数据
# ['global', '', 'dog', 'car', 'keyboard']
# ['dog', 'car', 'keyboard']
# ['global', '', 'dog', 'car', 'keyboard']
# ["dog", "chair", "backpack", "keyboard"]
for item in [""]:
    # ['dog']
    #
    vit=True
    # if item in ['dog', 'car', 'keyboard']:
    #     one_item=True
    # else:
    #     one_item=False
    one_item=False
    # 定义模型
    class SimpleClassifier(nn.Module):
        def __init__(self):
            super(SimpleClassifier, self).__init__()
            if vit==True:
                # self.fc1 = nn.Linear(589824, 512)
                self.fc1 = nn.Linear(1024, 512)
            else:
                self.fc1 = nn.Linear(4096, 512)  # 假设每个 tensor 是 1024 维
            self.fc2 = nn.Linear(512, 128)
            if one_item:
                self.fc3 = nn.Linear(128, 1)  # 10个分类
            else:
                self.fc3 = nn.Linear(128, 79)  # 10个分类

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
            # if one_item:
            #     x = self.fc3(x)
            # else:
            #     x = torch.sigmoid(self.fc3(x))
            return x
    
    print(f'processing {item}')
    if item == '':
        labels = torch.load('pope_label_list.pth')
        if not vit:
            input_datas = torch.load('pope_info_probe_list.pth')
            written_file="pope_result.txt"
        else:
            vit_feature_list = torch.load('../cbm/pope/pope_vit_feature_list.pth')
            written_file="vit_pope_result.txt"
            

    elif item == 'global':
        labels = torch.load('pope_label_list.pth')
        if not vit:
            input_datas = torch.load('globalpope_info_probe_list.pth')
            written_file="globalpope_result.txt"
        else:
            vit_feature_list = torch.load('../cbm/pope/globalpope_vit_feature_list.pth')
            written_file="vit_globalpope_result.txt"
        
    else:
        target_index=item_dict[item]
        labels = torch.load('pope_label_list.pth')
        # labels = [int(x[target_index]==1) for x in labels]
        input_datas = torch.load(f'pope_info_probe_list{item}.pth')
        written_file=f"pope_result_{item}.txt"


    # print(len(input_datas))
    # exit()
    
    if vit==True:
        written_file="vit_"+written_file
        info_save_index=[0]

    results={}
    inference_result={}
    # classifiy_res=[]
    # gt_res=[]
    
    for i, index in enumerate(info_save_index):
        print(f"-----------------index {index}-----------------")
        
        # 初始化模型、损失函数和优化器
        model = SimpleClassifier().to('cuda:0')
        if one_item:
            criterion = nn.BCELoss()
            # criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCELoss()  # 二元交叉熵损失
            
        if vit:
            optimizer = optim.Adam(model.parameters(), lr=0.005)
        else:    
            optimizer = optim.Adam(model.parameters(), lr=0.005)

        if not vit:
            input_data=input_datas[i]
        
        # 将数据转换为 Tensor
        if vit:
            data_tensor = torch.stack(vit_feature_list)
        else:
            data_tensor = torch.stack(input_data)  # (N, 1024) 假设每个 tensor 是 1024 维
        labels_tensor = torch.tensor(labels, device='cuda:0')

        # 创建 TensorDataset
        dataset = TensorDataset(data_tensor, labels_tensor)

        # # # 划分训练集和测试集
        # train_size = int(0.8 * len(dataset))
        # test_size = len(dataset) - train_size
        # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        # 划分训练集和测试集
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        # 创建索引
        train_indices = list(range(train_size))
        test_indices = list(range(train_size, len(dataset)))

        # 使用 Subset 创建子集
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
        
        # # 顺序划分
        # train_dataset = dataset[:train_size]
        # test_dataset = dataset[train_size:]


        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # 32
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
                # print(outputs)
                # print(target)
                # exit()
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
            
            
            
            # 测试模型
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                total_f1=[]
                for inputs, target in test_loader:
                    inputs = inputs.float().to('cuda:0')
                    target = target.int()
                    
                    test_outputs = model(inputs)
                    predicted = (test_outputs >= 0.5).int()  # 将输出转换为二进制
                    
                    classifiy_res_epoch+=predicted
                    if gt_res_index==[]:
                        gt_res_epoch+=target
                    
                    
                    if not one_item:
                        total_f1+=[f1_score(predicted.cpu().numpy().tolist()[i], target.cpu().numpy().tolist()[i]) for i in range(len(test_outputs))]
                    else:
                        total += target.size(0)
                        # print(predicted)
                        # print(target)
                        correct += (predicted.squeeze() == target).sum().item()
                        # print(correct, total)
                        # exit()
            if one_item:
                accuracy = 100 * correct / total
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
            else:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test F1: {sum(total_f1)/len(total_f1):.2f}')
            # 保存当前epoch的loss和accuracy
            
            all_losses.append(loss.item())
            if one_item:
                all_accuracies.append(accuracy)
            else:
                all_f1.append(sum(total_f1)/len(total_f1))
            
            classifiy_res_index.append(classifiy_res_epoch)
            if gt_res_index==[]:
                gt_res_index=gt_res_epoch
            
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
            break
        else:
            if one_item:
                results[index] = {
                    'losses': all_losses,
                    'accuracies': all_accuracies,
                }
                
                inference_result[index] = {
                    'classifiy_res': classifiy_res_index,
                    'gt': gt_res_index
                }
            else:
                results[index] = {
                    'losses': all_losses,
                    'accuracies': all_f1,
                }
                
                inference_result[index] = {
                    'classifiy_res': classifiy_res_index,
                    'gt': gt_res_index
                }
        
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