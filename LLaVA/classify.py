import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split


# # classify_type="cifar100"
# classify_type="cifar10"
# # classify_type="mnist"
# mask=True

# classify_types=["cifar100", "cifar10", "mnist"]
classify_types=["pope"]
# classify_types=["mnist"]
masks=[False]

for classify_type in classify_types:
    for mask in masks:
        # 定义模型
        if classify_types=="mnist" and masks==True:
            continue
        class SimpleClassifier(nn.Module):
            def __init__(self):
                super(SimpleClassifier, self).__init__()
                self.fc1 = nn.Linear(4096, 512)
                self.fc2 = nn.Linear(512, 128)
                
                if classify_type=="cifar100":
                    self.fc3 = nn.Linear(128, 100)  # 10个分类
                elif classify_type=="pope":
                    self.fc3 = nn.Linear(128, 2)  # 10个分类
                else:
                    self.fc3 = nn.Linear(128, 10)  # 10个分类

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x


        info_save_index=[0, 6, 12, 18, 24, 30]

        # 训练模型
        num_epochs = 500

        if classify_type=="mnist":
            labels = torch.load('label_list.pth')
            input_datas = torch.load('info_probe_list.pth')
            # output = torch.load('output_list.pth')
            # print(output)
            # exit()
        elif classify_type=="cifar100":
            if mask==True:
                labels = torch.load('label_list_cifar100_mask.pth')
                input_datas = torch.load('info_probe_list_cifar100_mask.pth')
            else:
                labels = torch.load('label_list_cifar100.pth')
                input_datas = torch.load('info_probe_list_cifar100.pth')
            # output = torch.load('cifar100_output_list.pth')
            # print(output)
            # exit()
        else:
            if mask==True:
                labels = torch.load('label_list_cifar10_mask.pth')
                input_datas = torch.load('info_probe_list_cifar10_mask.pth')
            else:
                labels = torch.load('label_list_cifar10.pth')
                input_datas = torch.load('info_probe_list_cifar10.pth')

        # 用于保存每个 index 的损失和准确性
        results = {}

        for i, index in enumerate(info_save_index):
            
            input_data=input_datas[i]
            
            print(f"-----------------index {index}-----------------")
            # 初始化模型、损失函数和优化器
            model = SimpleClassifier().to('cuda:0')
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)



            # 将数据转换为 Tensor
            data_tensor = torch.stack(input_data)  # (10, 1024) 假设每个 tensor 是 1024 维
            labels_tensor = torch.tensor(labels, device='cuda:0')

            # 创建 TensorDataset
            dataset = TensorDataset(data_tensor, labels_tensor)

            # 划分训练集和测试集
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

            # 创建 DataLoader
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
 
            # 初始化当前 index 的损失和准确性列表
            all_losses = []
            all_accuracies = []
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
                    for inputs, target in test_loader:
                        inputs = inputs.float().to('cuda:0')
                        target = target.to('cuda:0')
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()


                accuracy = 100 * correct / total
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

                # 保存当前epoch的loss和accuracy
                all_losses.append(loss.item())
                all_accuracies.append(accuracy)
                
            # 将当前 index 的结果存入字典
            results[index] = {
                'losses': all_losses,
                'accuracies': all_accuracies
            }
        print("训练完成！")

        if classify_type=="mnist":
            written_file="training_results.txt"
        elif classify_type=="cifar100":
            if mask==True:
                written_file="training_results_cifar100_mask.txt"
            else:
                written_file="training_results_cifar100.txt"
        else:
            if mask==True:
                written_file="training_results_cifar10_mask.txt"
            else:
                written_file="training_results_cifar10.txt"


        with open(written_file, "w") as f:
            f.write("Index\tEpoch\tLoss\tAccuracy\n")
            for index, metrics in results.items():
                for epoch in range(num_epochs):
                    f.write(f"{index}\t{epoch + 1}\t{metrics['losses'][epoch]:.4f}\t{metrics['accuracies'][epoch]:.2f}\n")