import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn.functional as F
import matplotlib.pyplot as plt

# info_save_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]    
info_save_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]    


ce_data_dir="/home/junda/zihan/cbm/pope/"

input_datas = torch.load(ce_data_dir+'pope_info_probe_list.pth')

labels = torch.load(ce_data_dir+'pope_label_list.pth')

text_margin=False
margin_num_list=[3,5,10,20,30,40,50,100]

for margin_num in margin_num_list:
    if text_margin:
        written_file=f'result/all_kls_text_margin_{margin_num}.txt'
        pic_file=f'result/line_chart_text_margin_{margin_num}.pdf'
    else:
        written_file='result/all_kls.txt'
        pic_file='result/line_chart.pdf'


    if text_margin:
        index_to_cluster=torch.load(f'margin/index_to_cluster-{margin_num}.pth')
        seperate_kl=[[] for _ in range(margin_num)]


    vit=False
    one_item=False
    # 定义模型
    class CE(nn.Module):
        def __init__(self):
            super(CE, self).__init__()

            self.fc1 = nn.Linear(4096, 512)  # 假设每个 tensor 是 1024 维
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, len(labels[0]))  # 10个分类
            # print(len(labels[0]))

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    if vit==True:
        written_file="vit_"+written_file
        info_save_index=[0]

    results={}
    inference_result={}
    all_kls=[]
    for i, index in enumerate(info_save_index):
        print(f"-----------------index {index}-----------------")
        
        # 初始化模型、损失函数和优化器
        model = CE()
        model.load_state_dict(torch.load(f'./result/ce_model_{index}.pth'))
        model.to('cuda:0')

        if not vit:
            input_data=input_datas[i]

        # 将数据转换为 Tensor
        if vit:
            data_tensor = torch.stack(vit_feature_list)
        else:
            data_tensor = torch.stack(input_data)  # (N, 1024) 假设每个 tensor 是 1024 维

        labels_tensor = torch.tensor(labels, device='cuda:0')

        # 创建 TensorDataset
        test_dataset = TensorDataset(data_tensor, labels_tensor)
        
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        all_losses=[]
        all_f1=[]
        all_accuracies = []
        classifiy_res_index=[]
        gt_res_index=[]


        with torch.no_grad():
            total_kl=[]
            for inputs, target in test_loader:
                inputs = inputs.float().to('cuda:0')
                target = target.int()
                predicted = model(inputs)

                for j in range(len(target)):
                    if torch.all(target[j] == 0):
                        continue

                    if text_margin:
                        seperate_kl[index_to_cluster[j]].append(F.kl_div(F.softmax(predicted[j]).log(), target[j].float()))
                    else:
                        total_kl.append(F.kl_div(F.softmax(predicted[j]).log(), target[j].float()))


        if text_margin:
            # 计算每个子列表的平均值
            averages = []
            for sublist in seperate_kl:
                if sublist:  # 检查子列表是否为空
                    avg = sum(sublist) / len(sublist)
                    averages.append(avg)

            # 计算所有非空子列表的平均值
            if averages:  # 检查averages是否为空
                overall_avg = sum(averages) / len(averages)
            else:
                overall_avg = None  # 如果没有非空子列表，返回None
            all_kls.append(overall_avg)
            print(f'KL: {overall_avg}')
        else:
            print("total_kl length", len(total_kl))
            print(f'KL: {sum(total_kl)/len(total_kl)}')
            all_kls.append(sum(total_kl)/len(total_kl))
        

    all_kls=[float(x.cpu()) for x in all_kls]
    print(all_kls)
    # 将列表保存到本地文件

    with open(written_file, 'w') as f:
        for item in all_kls:
            f.write(f"{item}\n")

    # 绘制折线图
    plt.plot(all_kls, marker='o')
    if not text_margin:
        plt.title('Upper bound of Unconditional MI Proxy')
        plt.xlabel('layer')
        plt.ylabel('metric value')
    else:
        plt.title('Upper bound of Conditional MI Proxy')
        plt.xlabel('layer')
        plt.ylabel('metric value')
    plt.grid()
    plt.savefig(pic_file, format='pdf')  # 保存图表为图片
    # plt.show()  # 显示图表    
    plt.clf()
    
    if text_margin==False:
        break