import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn.functional as F
import matplotlib.pyplot as plt

info_save_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]    
use_avg_prior=False
text_margin=False
seperate_layer=False

ce_data_dir="/home/junda/zihan/cbm/vqa/"
labels = torch.load(ce_data_dir+'vqa_ce_val_label_full_v1.pth')
input_datas = torch.load(ce_data_dir+'vqa_ce_val_image_full_v1.pth')

missing_index = [68, 69, 70, 270, 271, 272, 273, 928, 929, 930, 1240, 1241, 1242, 1502, 1503, 1504, 3823, 3824, 3825, 3826, 3827, 3828, 3829, 3830, 3831, 3912, 3913, 3914, 3915, 3920, 3921, 3922, 3923, 3924, 4171, 4172, 4173, 4174, 4175, 4176, 4177, 5159, 5160, 5161, 5162, 5163, 5164, 5165, 5166, 5167, 5168, 5169, 5170, 5171, 5172, 5173, 5174, 5175, 5176, 5177, 5178, 5179, 5180, 5181, 5182, 5183, 5944, 5945, 5946, 5947, 6007, 6008, 6009, 6803, 6804, 6805, 6806, 6807, 6808, 6809, 6810, 6811, 6812, 7022, 7023, 7024, 7025, 7026, 7027, 7028, 7029, 7030, 7031, 7032, 7033, 7034, 7229, 7230, 7231, 7609, 7610, 7611, 7771, 7772, 7773, 7774, 8054, 8055, 8056, 8270, 8271, 8272, 8505, 8506, 8507, 8508, 8509, 8931, 8932, 8933, 8934, 8935, 9600, 9601, 9602, 9603, 9604, 9605, 9606, 9607, 9608, 9609, 9863, 9864, 9865, 9866, 9867, 9868, 9869, 9870, 9871, 9872, 9873, 9992, 9993, 9994, 9995, 9996, 9997, 9998, 9999]
labels = [x for i, x in enumerate(labels) if i not in missing_index]

if use_avg_prior:
    sum_tensor = None
    num_tensors = len(labels)
    for tensor in labels:
        tensor=torch.tensor(tensor)
        if sum_tensor is None:
            sum_tensor = tensor
        else:
            sum_tensor += tensor
    average_tensor = [float(x) for x in list(sum_tensor / num_tensors)]
    labels = [average_tensor for _ in range(len(labels))]

for i in range(len(input_datas)):
    input_datas[i]=[x for i, x in enumerate(input_datas[i]) if i not in missing_index]

# margin_num_list=[3,5,10,20,30,40,50,100]
margin_num_list=[3] # ,20,30,40,50,100
epoch=25


for margin_num in margin_num_list:
    if text_margin:
        written_file=f'result/vqa_all_kls_text_margin_{margin_num}_trainepoch{epoch}_v1'
        pic_file=f'result/vqa_line_chart_text_margin_{margin_num}_trainepoch{epoch}_v1'
    else:
        written_file=f'result/image_vqa_all_kls_merge_trainepoch{epoch}_v1'
        pic_file=f'result/image_vqa_line_chart_merge_trainepoch{epoch}_v1'
        if seperate_layer:
            pic_file=f'result/image_vqa_line_chart_seperatelayer_trainepoch{epoch}_v1'

    if use_avg_prior:
        written_file+="_avg"

    written_file+=".txt"
    pic_file+=".pdf"

    if text_margin:
        index_to_cluster=torch.load(f'margin/vqa_index_to_cluster-{margin_num}.pth')
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
    
    # 初始化模型、损失函数和优化器
    if seperate_layer == False:
        model = CE()
        model.load_state_dict(torch.load(f'./result/ckpt/ce_model_merge_epoch{epoch}_v1.pth'))
        model.to('cuda:0')
    # print("info_save_index length:", len(info_save_index))
    for i, index in enumerate(info_save_index):
        print(f"-----------------index {index}-----------------")
        
        if seperate_layer:
            # if i>=30:
            #     break
            model = CE()
            model.load_state_dict((torch.load(f'./result/ckpt/ce_model_{i}_epoch{epoch}_v1.pth')))
            # model.load_state_dict((torch.load(f'./result/ce_model_{i}.pth')))
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
        print("test_dataset", len(test_dataset))
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        all_losses=[]
        all_f1=[]
        all_accuracies = []
        classifiy_res_index=[]
        gt_res_index=[]

        cnt=0
        with torch.no_grad():
            total_kl=[]
            for inputs, target in test_loader:
                inputs = inputs.float().to('cuda:0')
                predicted = model(inputs)

                for j in range(len(target)):
                    if torch.all(target[j] == 0):
                        cnt+=1
                        continue

                    if text_margin:
                        seperate_kl[index_to_cluster[j]].append(F.kl_div(torch.log_softmax(predicted[j], dim=0), target[j].float()))
                    else:
                        total_kl.append(F.kl_div(torch.log_softmax(predicted[j], dim=0), target[j].float()))

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