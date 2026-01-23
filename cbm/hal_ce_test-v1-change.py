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

ce_data_dir="/home/junda/zihan/cbm/hal/"
labels = torch.load(ce_data_dir+'hal_ce_val_label_full_v1.pth')
# labels = [torch.tensor(x) for x in labels]
input_datas = torch.load(ce_data_dir+'hal_ce_val_image_full_v1.pth')

missing_index = [128, 129, 130, 131, 136, 137, 138, 139, 520, 521, 522, 523, 1064, 1065, 1066, 1067, 1896, 1897, 1898, 1899, 2068, 2069, 2070, 2071, 2340, 2341, 2342, 2343, 2620, 2621, 2622, 2623, 2812, 2813, 2814, 2815, 3384, 3385, 3386, 3387, 3972, 3973, 3974, 3975, 4292, 4293, 4294, 4295, 4496, 4497, 4498, 4499, 5384, 5385, 5386, 5387, 5552, 5553, 5554, 5555, 6044, 6045, 6046, 6047, 6188, 6189, 6190, 6191, 6356, 6357, 6358, 6359, 6488, 6489, 6490, 6491, 6964, 6965, 6966, 6967, 7496, 7497, 7498, 7499, 7540, 7541, 7542, 7543, 7656, 7657, 7658, 7659, 7956, 7957, 7958, 7959, 8076, 8077, 8078, 8079, 8084, 8085, 8086, 8087, 8096, 8097, 8098, 8099, 8604, 8605, 8606, 8607, 8700, 8701, 8702, 8703, 9040, 9041, 9042, 9043, 9060, 9061, 9062, 9063, 9612, 9613, 9614, 9615, 9628, 9629, 9630, 9631, 10112, 10113, 10114, 10115, 10292, 10293, 10294, 10295, 10488, 10489, 10490, 10491, 11192, 11193, 11194, 11195, 11212, 11213, 11214, 11215, 11800, 11801, 11802, 11803, 12036, 12037, 12038, 12039, 12760, 12761, 12762, 12763, 13324, 13325, 13326, 13327, 13720, 13721, 13722, 13723, 13856, 13857, 13858, 13859, 14152, 14153, 14154, 14155, 14260, 14261, 14262, 14263, 14280, 14281, 14282, 14283, 14348, 14349, 14350, 14351, 14356, 14357, 14358, 14359, 14364, 14365, 14366, 14367, 14408, 14409, 14410, 14411, 14420, 14421, 14422, 14423, 14908, 14909, 14910, 14911, 14916, 14917, 14918, 14919, 14928, 14929, 14930, 14931, 15204, 15205, 15206, 15207, 15344, 15345, 15346, 15347, 15360, 15361, 15362, 15363, 15372, 15373, 15374, 15375, 15784, 15785, 15786, 15787, 15860, 15861, 15862, 15863, 16300, 16301, 16302, 16303, 16468, 16469, 16470, 16471, 16972, 16973, 16974, 16975, 17400, 17401, 17402, 17403, 17412, 17413, 17414, 17415, 17424, 17425, 17426, 17427, 17696, 17697, 17698, 17699, 17816, 17817, 17818, 17819, 17820, 17821, 17822, 17823, 17936, 17937, 17938, 17939, 18412, 18413, 18414, 18415, 18448, 18449, 18450, 18451, 18840, 18841, 18842, 18843, 18924, 18925, 18926, 18927, 19408, 19409, 19410, 19411, 19416, 19417, 19418, 19419]

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


text_margin=False
seperate_layer=False
# margin_num_list=[3,5,10,20,30,40,50,100]
margin_num_list=[3] # ,20,30,40,50,100
epoch=25


for margin_num in margin_num_list:
    if text_margin:
        written_file=f'result/hal_all_kls_text_margin_{margin_num}_trainepoch{epoch}_v1'
        pic_file=f'result/hal_line_chart_text_margin_{margin_num}_trainepoch{epoch}_v1'
    else:
        written_file=f'result/image_hal_all_kls_merge_trainepoch{epoch}_v1'
        pic_file=f'result/image_hal_line_chart_merge_trainepoch{epoch}_v1'
        if seperate_layer:
            pic_file=f'result/image_hal_line_chart_seperatelayer_trainepoch{epoch}_v1'

    if use_avg_prior:
        written_file+="_avg"

    written_file+=".txt"
    pic_file+=".pdf"

    if text_margin:
        index_to_cluster=torch.load(f'margin/hal_index_to_cluster-{margin_num}.pth')
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