import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn.functional as F
import matplotlib.pyplot as plt

info_save_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]    

use_avg_prior=True
text_margin=False
seperate_layer=False

ce_data_dir="/home/junda/zihan/cbm/chair/"
labels = torch.load(ce_data_dir+'chair_label_list_v1.pth')
input_datas = torch.load(ce_data_dir+'chair_info_probe_list_v1.pth')

missing_index = [12, 13, 26, 27, 58, 59, 60, 61, 62, 63, 78, 79, 92, 93, 112, 113, 118, 119, 126, 127, 132, 133, 146, 147, 152, 153, 180, 181, 214, 215, 218, 219, 220, 221, 228, 229, 246, 247, 254, 255, 260, 261, 268, 269, 290, 291, 300, 301, 316, 317, 346, 347, 402, 403, 410, 411, 448, 449, 460, 461, 464, 465, 486, 487, 494, 495, 496, 497, 514, 515, 550, 551, 554, 555, 576, 577, 592, 593, 594, 595, 596, 597, 602, 603, 604, 605, 618, 619, 668, 669, 726, 727, 736, 737, 754, 755, 832, 833, 854, 855, 868, 869, 878, 879, 882, 883, 888, 889, 894, 895, 918, 919, 930, 931, 942, 943, 946, 947, 954, 955, 958, 959, 972, 973, 976, 977, 978, 979, 998, 999, 1008, 1009, 1022, 1023, 1034, 1035, 1116, 1117, 1138, 1139, 1158, 1159, 1162, 1163, 1182, 1183, 1192, 1193, 1194, 1195, 1212, 1213, 1240, 1241, 1254, 1255, 1282, 1283, 1306, 1307, 1312, 1313, 1332, 1333, 1374, 1375, 1380, 1381, 1392, 1393, 1426, 1427, 1436, 1437, 1454, 1455, 1472, 1473, 1506, 1507, 1528, 1529, 1546, 1547, 1570, 1571, 1582, 1583, 1602, 1603, 1604, 1605, 1626, 1627, 1662, 1663, 1698, 1699, 1702, 1703, 1704, 1705, 1710, 1711, 1718, 1719, 1736, 1737, 1768, 1769, 1772, 1773, 1774, 1775, 1798, 1799, 1832, 1833, 1866, 1867, 1874, 1875, 1930, 1931, 1932, 1933, 1934, 1935, 1940, 1941, 1974, 1975, 1988, 1989, 2046, 2047, 2064, 2065, 2076, 2077, 2086, 2087, 2096, 2097, 2102, 2103, 2110, 2111, 2166, 2167, 2190, 2191, 2198, 2199, 2248, 2249, 2282, 2283, 2288, 2289, 2292, 2293, 2326, 2327, 2346, 2347, 2354, 2355, 2362, 2363, 2410, 2411, 2454, 2455, 2490, 2491, 2492, 2493, 2502, 2503, 2504, 2505, 2526, 2527, 2532, 2533, 2552, 2553, 2570, 2571, 2600, 2601, 2604, 2605, 2638, 2639, 2656, 2657, 2668, 2669, 2680, 2681, 2708, 2709, 2742, 2743, 2754, 2755, 2774, 2775, 2816, 2817, 2840, 2841, 2848, 2849, 2880, 2881, 2884, 2885, 2898, 2899, 2920, 2921, 2930, 2931, 2942, 2943, 2944, 2945, 2976, 2977, 3012, 3013, 3046, 3047, 3126, 3127, 3142, 3143, 3148, 3149, 3160, 3161, 3178, 3179, 3192, 3193, 3194, 3195, 3204, 3205, 3212, 3213, 3228, 3229, 3242, 3243, 3314, 3315, 3318, 3319, 3328, 3329, 3340, 3341, 3356, 3357, 3366, 3367, 3370, 3371, 3372, 3373, 3380, 3381, 3422, 3423, 3452, 3453, 3464, 3465, 3476, 3477, 3512, 3513, 3520, 3521, 3536, 3537, 3552, 3553, 3578, 3579, 3582, 3583, 3590, 3591, 3594, 3595, 3632, 3633, 3666, 3667, 3684, 3685, 3708, 3709, 3718, 3719, 3734, 3735, 3788, 3789, 3838, 3839, 3842, 3843, 3872, 3873, 3878, 3879, 3880, 3881, 3890, 3891, 3930, 3931, 3934, 3935, 3936, 3937, 3938, 3939, 3956, 3957, 3984, 3985, 3992, 3993, 4000, 4001, 4012, 4013, 4026, 4027, 4032, 4033, 4042, 4043, 4056, 4057, 4078, 4079, 4094, 4095, 4144, 4145, 4152, 4153, 4216, 4217, 4230, 4231, 4234, 4235, 4258, 4259, 4260, 4261, 4264, 4265, 4294, 4295, 4310, 4311, 4324, 4325, 4328, 4329, 4352, 4353, 4368, 4369, 4372, 4373, 4380, 4381, 4392, 4393, 4418, 4419, 4432, 4433, 4442, 4443, 4474, 4475, 4510, 4511, 4534, 4535, 4552, 4553, 4564, 4565, 4586, 4587, 4628, 4629, 4638, 4639, 4672, 4673, 4690, 4691, 4726, 4727, 4734, 4735, 4738, 4739, 4758, 4759, 4766, 4767, 4800, 4801, 4808, 4809, 4830, 4831, 4836, 4837, 4872, 4873, 4950, 4951, 4966, 4967, 4994, 4995, 5012, 5013, 5022, 5023, 5040, 5041, 5070, 5071, 5076, 5077, 5096, 5097, 5106, 5107, 5134, 5135, 5148, 5149, 5158, 5159, 5160, 5161, 5208, 5209, 5244, 5245, 5254, 5255, 5264, 5265, 5280, 5281, 5282, 5283, 5288, 5289, 5294, 5295, 5306, 5307, 5312, 5313, 5314, 5315, 5340, 5341, 5356, 5357, 5362, 5363, 5372, 5373, 5384, 5385, 5398, 5399, 5416, 5417, 5420, 5421, 5424, 5425, 5430, 5431, 5442, 5443, 5452, 5453, 5460, 5461, 5474, 5475, 5478, 5479, 5508, 5509, 5514, 5515, 5546, 5547, 5548, 5549, 5556, 5557, 5558, 5559, 5628, 5629, 5634, 5635, 5654, 5655, 5688, 5689, 5694, 5695, 5708, 5709, 5710, 5711, 5716, 5717, 5720, 5721, 5722, 5723, 5730, 5731, 5764, 5765, 5784, 5785, 5826, 5827, 5834, 5835, 5838, 5839, 5846, 5847, 5848, 5849, 5866, 5867, 5880, 5881, 5934, 5935, 5952, 5953, 5992, 5993, 6004, 6005, 6088, 6089, 6098, 6099, 6102, 6103, 6112, 6113, 6148, 6149, 6158, 6159, 6174, 6175, 6208, 6209, 6214, 6215, 6218, 6219, 6230, 6231, 6250, 6251, 6264, 6265, 6292, 6293, 6326, 6327, 6366, 6367, 6372, 6373, 6380, 6381, 6390, 6391, 6392, 6393, 6398, 6399, 6422, 6423, 6424, 6425, 6436, 6437, 6458, 6459, 6510, 6511, 6526, 6527, 6542, 6543, 6562, 6563, 6582, 6583, 6614, 6615, 6618, 6619, 6694, 6695, 6702, 6703, 6706, 6707, 6722, 6723, 6728, 6729, 6734, 6735, 6752, 6753, 6768, 6769, 6770, 6771, 6788, 6789, 6802, 6803, 6826, 6827, 6828, 6829, 6848, 6849, 6862, 6863, 6890, 6891, 6896, 6897, 6938, 6939, 6966, 6967, 7004, 7005, 7006, 7007, 7072, 7073, 7084, 7085, 7148, 7149, 7204, 7205, 7218, 7219, 7224, 7225, 7228, 7229, 7252, 7253, 7256, 7257, 7264, 7265, 7296, 7297, 7298, 7299, 7310, 7311, 7336, 7337, 7348, 7349, 7376, 7377, 7378, 7379, 7380, 7381, 7394, 7395, 7406, 7407, 7408, 7409, 7416, 7417, 7450, 7451, 7452, 7453, 7476, 7477, 7488, 7489, 7490, 7491, 7492, 7493, 7528, 7529, 7544, 7545, 7550, 7551, 7554, 7555, 7580, 7581, 7586, 7587, 7606, 7607, 7618, 7619, 7630, 7631, 7632, 7633, 7646, 7647, 7650, 7651, 7668, 7669, 7690, 7691, 7698, 7699, 7706, 7707, 7722, 7723, 7740, 7741, 7748, 7749, 7790, 7791, 7800, 7801, 7812, 7813, 7820, 7821, 7822, 7823, 7842, 7843, 7848, 7849, 7852, 7853, 7902, 7903, 7934, 7935, 7940, 7941, 7942, 7943, 7962, 7963, 7996, 7997, 8012, 8013, 8014, 8015, 8016, 8017, 8024, 8025, 8036, 8037, 8068, 8069, 8080, 8081, 8086, 8087, 8098, 8099, 8106, 8107, 8114, 8115, 8164, 8165, 8174, 8175, 8218, 8219, 8222, 8223, 8282, 8283, 8290, 8291, 8306, 8307, 8324, 8325, 8332, 8333, 8378, 8379, 8394, 8395, 8410, 8411, 8418, 8419, 8422, 8423, 8426, 8427, 8446, 8447, 8450, 8451, 8452, 8453, 8460, 8461, 8480, 8481, 8484, 8485, 8522, 8523, 8546, 8547, 8554, 8555, 8568, 8569, 8572, 8573, 8580, 8581, 8592, 8593, 8598, 8599, 8608, 8609, 8630, 8631, 8636, 8637, 8654, 8655, 8670, 8671, 8678, 8679, 8688, 8689, 8714, 8715, 8746, 8747, 8760, 8761, 8764, 8765, 8798, 8799, 8804, 8805, 8814, 8815, 8836, 8837, 8838, 8839, 8868, 8869, 8874, 8875, 8894, 8895, 8904, 8905, 8918, 8919, 8950, 8951, 8952, 8953, 8958, 8959, 8968, 8969, 8978, 8979, 9064, 9065, 9072, 9073, 9076, 9077, 9092, 9093, 9106, 9107, 9108, 9109, 9118, 9119, 9148, 9149, 9208, 9209, 9250, 9251, 9316, 9317, 9336, 9337, 9342, 9343, 9352, 9353, 9404, 9405, 9434, 9435, 9452, 9453, 9454, 9455, 9488, 9489, 9508, 9509, 9534, 9535, 9542, 9543, 9560, 9561, 9604, 9605, 9608, 9609, 9610, 9611, 9616, 9617, 9624, 9625, 9638, 9639, 9652, 9653, 9656, 9657, 9698, 9699, 9700, 9701]
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
        written_file=f'result/chair_all_kls_text_margin_{margin_num}_trainepoch{epoch}_v1'
        pic_file=f'result/chair_line_chart_text_margin_{margin_num}_trainepoch{epoch}_v1'
    else:
        written_file=f'result/chair_all_kls_merge_trainepoch{epoch}_v1'
        pic_file=f'result/chair_line_chart_merge_trainepoch{epoch}_v1'
        if seperate_layer:
            pic_file=f'result/chair_line_chart_seperatelayer_trainepoch{epoch}_v1'

    if use_avg_prior:
        written_file+="_avg"

    written_file+=".txt"
    pic_file+=".pdf"

    if text_margin:
        index_to_cluster=torch.load(f'margin/chair_index_to_cluster-{margin_num}.pth')
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