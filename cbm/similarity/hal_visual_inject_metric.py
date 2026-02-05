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

idx=-1

input_data_name_list={
    "": '../hal/origincorrect_hal_ce_val_full_v1.pth',
    # "original": '../hal/correctvisual_prompt_hal_ce_val_full_v1.pth',
    # "refer_correct": '../hal/correctvisual_prompt-1_hal_ce_val_full_v1.pth',
    # "refer_wrong": '../hal/correctvisual_prompt-2_hal_ce_val_full_v1.pth',
}
base_input_data_name_list={
    "":'../hal/origincorrect_hal_ce_val_image_full_v1.pth',
    # "original":'../hal/correctvisual_prompt_hal_ce_val_image_full_v1.pth',
    # "refer_correct":'../hal/correctvisual_prompt-1_hal_ce_val_image_full_v1.pth',
    # "refer_wrong":'../hal/correctvisual_prompt-2_hal_ce_val_image_full_v1.pth',
}

yolo_missing=[32, 34, 130, 266, 474, 517, 585, 655, 703, 846, 993, 1073, 1124, 1346, 1388, 1511, 1547, 1589, 1622, 1741, 1874, 1885, 1914, 1989, 2019, 2021, 2024, 2151, 2175, 2260, 2265, 2403, 2407, 2528, 2573, 2622, 2798, 2803, 2950, 3009, 3190, 3331, 3430, 3464, 3538, 3565, 3570, 3587, 3589, 3591, 3602, 3605, 3727, 3729, 3732, 3801, 3836, 3840, 3843, 3946, 3965, 4075, 4117, 4243, 4350, 4353, 4356, 4424, 4454, 4455, 4484, 4603, 4612, 4710, 4731, 4852, 4854]
visual_missing_basic=[20, 25, 32, 33, 34, 46, 53, 55, 57, 59, 63, 97, 98, 130, 147, 154, 155, 156, 160, 162, 166, 169, 174, 178, 179, 180, 183, 187, 190, 202, 208, 210, 211, 234, 239, 259, 263, 266, 276, 278, 280, 281, 282, 283, 284, 289, 290, 293, 296, 298, 301, 309, 310, 313, 331, 356, 358, 360, 366, 376, 377, 387, 388, 402, 405, 407, 408, 409, 410, 411, 413, 416, 438, 439, 442, 445, 446, 457, 479, 481, 504, 517, 519, 521, 523, 534, 546, 548, 549, 554, 557, 559, 560, 564, 565, 568, 570, 577, 580, 584, 585, 594, 639, 641, 646, 655, 657, 663, 665, 670, 672, 673, 678, 682, 683, 684, 685, 686, 692, 702, 703, 710, 714, 716, 717, 748, 761, 767, 784, 803, 805, 810, 814, 816, 817, 818, 819, 821, 824, 835, 836, 846, 865, 876, 885, 890, 894, 903, 905, 906, 907, 913, 914, 920, 922, 932, 939, 940, 943, 945, 947, 956, 973, 977, 978, 991, 993, 1001, 1021, 1024, 1027, 1028, 1031, 1032, 1033, 1034, 1037, 1042, 1044, 1045, 1046, 1048, 1066, 1067, 1073, 1085, 1086, 1088, 1090, 1106, 1120, 1124, 1133, 1135, 1139, 1140, 1141, 1143, 1145, 1154, 1156, 1161, 1163, 1164, 1168, 1185, 1234, 1249, 1255, 1256, 1260, 1263, 1265, 1267, 1273, 1281, 1293, 1296, 1299, 1309, 1318, 1331, 1343, 1346, 1360, 1362, 1366, 1370, 1372, 1377, 1381, 1383, 1387, 1388, 1393, 1406, 1419, 1427, 1431, 1437, 1440, 1456, 1460, 1465, 1467, 1470, 1479, 1483, 1491, 1494, 1495, 1501, 1503, 1507, 1511, 1529, 1539, 1545, 1547, 1553, 1583, 1589, 1593, 1595, 1602, 1619, 1622, 1629, 1643, 1649, 1661, 1686, 1711, 1712, 1722, 1733, 1735, 1741, 1743, 1749, 1753, 1754, 1757, 1758, 1759, 1760, 1764, 1766, 1780, 1803, 1822, 1825, 1827, 1830, 1853, 1854, 1860, 1864, 1867, 1870, 1877, 1878, 1880, 1885, 1888, 1890, 1898, 1904, 1911, 1914, 1917, 1921, 1923, 1924, 1933, 1939, 1943, 1945, 1946, 1962, 1979, 1980, 1982, 1984, 1989, 2013, 2014, 2017, 2019, 2021, 2024, 2045, 2063, 2072, 2104, 2110, 2117, 2136, 2137, 2141, 2147, 2151, 2153, 2159, 2163, 2164, 2165, 2169, 2175, 2176, 2177, 2182, 2189, 2205, 2230, 2235, 2238, 2239, 2260, 2265, 2271, 2295, 2304, 2319, 2330, 2342, 2354, 2357, 2359, 2362, 2368, 2376, 2379, 2380, 2381, 2382, 2383, 2385, 2386, 2387, 2393, 2397, 2398, 2399, 2401, 2403, 2404, 2405, 2407, 2411, 2427, 2431, 2434, 2441, 2449, 2450, 2483, 2488, 2489, 2510, 2514, 2518, 2522, 2523, 2524, 2525, 2527, 2528, 2538, 2539, 2553, 2570, 2573, 2576, 2581, 2582, 2601, 2603, 2607, 2620, 2628, 2631, 2632, 2638, 2639, 2641, 2644, 2653, 2654, 2655, 2657, 2662, 2672, 2679, 2686, 2695, 2697, 2743, 2747, 2751, 2753, 2767, 2770, 2781, 2784, 2787, 2788, 2790, 2792, 2794, 2797, 2798, 2799, 2800, 2801, 2803, 2807, 2834, 2838, 2842, 2852, 2858, 2860, 2863, 2878, 2887, 2889, 2898, 2905, 2909, 2924, 2932, 2933, 2934, 2937, 2947, 2950, 2990, 2993, 2996, 2999, 3002, 3004, 3009, 3012, 3013, 3014, 3016, 3021, 3025, 3027, 3029, 3035, 3036, 3039, 3047, 3060, 3061, 3063, 3086, 3088, 3102, 3114, 3131, 3137, 3140, 3147, 3148, 3153, 3157, 3160, 3171, 3176, 3182, 3187, 3189, 3190, 3196, 3206, 3210, 3215, 3229, 3232, 3233, 3272, 3288, 3295, 3296, 3306, 3312, 3318, 3320, 3321, 3330, 3331, 3333, 3334, 3342, 3343, 3348, 3351, 3352, 3353, 3385, 3398, 3420, 3434, 3438, 3439, 3440, 3442, 3449, 3464, 3470, 3485, 3486, 3487, 3489, 3523, 3538, 3541, 3542, 3543, 3548, 3549, 3567, 3568, 3570, 3572, 3573, 3587, 3589, 3591, 3592, 3594, 3596, 3597, 3598, 3599, 3602, 3605, 3655, 3657, 3681, 3688, 3695, 3697, 3717, 3721, 3722, 3723, 3726, 3727, 3729, 3732, 3735, 3752, 3756, 3760, 3762, 3780, 3786, 3787, 3789, 3801, 3816, 3818, 3819, 3826, 3831, 3834, 3836, 3837, 3840, 3841, 3842, 3843, 3844, 3845, 3846, 3862, 3865, 3869, 3874, 3875, 3902, 3903, 3910, 3911, 3927, 3946, 3948, 3965, 3966, 3970, 3979, 3997, 3999, 4001, 4005, 4006, 4015, 4019, 4024, 4045, 4048, 4057, 4059, 4071, 4072, 4074, 4075, 4081, 4084, 4091, 4093, 4097, 4104, 4109, 4110, 4112, 4114, 4115, 4117, 4118, 4134, 4147, 4159, 4162, 4165, 4167, 4187, 4194, 4211, 4216, 4217, 4218, 4220, 4224, 4228, 4234, 4237, 4243, 4249, 4256, 4272, 4287, 4289, 4309, 4310, 4320, 4325, 4327, 4330, 4333, 4339, 4341, 4344, 4346, 4350, 4351, 4353, 4356, 4358, 4359, 4361, 4373, 4391, 4398, 4404, 4408, 4424, 4425, 4439, 4442, 4454, 4455, 4457, 4459, 4463, 4464, 4465, 4470, 4473, 4482, 4484, 4488, 4489, 4495, 4501, 4505, 4511, 4518, 4532, 4546, 4579, 4582, 4598, 4603, 4608, 4612, 4622, 4623, 4628, 4635, 4645, 4687, 4689, 4698, 4706, 4709, 4710, 4715, 4717, 4720, 4723, 4728, 4729, 4731, 4733, 4735, 4738, 4740, 4741, 4742, 4745, 4746, 4749, 4753, 4755, 4756, 4757, 4773, 4781, 4810, 4818, 4827, 4830, 4838, 4846, 4851, 4852, 4854, 4857, 4862, 4865, 4866, 4870, 4871, 4874, 4876, 4891, 4900, 4923, 4928, 4937, 4938, 4939, 4943, 4944, 4947, 4948, 4949, 4953, 4958, 4960, 4961, 4964, 4990, 4992, 4993]
visual_missing=list(set(yolo_missing) | set(visual_missing_basic))

missing_digit_lists=[
    yolo_missing,
    visual_missing,
    visual_missing,
    visual_missing
]


for _, (input_data_name, base_input_data_name) in enumerate(zip(input_data_name_list.values(), base_input_data_name_list.values())):
    
    if idx!=-1:
        input_data_name=list(input_data_name_list.values())[idx]
        base_input_data_name=list(base_input_data_name_list.values())[idx]
    
    print("processing:", input_data_name, base_input_data_name)
    
    input_datas = torch.load(input_data_name)
    base_input_datas = torch.load(base_input_data_name)
    
    if missing_digit_lists[idx]:
        for i in range(len(input_datas)):
            input_datas[i]=[input_datas[i][j] for j in range(len(input_datas[i])) if j not in missing_digit_lists[idx]]
            base_input_datas[i]=[base_input_datas[i][j] for j in range(len(base_input_datas[i])) if j not in missing_digit_lists[idx]]

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

    torch.save({'scores': scores, 'base_scores': base_scores}, f'visual/hal_{list(input_data_name_list.keys())[idx]}_scores_data.pth')
    print("save to ", f'visual/hal_{list(input_data_name_list.keys())[idx]}_scores_data.pth')
    if idx!=-1:
        break