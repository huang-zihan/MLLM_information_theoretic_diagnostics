import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn.functional as F
import matplotlib.pyplot as plt

is_qwen=True

if is_qwen:
    info_save_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
else:
    info_save_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]    

use_avg_prior=False
seperate_layer=False
# epoch=19
epoch=25

QUESTION_ONLY=False
FIXED_QUESTION=False
avg_token=False
origin=False

if QUESTION_ONLY:
    prefix="question_only_"
elif origin:
    if not avg_token:
        prefix="origin"
    else:
        prefix="originavg_"
elif FIXED_QUESTION:
    prefix="fixed_question_"
elif avg_token:
    prefix="newavg_"
else:
    prefix=""

# question = ""
# visual_prompt-1
question="visual_prompt"
# question="visual_prompt-1" # 2791
# question="visual_prompt-2" # 8922

for text_margin in [True, False]:
    # text_margin=False
    for correct_answer in [True, False]: # , False
        # correct_answer=False
        # 4/4
        ce_data_dir="../cbm/hal/"
        # labels = torch.load(ce_data_dir+'hal_ce_val_label_full_v1.pth')
        # input_datas = torch.load(ce_data_dir+'hal_ce_val_full_v1.pth')

        if correct_answer:
            # if QUESTION_ONLY or FIXED_QUESTION:
            labels = torch.load(ce_data_dir+f'{prefix}correct{question}_hal_ce_val_label_full_v1.pth')
            print("load from:", ce_data_dir+f'{prefix}correct{question}_hal_ce_val_label_full_v1.pth')
            
            # else:
            #     labels = torch.load(ce_data_dir+f'correct_hal_ce_val_label_full_v1.pth')

            input_datas = torch.load(ce_data_dir+f'{prefix}correct{question}_hal_ce_val_full_v1.pth')
            print(ce_data_dir+f'{prefix}correct{question}_hal_ce_val_full_v1.pth')
            # input_datas = torch.load(ce_data_dir+f'{prefix}correct_info_save_list_hal_ce_val_full_v1.pth')
            # print(ce_data_dir+f'{prefix}correct_info_save_list_hal_ce_val_full_v1.pth')
            visual_missing = [20, 25, 32, 33, 34, 46, 53, 55, 57, 59, 63, 97, 98, 130, 147, 154, 155, 156, 160, 162, 166, 169, 174, 178, 179, 180, 183, 187, 190, 202, 208, 210, 211, 234, 239, 259, 263, 266, 276, 278, 280, 281, 282, 283, 284, 289, 290, 293, 296, 298, 301, 309, 310, 313, 331, 356, 358, 360, 366, 376, 377, 387, 388, 402, 405, 407, 408, 409, 410, 411, 413, 416, 438, 439, 442, 445, 446, 457, 479, 481, 504, 517, 519, 521, 523, 534, 546, 548, 549, 554, 557, 559, 560, 564, 565, 568, 570, 577, 580, 584, 585, 594, 639, 641, 646, 655, 657, 663, 665, 670, 672, 673, 678, 682, 683, 684, 685, 686, 692, 702, 703, 710, 714, 716, 717, 748, 761, 767, 784, 803, 805, 810, 814, 816, 817, 818, 819, 821, 824, 835, 836, 846, 865, 876, 885, 890, 894, 903, 905, 906, 907, 913, 914, 920, 922, 932, 939, 940, 943, 945, 947, 956, 973, 977, 978, 991, 993, 1001, 1021, 1024, 1027, 1028, 1031, 1032, 1033, 1034, 1037, 1042, 1044, 1045, 1046, 1048, 1066, 1067, 1073, 1085, 1086, 1088, 1090, 1106, 1120, 1124, 1133, 1135, 1139, 1140, 1141, 1143, 1145, 1154, 1156, 1161, 1163, 1164, 1168, 1185, 1234, 1249, 1255, 1256, 1260, 1263, 1265, 1267, 1273, 1281, 1293, 1296, 1299, 1309, 1318, 1331, 1343, 1346, 1360, 1362, 1366, 1370, 1372, 1377, 1381, 1383, 1387, 1388, 1393, 1406, 1419, 1427, 1431, 1437, 1440, 1456, 1460, 1465, 1467, 1470, 1479, 1483, 1491, 1494, 1495, 1501, 1503, 1507, 1511, 1529, 1539, 1545, 1547, 1553, 1583, 1589, 1593, 1595, 1602, 1619, 1622, 1629, 1643, 1649, 1661, 1686, 1711, 1712, 1722, 1733, 1735, 1741, 1743, 1749, 1753, 1754, 1757, 1758, 1759, 1760, 1764, 1766, 1780, 1803, 1822, 1825, 1827, 1830, 1853, 1854, 1860, 1864, 1867, 1870, 1877, 1878, 1880, 1885, 1888, 1890, 1898, 1904, 1911, 1914, 1917, 1921, 1923, 1924, 1933, 1939, 1943, 1945, 1946, 1962, 1979, 1980, 1982, 1984, 1989, 2013, 2014, 2017, 2019, 2021, 2024, 2045, 2063, 2072, 2104, 2110, 2117, 2136, 2137, 2141, 2147, 2151, 2153, 2159, 2163, 2164, 2165, 2169, 2175, 2176, 2177, 2182, 2189, 2205, 2230, 2235, 2238, 2239, 2260, 2265, 2271, 2295, 2304, 2319, 2330, 2342, 2354, 2357, 2359, 2362, 2368, 2376, 2379, 2380, 2381, 2382, 2383, 2385, 2386, 2387, 2393, 2397, 2398, 2399, 2401, 2403, 2404, 2405, 2407, 2411, 2427, 2431, 2434, 2441, 2449, 2450, 2483, 2488, 2489, 2510, 2514, 2518, 2522, 2523, 2524, 2525, 2527, 2528, 2538, 2539, 2553, 2570, 2573, 2576, 2581, 2582, 2601, 2603, 2607, 2620, 2628, 2631, 2632, 2638, 2639, 2641, 2644, 2653, 2654, 2655, 2657, 2662, 2672, 2679, 2686, 2695, 2697, 2743, 2747, 2751, 2753, 2767, 2770, 2781, 2784, 2787, 2788, 2790, 2792, 2794, 2797, 2798, 2799, 2800, 2801, 2803, 2807, 2834, 2838, 2842, 2852, 2858, 2860, 2863, 2878, 2887, 2889, 2898, 2905, 2909, 2924, 2932, 2933, 2934, 2937, 2947, 2950, 2990, 2993, 2996, 2999, 3002, 3004, 3009, 3012, 3013, 3014, 3016, 3021, 3025, 3027, 3029, 3035, 3036, 3039, 3047, 3060, 3061, 3063, 3086, 3088, 3102, 3114, 3131, 3137, 3140, 3147, 3148, 3153, 3157, 3160, 3171, 3176, 3182, 3187, 3189, 3190, 3196, 3206, 3210, 3215, 3229, 3232, 3233, 3272, 3288, 3295, 3296, 3306, 3312, 3318, 3320, 3321, 3330, 3331, 3333, 3334, 3342, 3343, 3348, 3351, 3352, 3353, 3385, 3398, 3420, 3434, 3438, 3439, 3440, 3442, 3449, 3464, 3470, 3485, 3486, 3487, 3489, 3523, 3538, 3541, 3542, 3543, 3548, 3549, 3567, 3568, 3570, 3572, 3573, 3587, 3589, 3591, 3592, 3594, 3596, 3597, 3598, 3599, 3602, 3605, 3655, 3657, 3681, 3688, 3695, 3697, 3717, 3721, 3722, 3723, 3726, 3727, 3729, 3732, 3735, 3752, 3756, 3760, 3762, 3780, 3786, 3787, 3789, 3801, 3816, 3818, 3819, 3826, 3831, 3834, 3836, 3837, 3840, 3841, 3842, 3843, 3844, 3845, 3846, 3862, 3865, 3869, 3874, 3875, 3902, 3903, 3910, 3911, 3927, 3946, 3948, 3965, 3966, 3970, 3979, 3997, 3999, 4001, 4005, 4006, 4015, 4019, 4024, 4045, 4048, 4057, 4059, 4071, 4072, 4074, 4075, 4081, 4084, 4091, 4093, 4097, 4104, 4109, 4110, 4112, 4114, 4115, 4117, 4118, 4134, 4147, 4159, 4162, 4165, 4167, 4187, 4194, 4211, 4216, 4217, 4218, 4220, 4224, 4228, 4234, 4237, 4243, 4249, 4256, 4272, 4287, 4289, 4309, 4310, 4320, 4325, 4327, 4330, 4333, 4339, 4341, 4344, 4346, 4350, 4351, 4353, 4356, 4358, 4359, 4361, 4373, 4391, 4398, 4404, 4408, 4424, 4425, 4439, 4442, 4454, 4455, 4457, 4459, 4463, 4464, 4465, 4470, 4473, 4482, 4484, 4488, 4489, 4495, 4501, 4505, 4511, 4518, 4532, 4546, 4579, 4582, 4598, 4603, 4608, 4612, 4622, 4623, 4628, 4635, 4645, 4687, 4689, 4698, 4706, 4709, 4710, 4715, 4717, 4720, 4723, 4728, 4729, 4731, 4733, 4735, 4738, 4740, 4741, 4742, 4745, 4746, 4749, 4753, 4755, 4756, 4757, 4773, 4781, 4810, 4818, 4827, 4830, 4838, 4846, 4851, 4852, 4854, 4857, 4862, 4865, 4866, 4870, 4871, 4874, 4876, 4891, 4900, 4923, 4928, 4937, 4938, 4939, 4943, 4944, 4947, 4948, 4949, 4953, 4958, 4960, 4961, 4964, 4990, 4992, 4993]
            missing_index = [32, 34, 130, 266, 474, 517, 585, 655, 703, 846, 993, 1073, 1124, 1346, 1388, 1511, 1547, 1589, 1622, 1741, 1874, 1885, 1914, 1989, 2019, 2021, 2024, 2151, 2175, 2260, 2265, 2403, 2407, 2528, 2573, 2622, 2798, 2803, 2950, 3009, 3190, 3331, 3430, 3464, 3538, 3565, 3570, 3587, 3589, 3591, 3602, 3605, 3727, 3729, 3732, 3801, 3836, 3840, 3843, 3946, 3965, 4075, 4117, 4243, 4350, 4353, 4356, 4424, 4454, 4455, 4484, 4603, 4612, 4710, 4731, 4852, 4854]
            missing_index = list(set(visual_missing) | set(missing_index))
        else:
            
            # if QUESTION_ONLY or FIXED_QUESTION:
            labels = torch.load(ce_data_dir+f'{prefix}wrong{question}_hal_ce_val_label_full_v1.pth')
            print(ce_data_dir+f'{prefix}wrong{question}_hal_ce_val_label_full_v1.pth')
            # else:
            #     labels = torch.load(ce_data_dir+f'wrong_hal_ce_val_label_full_v1.pth')
            
            input_datas = torch.load(ce_data_dir+f'{prefix}wrong{question}_hal_ce_val_full_v1.pth')
            print(ce_data_dir+f'{prefix}wrong{question}_hal_ce_val_full_v1.pth')
            # input_datas = torch.load(ce_data_dir+f'{prefix}wrong_info_save_list_hal_ce_val_full_v1.pth')
            # print(ce_data_dir+f'{prefix}wrong_info_save_list_hal_ce_val_full_v1.pth')
            missing_index = [128, 129, 130, 131, 136, 137, 138, 139, 520, 521, 522, 523, 1064, 1065, 1066, 1067, 1896, 1897, 1898, 1899, 2068, 2069, 2070, 2071, 2340, 2341, 2342, 2343, 2620, 2621, 2622, 2623, 2812, 2813, 2814, 2815, 3384, 3385, 3386, 3387, 3972, 3973, 3974, 3975, 4292, 4293, 4294, 4295, 4496, 4497, 4498, 4499, 5384, 5385, 5386, 5387, 5552, 5553, 5554, 5555, 6044, 6045, 6046, 6047, 6188, 6189, 6190, 6191, 6356, 6357, 6358, 6359, 6488, 6489, 6490, 6491, 6964, 6965, 6966, 6967, 7496, 7497, 7498, 7499, 7540, 7541, 7542, 7543, 7656, 7657, 7658, 7659, 7956, 7957, 7958, 7959, 8076, 8077, 8078, 8079, 8084, 8085, 8086, 8087, 8096, 8097, 8098, 8099, 8604, 8605, 8606, 8607, 8700, 8701, 8702, 8703, 9040, 9041, 9042, 9043, 9060, 9061, 9062, 9063, 9612, 9613, 9614, 9615, 9628, 9629, 9630, 9631, 10112, 10113, 10114, 10115, 10292, 10293, 10294, 10295, 10488, 10489, 10490, 10491, 11192, 11193, 11194, 11195, 11212, 11213, 11214, 11215, 11800, 11801, 11802, 11803, 12036, 12037, 12038, 12039, 12760, 12761, 12762, 12763, 13324, 13325, 13326, 13327, 13720, 13721, 13722, 13723, 13856, 13857, 13858, 13859, 14152, 14153, 14154, 14155, 14260, 14261, 14262, 14263, 14280, 14281, 14282, 14283, 14348, 14349, 14350, 14351, 14356, 14357, 14358, 14359, 14364, 14365, 14366, 14367, 14408, 14409, 14410, 14411, 14420, 14421, 14422, 14423, 14908, 14909, 14910, 14911, 14916, 14917, 14918, 14919, 14928, 14929, 14930, 14931, 15204, 15205, 15206, 15207, 15344, 15345, 15346, 15347, 15360, 15361, 15362, 15363, 15372, 15373, 15374, 15375, 15784, 15785, 15786, 15787, 15860, 15861, 15862, 15863, 16300, 16301, 16302, 16303, 16468, 16469, 16470, 16471, 16972, 16973, 16974, 16975, 17400, 17401, 17402, 17403, 17412, 17413, 17414, 17415, 17424, 17425, 17426, 17427, 17696, 17697, 17698, 17699, 17816, 17817, 17818, 17819, 17820, 17821, 17822, 17823, 17936, 17937, 17938, 17939, 18412, 18413, 18414, 18415, 18448, 18449, 18450, 18451, 18840, 18841, 18842, 18843, 18924, 18925, 18926, 18927, 19408, 19409, 19410, 19411, 19416, 19417, 19418, 19419]
            
        # missing_index = [128, 129, 130, 131, 136, 137, 138, 139, 520, 521, 522, 523, 1064, 1065, 1066, 1067, 1896, 1897, 1898, 1899, 2068, 2069, 2070, 2071, 2340, 2341, 2342, 2343, 2620, 2621, 2622, 2623, 2812, 2813, 2814, 2815, 3384, 3385, 3386, 3387, 3972, 3973, 3974, 3975, 4292, 4293, 4294, 4295, 4496, 4497, 4498, 4499, 5384, 5385, 5386, 5387, 5552, 5553, 5554, 5555, 6044, 6045, 6046, 6047, 6188, 6189, 6190, 6191, 6356, 6357, 6358, 6359, 6488, 6489, 6490, 6491, 6964, 6965, 6966, 6967, 7496, 7497, 7498, 7499, 7540, 7541, 7542, 7543, 7656, 7657, 7658, 7659, 7956, 7957, 7958, 7959, 8076, 8077, 8078, 8079, 8084, 8085, 8086, 8087, 8096, 8097, 8098, 8099, 8604, 8605, 8606, 8607, 8700, 8701, 8702, 8703, 9040, 9041, 9042, 9043, 9060, 9061, 9062, 9063, 9612, 9613, 9614, 9615, 9628, 9629, 9630, 9631, 10112, 10113, 10114, 10115, 10292, 10293, 10294, 10295, 10488, 10489, 10490, 10491, 11192, 11193, 11194, 11195, 11212, 11213, 11214, 11215, 11800, 11801, 11802, 11803, 12036, 12037, 12038, 12039, 12760, 12761, 12762, 12763, 13324, 13325, 13326, 13327, 13720, 13721, 13722, 13723, 13856, 13857, 13858, 13859, 14152, 14153, 14154, 14155, 14260, 14261, 14262, 14263, 14280, 14281, 14282, 14283, 14348, 14349, 14350, 14351, 14356, 14357, 14358, 14359, 14364, 14365, 14366, 14367, 14408, 14409, 14410, 14411, 14420, 14421, 14422, 14423, 14908, 14909, 14910, 14911, 14916, 14917, 14918, 14919, 14928, 14929, 14930, 14931, 15204, 15205, 15206, 15207, 15344, 15345, 15346, 15347, 15360, 15361, 15362, 15363, 15372, 15373, 15374, 15375, 15784, 15785, 15786, 15787, 15860, 15861, 15862, 15863, 16300, 16301, 16302, 16303, 16468, 16469, 16470, 16471, 16972, 16973, 16974, 16975, 17400, 17401, 17402, 17403, 17412, 17413, 17414, 17415, 17424, 17425, 17426, 17427, 17696, 17697, 17698, 17699, 17816, 17817, 17818, 17819, 17820, 17821, 17822, 17823, 17936, 17937, 17938, 17939, 18412, 18413, 18414, 18415, 18448, 18449, 18450, 18451, 18840, 18841, 18842, 18843, 18924, 18925, 18926, 18927, 19408, 19409, 19410, 19411, 19416, 19417, 19418, 19419]

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

        # epoch=49


        for margin_num in margin_num_list:
            
            temp_prefix = prefix+("correct_" if correct_answer else "wrong_")
            
            if is_qwen:
                temp_prefix ="qwen_"+temp_prefix
            
            if text_margin:
                written_file=f'{ce_data_dir}/{temp_prefix}{question}hal_all_kls_text_margin_{margin_num}_trainepoch{epoch}_v1'
                pic_file=f'{ce_data_dir}/{temp_prefix}{question}hal_line_chart_text_margin_{margin_num}_trainepoch{epoch}_v1'
            else:
                written_file=f'{ce_data_dir}/{temp_prefix}{question}hal_all_kls_merge_trainepoch{epoch}_v1'
                pic_file=f'{ce_data_dir}/{temp_prefix}{question}hal_line_chart_merge_trainepoch{epoch}_v1'
                
                
                if seperate_layer:
                    pic_file=f'{ce_data_dir}/{temp_prefix}{question}hal_line_chart_seperatelayer_trainepoch{epoch}_v1'

            if use_avg_prior:
                written_file+="_avg"

            written_file+=".txt"
            pic_file+=".pdf"

            if text_margin:
                index_to_cluster=torch.load(f'margin/{("correct_" if correct_answer else "wrong_")}hal_index_to_cluster-{margin_num}.pth', weights_only=False)
                seperate_kl=[[] for _ in range(margin_num)]


            vit=False
            one_item=False
            class CE(nn.Module):
                def __init__(self):
                    super(CE, self).__init__()

                    self.bn = nn.BatchNorm1d(2048 if is_qwen else 4096)
            
                    if is_qwen:
                        self.fc1 = nn.Linear(2048, 512)
                    else:
                        self.fc1 = nn.Linear(4096, 512)  # 假设每个 tensor 是 1024 维
                    self.fc2 = nn.Linear(512, 128)
                    self.fc3 = nn.Linear(128, len(labels[0]))  # 10个分类
                    # print(len(labels[0]))

                def forward(self, x):
                    x = self.bn(x)
                    
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
            # print(all_kls)
            # 将列表保存到本地文件
            print("written to", written_file)
            with open(written_file, 'w') as f:
                for item in all_kls:
                    f.write(f"{item}\n")

            # # 绘制折线图
            # plt.plot(all_kls, marker='o')
            # if not text_margin:
            #     plt.title('Upper bound of Unconditional MI Proxy')
            #     plt.xlabel('layer')
            #     plt.ylabel('metric value')
            # else:
            #     plt.title('Upper bound of Conditional MI Proxy')
            #     plt.xlabel('layer')
            #     plt.ylabel('metric value')
            # plt.grid()
            # plt.savefig(pic_file, format='pdf')  # 保存图表为图片
            # # plt.show()  # 显示图表    
            # plt.clf()
            
            # if text_margin==False:
            #     break