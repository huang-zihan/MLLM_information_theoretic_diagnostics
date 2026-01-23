import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datasets import load_dataset
import torch

###################
# test_data="vqa"
# test_data="coco_caption"


# test_data="hal"
test_data="aokvqa"
# test_data="vqa_origin"
correct_answer=True
QUESTION_ONLY=False

prefix="" if not QUESTION_ONLY else "question_only_"


###################

# test_data="chair"
# test_data="outdom"
# test_data="mislead"

#!! no missing data hal aokvqa

if test_data=="hal" or test_data=="chair" or test_data=="coco_caption" or test_data=="aokvqa" or test_data=="vqa_origin":
    cbm_path='/home/junda/zihan/cbm/'
    
    if test_data == "hal":
        
        if correct_answer:
            missing_index = [32, 34, 130, 266, 474, 517, 585, 655, 703, 846, 993, 1073, 1124, 1346, 1388, 1511, 1547, 1589, 1622, 1741, 1874, 1885, 1914, 1989, 2019, 2021, 2024, 2151, 2175, 2260, 2265, 2403, 2407, 2528, 2573, 2622, 2798, 2803, 2950, 3009, 3190, 3331, 3430, 3464, 3538, 3565, 3570, 3587, 3589, 3591, 3602, 3605, 3727, 3729, 3732, 3801, 3836, 3840, 3843, 3946, 3965, 4075, 4117, 4243, 4350, 4353, 4356, 4424, 4454, 4455, 4484, 4603, 4612, 4710, 4731, 4852, 4854]
            input_datas = torch.load(f'hal/{prefix}correct_hal_ce_val_question_full_v1.pth')
            #[128, 129, 130, 131, 136, 137, 138, 139, 520, 521, 522, 523, 1064, 1065, 1066, 1067, 1896, 1897, 1898, 1899, 2068, 2069, 2070, 2071, 2340, 2341, 2342, 2343, 2620, 2621, 2622, 2623, 2812, 2813, 2814, 2815, 3384, 3385, 3386, 3387, 3972, 3973, 3974, 3975, 4292, 4293, 4294, 4295, 4496, 4497, 4498, 4499, 5384, 5385, 5386, 5387, 5552, 5553, 5554, 5555, 6044, 6045, 6046, 6047, 6188, 6189, 6190, 6191, 6356, 6357, 6358, 6359, 6488, 6489, 6490, 6491, 6964, 6965, 6966, 6967, 7496, 7497, 7498, 7499, 7540, 7541, 7542, 7543, 7656, 7657, 7658, 7659, 7956, 7957, 7958, 7959, 8076, 8077, 8078, 8079, 8084, 8085, 8086, 8087, 8096, 8097, 8098, 8099, 8604, 8605, 8606, 8607, 8700, 8701, 8702, 8703, 9040, 9041, 9042, 9043, 9060, 9061, 9062, 9063, 9612, 9613, 9614, 9615, 9628, 9629, 9630, 9631, 10112, 10113, 10114, 10115, 10292, 10293, 10294, 10295, 10488, 10489, 10490, 10491, 11192, 11193, 11194, 11195, 11212, 11213, 11214, 11215, 11800, 11801, 11802, 11803, 12036, 12037, 12038, 12039, 12760, 12761, 12762, 12763, 13324, 13325, 13326, 13327, 13720, 13721, 13722, 13723, 13856, 13857, 13858, 13859, 14152, 14153, 14154, 14155, 14260, 14261, 14262, 14263, 14280, 14281, 14282, 14283, 14348, 14349, 14350, 14351, 14356, 14357, 14358, 14359, 14364, 14365, 14366, 14367, 14408, 14409, 14410, 14411, 14420, 14421, 14422, 14423, 14908, 14909, 14910, 14911, 14916, 14917, 14918, 14919, 14928, 14929, 14930, 14931, 15204, 15205, 15206, 15207, 15344, 15345, 15346, 15347, 15360, 15361, 15362, 15363, 15372, 15373, 15374, 15375, 15784, 15785, 15786, 15787, 15860, 15861, 15862, 15863, 16300, 16301, 16302, 16303, 16468, 16469, 16470, 16471, 16972, 16973, 16974, 16975, 17400, 17401, 17402, 17403, 17412, 17413, 17414, 17415, 17424, 17425, 17426, 17427, 17696, 17697, 17698, 17699, 17816, 17817, 17818, 17819, 17820, 17821, 17822, 17823, 17936, 17937, 17938, 17939, 18412, 18413, 18414, 18415, 18448, 18449, 18450, 18451, 18840, 18841, 18842, 18843, 18924, 18925, 18926, 18927, 19408, 19409, 19410, 19411, 19416, 19417, 19418, 19419]
        else:
            missing_index = [128, 129, 130, 131, 136, 137, 138, 139, 520, 521, 522, 523, 1064, 1065, 1066, 1067, 1896, 1897, 1898, 1899, 2068, 2069, 2070, 2071, 2340, 2341, 2342, 2343, 2620, 2621, 2622, 2623, 2812, 2813, 2814, 2815, 3384, 3385, 3386, 3387, 3972, 3973, 3974, 3975, 4292, 4293, 4294, 4295, 4496, 4497, 4498, 4499, 5384, 5385, 5386, 5387, 5552, 5553, 5554, 5555, 6044, 6045, 6046, 6047, 6188, 6189, 6190, 6191, 6356, 6357, 6358, 6359, 6488, 6489, 6490, 6491, 6964, 6965, 6966, 6967, 7496, 7497, 7498, 7499, 7540, 7541, 7542, 7543, 7656, 7657, 7658, 7659, 7956, 7957, 7958, 7959, 8076, 8077, 8078, 8079, 8084, 8085, 8086, 8087, 8096, 8097, 8098, 8099, 8604, 8605, 8606, 8607, 8700, 8701, 8702, 8703, 9040, 9041, 9042, 9043, 9060, 9061, 9062, 9063, 9612, 9613, 9614, 9615, 9628, 9629, 9630, 9631, 10112, 10113, 10114, 10115, 10292, 10293, 10294, 10295, 10488, 10489, 10490, 10491, 11192, 11193, 11194, 11195, 11212, 11213, 11214, 11215, 11800, 11801, 11802, 11803, 12036, 12037, 12038, 12039, 12760, 12761, 12762, 12763, 13324, 13325, 13326, 13327, 13720, 13721, 13722, 13723, 13856, 13857, 13858, 13859, 14152, 14153, 14154, 14155, 14260, 14261, 14262, 14263, 14280, 14281, 14282, 14283, 14348, 14349, 14350, 14351, 14356, 14357, 14358, 14359, 14364, 14365, 14366, 14367, 14408, 14409, 14410, 14411, 14420, 14421, 14422, 14423, 14908, 14909, 14910, 14911, 14916, 14917, 14918, 14919, 14928, 14929, 14930, 14931, 15204, 15205, 15206, 15207, 15344, 15345, 15346, 15347, 15360, 15361, 15362, 15363, 15372, 15373, 15374, 15375, 15784, 15785, 15786, 15787, 15860, 15861, 15862, 15863, 16300, 16301, 16302, 16303, 16468, 16469, 16470, 16471, 16972, 16973, 16974, 16975, 17400, 17401, 17402, 17403, 17412, 17413, 17414, 17415, 17424, 17425, 17426, 17427, 17696, 17697, 17698, 17699, 17816, 17817, 17818, 17819, 17820, 17821, 17822, 17823, 17936, 17937, 17938, 17939, 18412, 18413, 18414, 18415, 18448, 18449, 18450, 18451, 18840, 18841, 18842, 18843, 18924, 18925, 18926, 18927, 19408, 19409, 19410, 19411, 19416, 19417, 19418, 19419]
            input_datas = torch.load(f'hal/{prefix}wrong_hal_ce_val_question_full_v1.pth')
        sentences = [input_datas[i] for i in range(len(input_datas)) if i not in missing_index]
        # sentences = input_datas
        # print(sentences[:10])
        # exit()
    elif test_data == "aokvqa":
        if correct_answer:
            missing_index = [795, 875, 976, 1074]
            input_datas = torch.load(f'aokvqa/{prefix}correct_aokvqa_question_list_v1.pth')
        else:
            missing_index = [2385, 2386, 2387, 2625, 2626, 2627, 2928, 2929, 2930, 3222, 3223, 3224]
            input_datas = torch.load(f'aokvqa/{prefix}wrong_aokvqa_question_list_v1.pth')

        sentences = [input_datas[i] for i in range(len(input_datas)) if i not in missing_index]
    
    elif test_data == "chair":
        missing_index = [12, 13, 26, 27, 58, 59, 60, 61, 62, 63, 78, 79, 92, 93, 112, 113, 118, 119, 126, 127, 132, 133, 146, 147, 152, 153, 180, 181, 214, 215, 218, 219, 220, 221, 228, 229, 246, 247, 254, 255, 260, 261, 268, 269, 290, 291, 300, 301, 316, 317, 346, 347, 402, 403, 410, 411, 448, 449, 460, 461, 464, 465, 486, 487, 494, 495, 496, 497, 514, 515, 550, 551, 554, 555, 576, 577, 592, 593, 594, 595, 596, 597, 602, 603, 604, 605, 618, 619, 668, 669, 726, 727, 736, 737, 754, 755, 832, 833, 854, 855, 868, 869, 878, 879, 882, 883, 888, 889, 894, 895, 918, 919, 930, 931, 942, 943, 946, 947, 954, 955, 958, 959, 972, 973, 976, 977, 978, 979, 998, 999, 1008, 1009, 1022, 1023, 1034, 1035, 1116, 1117, 1138, 1139, 1158, 1159, 1162, 1163, 1182, 1183, 1192, 1193, 1194, 1195, 1212, 1213, 1240, 1241, 1254, 1255, 1282, 1283, 1306, 1307, 1312, 1313, 1332, 1333, 1374, 1375, 1380, 1381, 1392, 1393, 1426, 1427, 1436, 1437, 1454, 1455, 1472, 1473, 1506, 1507, 1528, 1529, 1546, 1547, 1570, 1571, 1582, 1583, 1602, 1603, 1604, 1605, 1626, 1627, 1662, 1663, 1698, 1699, 1702, 1703, 1704, 1705, 1710, 1711, 1718, 1719, 1736, 1737, 1768, 1769, 1772, 1773, 1774, 1775, 1798, 1799, 1832, 1833, 1866, 1867, 1874, 1875, 1930, 1931, 1932, 1933, 1934, 1935, 1940, 1941, 1974, 1975, 1988, 1989, 2046, 2047, 2064, 2065, 2076, 2077, 2086, 2087, 2096, 2097, 2102, 2103, 2110, 2111, 2166, 2167, 2190, 2191, 2198, 2199, 2248, 2249, 2282, 2283, 2288, 2289, 2292, 2293, 2326, 2327, 2346, 2347, 2354, 2355, 2362, 2363, 2410, 2411, 2454, 2455, 2490, 2491, 2492, 2493, 2502, 2503, 2504, 2505, 2526, 2527, 2532, 2533, 2552, 2553, 2570, 2571, 2600, 2601, 2604, 2605, 2638, 2639, 2656, 2657, 2668, 2669, 2680, 2681, 2708, 2709, 2742, 2743, 2754, 2755, 2774, 2775, 2816, 2817, 2840, 2841, 2848, 2849, 2880, 2881, 2884, 2885, 2898, 2899, 2920, 2921, 2930, 2931, 2942, 2943, 2944, 2945, 2976, 2977, 3012, 3013, 3046, 3047, 3126, 3127, 3142, 3143, 3148, 3149, 3160, 3161, 3178, 3179, 3192, 3193, 3194, 3195, 3204, 3205, 3212, 3213, 3228, 3229, 3242, 3243, 3314, 3315, 3318, 3319, 3328, 3329, 3340, 3341, 3356, 3357, 3366, 3367, 3370, 3371, 3372, 3373, 3380, 3381, 3422, 3423, 3452, 3453, 3464, 3465, 3476, 3477, 3512, 3513, 3520, 3521, 3536, 3537, 3552, 3553, 3578, 3579, 3582, 3583, 3590, 3591, 3594, 3595, 3632, 3633, 3666, 3667, 3684, 3685, 3708, 3709, 3718, 3719, 3734, 3735, 3788, 3789, 3838, 3839, 3842, 3843, 3872, 3873, 3878, 3879, 3880, 3881, 3890, 3891, 3930, 3931, 3934, 3935, 3936, 3937, 3938, 3939, 3956, 3957, 3984, 3985, 3992, 3993, 4000, 4001, 4012, 4013, 4026, 4027, 4032, 4033, 4042, 4043, 4056, 4057, 4078, 4079, 4094, 4095, 4144, 4145, 4152, 4153, 4216, 4217, 4230, 4231, 4234, 4235, 4258, 4259, 4260, 4261, 4264, 4265, 4294, 4295, 4310, 4311, 4324, 4325, 4328, 4329, 4352, 4353, 4368, 4369, 4372, 4373, 4380, 4381, 4392, 4393, 4418, 4419, 4432, 4433, 4442, 4443, 4474, 4475, 4510, 4511, 4534, 4535, 4552, 4553, 4564, 4565, 4586, 4587, 4628, 4629, 4638, 4639, 4672, 4673, 4690, 4691, 4726, 4727, 4734, 4735, 4738, 4739, 4758, 4759, 4766, 4767, 4800, 4801, 4808, 4809, 4830, 4831, 4836, 4837, 4872, 4873, 4950, 4951, 4966, 4967, 4994, 4995, 5012, 5013, 5022, 5023, 5040, 5041, 5070, 5071, 5076, 5077, 5096, 5097, 5106, 5107, 5134, 5135, 5148, 5149, 5158, 5159, 5160, 5161, 5208, 5209, 5244, 5245, 5254, 5255, 5264, 5265, 5280, 5281, 5282, 5283, 5288, 5289, 5294, 5295, 5306, 5307, 5312, 5313, 5314, 5315, 5340, 5341, 5356, 5357, 5362, 5363, 5372, 5373, 5384, 5385, 5398, 5399, 5416, 5417, 5420, 5421, 5424, 5425, 5430, 5431, 5442, 5443, 5452, 5453, 5460, 5461, 5474, 5475, 5478, 5479, 5508, 5509, 5514, 5515, 5546, 5547, 5548, 5549, 5556, 5557, 5558, 5559, 5628, 5629, 5634, 5635, 5654, 5655, 5688, 5689, 5694, 5695, 5708, 5709, 5710, 5711, 5716, 5717, 5720, 5721, 5722, 5723, 5730, 5731, 5764, 5765, 5784, 5785, 5826, 5827, 5834, 5835, 5838, 5839, 5846, 5847, 5848, 5849, 5866, 5867, 5880, 5881, 5934, 5935, 5952, 5953, 5992, 5993, 6004, 6005, 6088, 6089, 6098, 6099, 6102, 6103, 6112, 6113, 6148, 6149, 6158, 6159, 6174, 6175, 6208, 6209, 6214, 6215, 6218, 6219, 6230, 6231, 6250, 6251, 6264, 6265, 6292, 6293, 6326, 6327, 6366, 6367, 6372, 6373, 6380, 6381, 6390, 6391, 6392, 6393, 6398, 6399, 6422, 6423, 6424, 6425, 6436, 6437, 6458, 6459, 6510, 6511, 6526, 6527, 6542, 6543, 6562, 6563, 6582, 6583, 6614, 6615, 6618, 6619, 6694, 6695, 6702, 6703, 6706, 6707, 6722, 6723, 6728, 6729, 6734, 6735, 6752, 6753, 6768, 6769, 6770, 6771, 6788, 6789, 6802, 6803, 6826, 6827, 6828, 6829, 6848, 6849, 6862, 6863, 6890, 6891, 6896, 6897, 6938, 6939, 6966, 6967, 7004, 7005, 7006, 7007, 7072, 7073, 7084, 7085, 7148, 7149, 7204, 7205, 7218, 7219, 7224, 7225, 7228, 7229, 7252, 7253, 7256, 7257, 7264, 7265, 7296, 7297, 7298, 7299, 7310, 7311, 7336, 7337, 7348, 7349, 7376, 7377, 7378, 7379, 7380, 7381, 7394, 7395, 7406, 7407, 7408, 7409, 7416, 7417, 7450, 7451, 7452, 7453, 7476, 7477, 7488, 7489, 7490, 7491, 7492, 7493, 7528, 7529, 7544, 7545, 7550, 7551, 7554, 7555, 7580, 7581, 7586, 7587, 7606, 7607, 7618, 7619, 7630, 7631, 7632, 7633, 7646, 7647, 7650, 7651, 7668, 7669, 7690, 7691, 7698, 7699, 7706, 7707, 7722, 7723, 7740, 7741, 7748, 7749, 7790, 7791, 7800, 7801, 7812, 7813, 7820, 7821, 7822, 7823, 7842, 7843, 7848, 7849, 7852, 7853, 7902, 7903, 7934, 7935, 7940, 7941, 7942, 7943, 7962, 7963, 7996, 7997, 8012, 8013, 8014, 8015, 8016, 8017, 8024, 8025, 8036, 8037, 8068, 8069, 8080, 8081, 8086, 8087, 8098, 8099, 8106, 8107, 8114, 8115, 8164, 8165, 8174, 8175, 8218, 8219, 8222, 8223, 8282, 8283, 8290, 8291, 8306, 8307, 8324, 8325, 8332, 8333, 8378, 8379, 8394, 8395, 8410, 8411, 8418, 8419, 8422, 8423, 8426, 8427, 8446, 8447, 8450, 8451, 8452, 8453, 8460, 8461, 8480, 8481, 8484, 8485, 8522, 8523, 8546, 8547, 8554, 8555, 8568, 8569, 8572, 8573, 8580, 8581, 8592, 8593, 8598, 8599, 8608, 8609, 8630, 8631, 8636, 8637, 8654, 8655, 8670, 8671, 8678, 8679, 8688, 8689, 8714, 8715, 8746, 8747, 8760, 8761, 8764, 8765, 8798, 8799, 8804, 8805, 8814, 8815, 8836, 8837, 8838, 8839, 8868, 8869, 8874, 8875, 8894, 8895, 8904, 8905, 8918, 8919, 8950, 8951, 8952, 8953, 8958, 8959, 8968, 8969, 8978, 8979, 9064, 9065, 9072, 9073, 9076, 9077, 9092, 9093, 9106, 9107, 9108, 9109, 9118, 9119, 9148, 9149, 9208, 9209, 9250, 9251, 9316, 9317, 9336, 9337, 9342, 9343, 9352, 9353, 9404, 9405, 9434, 9435, 9452, 9453, 9454, 9455, 9488, 9489, 9508, 9509, 9534, 9535, 9542, 9543, 9560, 9561, 9604, 9605, 9608, 9609, 9610, 9611, 9616, 9617, 9624, 9625, 9638, 9639, 9652, 9653, 9656, 9657, 9698, 9699, 9700, 9701]
        input_datas = torch.load(cbm_path+'result/chair_feature/chair_ce_test_data_yolo11l.pt.pth')
        sentences=[item['question'] for i, item in enumerate(input_datas) if i not in missing_index]
    elif test_data == "coco_caption":
        missing_index = [73, 98, 188, 190, 232, 235, 258, 267, 275, 287, 297, 309, 316, 338, 365, 423, 470, 480, 483, 657, 1256, 1284, 1323, 1453, 1484, 1654, 1701, 1717, 1726, 1729, 1794, 1796, 1815, 1830, 1871, 1891, 1942, 1954, 2004, 2017, 2037, 2038, 2082, 2096, 2144, 2207, 2282, 2373, 2477, 2502, 2529, 2631, 2778, 2850, 2912, 2941, 2952, 2953, 3013, 3022, 3049, 3067, 3069, 3081, 3185, 3188, 3196, 3202, 3212, 3242, 3296, 3309, 3344, 3355, 3494, 3611, 3687, 3699, 3797, 3824, 3887, 3892, 3999, 4003, 4081, 4819, 4855, 4862, 4980, 5001, 5029, 5096, 5112, 5189, 5280, 5309, 5393, 5523, 5537, 5618, 5669, 5784, 5815, 5854, 5992, 6059, 6063, 6101, 6125, 6312, 6456, 6590, 6597, 6611, 6632, 6666, 6720, 6728, 6731, 6763, 7493, 7523, 7605, 7675, 7694, 7717, 7773, 7778, 7892, 7895, 8206, 8521, 8533, 8689, 8707, 8759, 8802, 8821, 8825, 8921, 8951, 9020, 9021, 9073, 9112, 9134, 9139, 9147, 9251, 9257, 9320, 9363, 9430, 9431, 9440, 9477, 9484, 9490, 9518, 9538, 9541, 9564, 9603, 9606, 9689, 9721, 9740, 9754, 9864, 9932, 9974, 9992]
        input_datas = torch.load(f'coco_caption/{prefix}coco_caption_ce_val_question_full_v1.pth')
        # sentences = input_datas
        # sentences
        sentences = [input_datas[i] for i in range(len(input_datas)) if i not in missing_index]
        
    elif test_data == "vqa_origin":
        missing_index=[]
        input_datas = torch.load(f'vqa/{prefix}vqa_ce_val_full_v1.pth')
        sentences = torch.load(f"vqa/{prefix}vqa_ce_val_question_full_v1.pth")
    
    
    vectorizer = TfidfVectorizer(stop_words='english')  # 去除停用词
    X = vectorizer.fit_transform(sentences)
    
    for num_clusters in [3,5,10]: # ,20,30,40,50,100
        # 2. 使用 K-Means 进行聚类
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)

        # 获取聚类标签
        labels = kmeans.labels_

        # 3. 可视化聚类结果（使用 PCA 降维）
        pca = PCA(n_components=2)  # 降维到 2D
        X_pca = pca.fit_transform(X.toarray())

        # 绘制聚类结果
        plt.figure(figsize=(8, 6))
        for i in range(num_clusters):
            plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f'Cluster {i+1}')
            
        # 检查数据数量
        print(f"Number of sentences: {len(sentences)}")

        # 检查聚类标签分布
        import numpy as np
        unique, counts = np.unique(labels, return_counts=True)
        print("Cluster distribution:", dict(zip(unique, counts)))

        # 4. 映射每个索引到对应的类别编号
        index_to_cluster = {index: label for index, label in enumerate(labels)}  # 类别编号从 1 开始

        # 输出每个索引及其对应的类别编号
        for index, cluster in index_to_cluster.items():
            print(f"Index: {index} -> Cluster: {cluster}")
        
        if test_data=="hal":
            if correct_answer:
                torch.save(index_to_cluster, f'margin/{prefix}correct_hal_index_to_cluster-{num_clusters}.pth')
                print(len(index_to_cluster), f'margin/{prefix}correct_hal_index_to_cluster-{num_clusters}.pth')
            else:
                torch.save(index_to_cluster, f'margin/{prefix}wrong_hal_index_to_cluster-{num_clusters}.pth')
                print(len(index_to_cluster), f'margin/{prefix}wrong_hal_index_to_cluster-{num_clusters}.pth')                
        elif test_data=="aokvqa":
            if correct_answer:
                torch.save(index_to_cluster, f'margin/{prefix}correct_aokvqa_index_to_cluster-{num_clusters}.pth')
                print(len(index_to_cluster), f'margin/{prefix}correct_aokvqa_index_to_cluster-{num_clusters}.pth')
            else:
                torch.save(index_to_cluster, f'margin/{prefix}wrong_aokvqa_index_to_cluster-{num_clusters}.pth')
                print(len(index_to_cluster), f'margin/{prefix}wrong_aokvqa_index_to_cluster-{num_clusters}.pth')                
        
        elif test_data == "chair":
            torch.save(index_to_cluster, f'margin/chair_index_to_cluster-{num_clusters}.pth')
            print(len(index_to_cluster), f'margin/chair_index_to_cluster-{num_clusters}.pth')
        elif test_data == "coco_caption":
            torch.save(index_to_cluster, f'margin/{prefix}coco_caption_index_to_cluster-{num_clusters}.pth')
            print(len(index_to_cluster), f'margin/{prefix}coco_caption_index_to_cluster-{num_clusters}.pth')

        elif test_data=="vqa_origin":
            torch.save(index_to_cluster, f'margin/{prefix}vqa_origin_index_to_cluster-{num_clusters}.pth')
            print(len(index_to_cluster), f'margin/{prefix}vqa_origin_index_to_cluster-{num_clusters}.pth')


    print(sentences[:5])
        
elif test_data=="vqa":
    missing_index = [68, 69, 70, 270, 271, 272, 273, 928, 929, 930, 1240, 1241, 1242, 1502, 1503, 1504, 3823, 3824, 3825, 3826, 3827, 3828, 3829, 3830, 3831, 3912, 3913, 3914, 3915, 3920, 3921, 3922, 3923, 3924, 4171, 4172, 4173, 4174, 4175, 4176, 4177, 5159, 5160, 5161, 5162, 5163, 5164, 5165, 5166, 5167, 5168, 5169, 5170, 5171, 5172, 5173, 5174, 5175, 5176, 5177, 5178, 5179, 5180, 5181, 5182, 5183, 5944, 5945, 5946, 5947, 6007, 6008, 6009, 6803, 6804, 6805, 6806, 6807, 6808, 6809, 6810, 6811, 6812, 7022, 7023, 7024, 7025, 7026, 7027, 7028, 7029, 7030, 7031, 7032, 7033, 7034, 7229, 7230, 7231, 7609, 7610, 7611, 7771, 7772, 7773, 7774, 8054, 8055, 8056, 8270, 8271, 8272, 8505, 8506, 8507, 8508, 8509, 8931, 8932, 8933, 8934, 8935, 9600, 9601, 9602, 9603, 9604, 9605, 9606, 9607, 9608, 9609, 9863, 9864, 9865, 9866, 9867, 9868, 9869, 9870, 9871, 9872, 9873, 9992, 9993, 9994, 9995, 9996, 9997, 9998, 9999]
    input_datas = torch.load(f'vqa/{prefix}vqa_ce_val_full_v1.pth')
    sentences = torch.load(f"vqa/{prefix}vqa_ce_val_question_full_v1.pth")
        
    # sentences=[item['question'] for i, item in enumerate(meta_data) if i not in missing_index]
    vectorizer = TfidfVectorizer(stop_words='english')  # 去除停用词
    X = vectorizer.fit_transform(sentences)
    
    for num_clusters in [3,5,10]:
        # 2. 使用 K-Means 进行聚类
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)

        # 获取聚类标签
        labels = kmeans.labels_

        # 3. 可视化聚类结果（使用 PCA 降维）
        pca = PCA(n_components=2)  # 降维到 2D
        X_pca = pca.fit_transform(X.toarray())

        # 绘制聚类结果
        plt.figure(figsize=(8, 6))
        for i in range(num_clusters):
            plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f'Cluster {i+1}')
            
        # 检查数据数量
        print(f"Number of sentences: {len(sentences)}")

        # 检查聚类标签分布
        import numpy as np
        unique, counts = np.unique(labels, return_counts=True)
        print("Cluster distribution:", dict(zip(unique, counts)))

        # 4. 映射每个索引到对应的类别编号
        index_to_cluster = {index: label for index, label in enumerate(labels)}  # 类别编号从 1 开始

        # 输出每个索引及其对应的类别编号
        for index, cluster in index_to_cluster.items():
            print(f"Index: {index} -> Cluster: {cluster}")

        torch.save(index_to_cluster, f'margin/{prefix}vqa_index_to_cluster-{num_clusters}.pth')
        print(len(index_to_cluster), f'margin/{prefix}vqa_index_to_cluster-{num_clusters}.pth')
        
    print(sentences[:5])
        
    
elif test_data=="pope" or test_data=="insuff_att" or test_data=="amb_desc" or test_data=="mislead":

    train_dataset = load_dataset("lmms-lab/POPE", split="test") #"default"

    # 示例数据：一系列简短语句
    sentences=[item['question'] for item in train_dataset]
    # 1. 使用 TF-IDF 提取特征
    vectorizer = TfidfVectorizer(stop_words='english')  # 去除停用词
    X = vectorizer.fit_transform(sentences)


    for num_clusters in [3,5,10]: # ,20,30,40,50,100
        # 2. 使用 K-Means 进行聚类
        # num_clusters = 5
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)

        # 获取聚类标签
        labels = kmeans.labels_

        # 3. 可视化聚类结果（使用 PCA 降维）
        pca = PCA(n_components=2)  # 降维到 2D
        X_pca = pca.fit_transform(X.toarray())



        # 绘制聚类结果
        plt.figure(figsize=(8, 6))
        for i in range(num_clusters):
            plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f'Cluster {i+1}')
            
            
        # 检查数据数量
        print(f"Number of sentences: {len(sentences)}")

        # 检查聚类标签分布
        import numpy as np
        unique, counts = np.unique(labels, return_counts=True)
        print("Cluster distribution:", dict(zip(unique, counts)))

        # 4. 映射每个索引到对应的类别编号
        index_to_cluster = {index: label for index, label in enumerate(labels)}  # 类别编号从 1 开始

        # 输出每个索引及其对应的类别编号
        for index, cluster in index_to_cluster.items():
            print(f"Index: {index} -> Cluster: {cluster}")

        torch.save(index_to_cluster, f'margin/{prefix}index_to_cluster-{num_clusters}.pth')
        
elif test_data=="indom" or test_data=="outdom":
    sentences=torch.load(f'pope/{prefix}pope_question_list{test_data}.pth')
    # sentences=[item['question'] for item in train_dataset]
    vectorizer = TfidfVectorizer(stop_words='english')  # 去除停用词
    X = vectorizer.fit_transform(sentences)


    for num_clusters in [3,5,10]: # ,20,30,40,50,100
        # 2. 使用 K-Means 进行聚类
        # num_clusters = 5
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)

        # 获取聚类标签
        labels = kmeans.labels_

        # 3. 可视化聚类结果（使用 PCA 降维）
        pca = PCA(n_components=2)  # 降维到 2D
        X_pca = pca.fit_transform(X.toarray())



        # 绘制聚类结果
        plt.figure(figsize=(8, 6))
        for i in range(num_clusters):
            plt.scatter(X_pca[labels == i, 0], X_pca[labels == i, 1], label=f'Cluster {i+1}')
            
            
        # 检查数据数量
        print(f"Number of sentences: {len(sentences)}")

        # 检查聚类标签分布
        import numpy as np
        unique, counts = np.unique(labels, return_counts=True)
        print("Cluster distribution:", dict(zip(unique, counts)))

        # 4. 映射每个索引到对应的类别编号
        index_to_cluster = {index: label for index, label in enumerate(labels)}  # 类别编号从 1 开始

        # 输出每个索引及其对应的类别编号
        for index, cluster in index_to_cluster.items():
            print(f"Index: {index} -> Cluster: {cluster}")
        
        torch.save(index_to_cluster, f'margin/{prefix}{test_data}_index_to_cluster-{num_clusters}.pth')