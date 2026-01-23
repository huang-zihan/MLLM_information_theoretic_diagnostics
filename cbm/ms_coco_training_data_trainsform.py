import torch
import numpy as np
import math

# dataset = torch.load("ce_data_full.pth")
# print(len(dataset))
# dataset = torch.load("ce_data_full_v1.pth")
# print(len(dataset))
# exit()

# object_dict1 = torch.load('object_dict.pth')  # 物体名称与编号的映射
# object_dict2 = torch.load('object_dict_full.pth')  # 物体名称与编号的映射
# print(len(object_dict1), len(object_dict2))
# print(object_dict1)
# exit()


# 加载数据
object_dict = torch.load('object_dict_full.pth')  # 物体名称与编号的映射
raw_result = torch.load('raw_result_full.pth')  # 图片与物体得分的映射

# 获取物体的数量
num_classes = len(object_dict)

# 初始化数据集
dataset = []

def softmax(lst):
    # # 减去最大值，避免数值溢出
    # max_val = max(lst)
    # exp_lst = [math.exp(x - max_val) for x in lst]
    
    # 计算所有元素的指数之和
    sum_exp = sum(lst)
    
    # 计算Softmax值
    softmax_lst = [x / sum_exp for x in lst]
    
    return softmax_lst


# 遍历原始结果
for i, (image_name, scores) in enumerate(raw_result.items()):
    
    print(i, "/", len(raw_result), end='\r')
    
    if len(scores)==0:
        continue
    
    # 创建注释列表，初始化为0
    annotation = [0] * num_classes
    

    
    # 填充注释列表
    for obj_index, score in scores.items():
        annotation[obj_index] = sum(score)
    annotation=softmax(annotation)
    
    # 构建数据项
    data_item = {
        'image': image_name,
        'annotation': annotation
    }
    
    # 添加到数据集中
    dataset.append(data_item)

torch.save(dataset, "ce_data_full.pth")
