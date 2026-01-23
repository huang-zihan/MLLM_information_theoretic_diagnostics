import torch
import numpy as np
import math

model_name = 'yolo11l.pt'

# 加载数据
object_dict = torch.load(f'result/coco_feature/object_dict_full_{model_name}.pth')  # 物体名称与编号的映射
raw_result = torch.load(f'result/coco_feature/raw_result_full_{model_name}.pth')  # 图片与物体得分的映射
# print(len(raw_result))
# # print(raw_result)
# exit()
# 获取物体的数量
num_classes = 80 #len(object_dict)
# print(num_classes)
# print()
# exit()
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
        # continue
        data_item = {
            'image': image_name,
            'annotation': None
        }
        dataset.append(data_item)
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

torch.save(dataset, f"ce_data_full_v1.pth")
