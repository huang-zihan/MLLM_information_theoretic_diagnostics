import torch
import numpy as np
import math
from datasets import load_dataset


# 加载数据
object_dict = torch.load('object_dict.pth')  # 物体名称与编号的映射

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


train_dataset = load_dataset("lmms-lab/POPE", split="test") #"default"
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 可以选择其他模型如 'yolov5m', 'yolov5l', 'yolov5x'


# 遍历原始结果
for i, item in enumerate(train_dataset):
    print(i, "/", len(train_dataset), end='\r')

    # print(train_dataset)
    image = item['image'].convert('RGB')
    results = model(image)
    detections = results.pandas().xyxy[0]
    
    annotation=[0]*len(object_dict)
    for index, row in detections.iterrows():
        if row['name'] not in object_dict:
            continue
        annotation[object_dict[row['name']]]+=row['confidence']
        
    annotation=softmax(annotation)

    # # 填充注释列表
    # for obj_index, score in scores.items():
    #     annotation[obj_index] = sum(score)
    # annotation=softmax(annotation)
    
    # 构建数据项
    data_item = {
        'id': item['id'],
        'annotation': annotation
    }
    
    # 添加到数据集中
    dataset.append(data_item)

torch.save(dataset, "pope_ce_test_data.pth")
