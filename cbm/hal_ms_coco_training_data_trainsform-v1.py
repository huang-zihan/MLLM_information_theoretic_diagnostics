import torch
import numpy as np
import math
import json
import os

model_name = 'yolo11l.pt'

# generate MSCOCO training
training_set=True


object_dict = torch.load(f'result/coco_feature/hal_object_dict_full_{model_name}.pth')  # 物体名称与编号的映射
raw_result = torch.load(f'result/coco_feature/hal_raw_result_full_{model_name}.pth')  # 图片与物体得分的映射

# 获取物体的数量
num_classes = 80 #len(object_dict)

def softmax(lst):
    # 计算所有元素的指数之和
    sum_exp = sum(lst)
    
    # 计算Softmax值
    softmax_lst = [x / sum_exp for x in lst]
    
    return softmax_lst


# 遍历原始结果
image_folder = '/deepfreeze/junda/datasets/COCO2014/val2014'
# questions_path = '/deepfreeze/junda/datasets/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json'

file_path = 'hal_eval/in_domain_evaluation.json'
# 加载 JSON 文件
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
hal_questions=[]
questions=[]
# image_paths=[]
for item in data:
    hal_captions = item['hal_caption']
    # image_paths.append()
    for hal in hal_captions:
        hal_questions.append(
            {
                'image_name': os.path.basename(item['image']),
                'question': hal['caption'],
                'hal_type': hal['type']   
            }
        )
        
    questions.append(
        {
            'image_name': os.path.basename(item['image']),
            'question': item['caption'],
        }
    )


### hallucinated result
# 初始化数据集
dataset = []

for i, item in enumerate(hal_questions):

    scores=raw_result[item['image_name']]
    print(i, "/", len(hal_questions), end='\r')
    
    if len(scores)==0:
        # continue
        data_item = {
            'question': item['question'],
            'image': item['image_name'],
            'hal_type': item['hal_type'],
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
        'question': item['question'],
        'image': item['image_name'],
        'hal_type': item['hal_type'],
        'annotation': annotation
    }
    
    # 添加到数据集中
    dataset.append(data_item)

print(dataset[0])
torch.save(dataset, f"wrong_hal_ce_data_full_v1.pth")


### correct result
# 初始化数据集
dataset = []

for i, item in enumerate(questions):

    scores=raw_result[item['image_name']]
    print(i, "/", len(questions), end='\r')
    
    if len(scores)==0:
        # continue
        data_item = {
            'question': item['question'],
            'image': item['image_name'],
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
        'question': item['question'],
        'image': item['image_name'],
        'annotation': annotation
    }
    
    # 添加到数据集中
    dataset.append(data_item)

print(dataset[0])
torch.save(dataset, f"correct_hal_ce_data_full_v1.pth")
