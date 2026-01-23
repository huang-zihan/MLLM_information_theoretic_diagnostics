import torch
import numpy as np
import math
import json
import os

model_name = 'yolo11l.pt'

###########################################

###########################################


# generate MSCOCO training
training_set=False

if training_set:
    object_dict = torch.load(f'result/coco_feature/object_dict_full_{model_name}.pth')  # 物体名称与编号的映射
    raw_result = torch.load(f'result/coco_feature/raw_result_full_{model_name}.pth')  # 图片与物体得分的映射
else:
    object_dict = torch.load(f'result/coco_feature/vqa_object_dict_full_{model_name}.pth')  # 物体名称与编号的映射
    raw_result = torch.load(f'result/coco_feature/vqa_raw_result_full_{model_name}.pth')  # 图片与物体得分的映射

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
    # 计算所有元素的指数之和
    sum_exp = sum(lst)
    
    # 计算Softmax值
    softmax_lst = [x / sum_exp for x in lst]
    
    return softmax_lst


# 遍历原始结果
if training_set:
    image_folder = '/deepfreeze/junda/datasets/COCO2014/train2014'
    questions_path = '/deepfreeze/junda/datasets/VQAv2/v2_OpenEnded_mscoco_train2014_questions.json'
    # annotations_path = '/deepfreeze/junda/datasets/VQAv2/v2_mscoco_val2014_annotations.json'
else:
    image_folder = '/deepfreeze/junda/datasets/COCO2014/val2014'
    questions_path = '/deepfreeze/junda/datasets/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json'
    annotations_path = '/deepfreeze/junda/datasets/VQAv2/v2_mscoco_val2014_annotations.json'


with open(questions_path, 'r') as f:
    questions = json.load(f)

with open(annotations_path, 'r') as f:
    annotations = json.load(f)

# print(annotations['annotations'][0])
# exit()

# print(questions['question_type'])
# exit()

if training_set:
    questions = questions['questions'] #[:10000]
    annotations = annotations['annotations'] #[:10000]
    image_paths = [os.path.join(image_folder, f"""COCO_train2014_{questions[i]['image_id']:012d}.jpg""") for i in range(len(questions))]
else:
    questions = questions['questions'][:10000]
    annotations = annotations['annotations'][:10000]
    image_paths = [os.path.join(image_folder, f"""COCO_val2014_{questions[i]['image_id']:012d}.jpg""") for i in range(len(questions))]

qs = [os.path.join(image_folder, f"""{questions[i]['question']}""") for i in range(len(questions))]

for i, item in enumerate(questions):
    if training_set:
        image_name=f"""COCO_train2014_{item['image_id']:012d}.jpg"""
    else:
        image_name=f"""COCO_val2014_{item['image_id']:012d}.jpg"""
    scores=raw_result[image_name]
    print(i, "/", len(questions), end='\r')
    assert annotations[i]['question_id']==item['question_id']
    if len(scores)==0:
        # continue
        data_item = {
            'question': item['question'],
            'response': annotations[i]['answers'][0]['answer'],
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
        'question': item['question'],
        'response': annotations[i]['answers'][0]['answer'],
        'image': image_name,
        'annotation': annotation
    }
    
    # 添加到数据集中
    dataset.append(data_item)

if training_set:
    torch.save(dataset, f"vqa_ce_data_full_v1_train.pth")
else:
    torch.save(dataset, f"vqa_ce_data_full_v1.pth")
