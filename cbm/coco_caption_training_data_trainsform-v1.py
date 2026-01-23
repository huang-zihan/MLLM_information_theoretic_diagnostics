import os
import zipfile
import requests
import json


import torch
import numpy as np
import math
import json
import os

def download_coco_captions(url, download_path):
    # # 确保下载路径存在
    # os.makedirs(download_path, exist_ok=True)
    
    # # 设置ZIP文件的完整路径
    # zip_file_path = os.path.join(download_path, 'annotations_trainval2014.zip')
    
    # # 下载ZIP文件
    # response = requests.get(url)
    # response.raise_for_status()  # 检查请求是否成功

    # with open(zip_file_path, 'wb') as f:
    #     f.write(response.content)

    # # 解压缩ZIP文件
    # with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    #     zip_ref.extractall(download_path)

    return os.path.join(download_path, 'annotations', 'captions_val2014.json')

def load_coco_captions(length=10000):
    # COCO2014训练集的注释文件URL
    url = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
    download_path = '/deepfreeze/junda/datasets/COCO2014_caption'
    
    # 下载并获取注释文件路径
    annotation_file = download_coco_captions(url, download_path)

    # 读取JSON文件
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # 创建一个字典来存储图像ID及其标题
    coco_captions = []
    
    # 遍历注释，提取图像ID和标题
    for i, annotation in enumerate(annotations['annotations']):
        # print(annotation)
        # exit()
        if i==length:
            break
        
        image_id = annotation['image_id']
        caption = annotation['caption']
        
        # if image_id not in coco_captions:
        #     coco_captions[image_id] = []
        
        coco_captions.append(
            {"image_id": image_id, "caption": caption}
        )
        
    return coco_captions

# 使用示例
coco_captions = load_coco_captions(length=10000)

model_name = 'yolo11l.pt'

object_dict = torch.load(f'result/coco_feature/coco_caption_object_dict_full_{model_name}.pth')  # 物体名称与编号的映射
raw_result = torch.load(f'result/coco_feature/coco_caption_raw_result_full_{model_name}.pth')  # 图片与物体得分的映射

# 获取物体的数量
num_classes = 80

# 初始化数据集
dataset = []

def softmax(lst):
    # 计算所有元素的指数之和
    sum_exp = sum(lst)
    
    # 计算Softmax值
    softmax_lst = [x / sum_exp for x in lst]
    
    return softmax_lst


# 遍历原始结果
image_folder = '/deepfreeze/junda/datasets/COCO2014/val2014'

questions = coco_captions
image_paths = [os.path.join(image_folder, f"""COCO_val2014_{questions[i]['image_id']:012d}.jpg""") for i in range(len(questions))]

qs = [os.path.join(image_folder, f"""{questions[i]['caption']}""") for i in range(len(questions))]
print("question number:", len(questions), "diffent image:", len(set(image_paths)))

for i, item in enumerate(questions):

    image_name=f"""COCO_val2014_{item['image_id']:012d}.jpg"""
    scores=raw_result[image_name]
    print(i, "/", len(questions), end='\r')
    
    if len(scores)==0:
        # continue
        data_item = {
            'question': item['caption'],
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
    
    # print(item)
    # exit()
    # 构建数据项
    data_item = {
        'question': item['caption'],
        'image': image_name,
        'annotation': annotation
    }
    
    # 添加到数据集中
    dataset.append(data_item)


torch.save(dataset, f"coco_caption_ce_data_full_v1.pth")
