import os
import zipfile
import requests
import json

import torch
from PIL import Image
import os
import numpy as np
from ultralytics import YOLO


def download_coco_captions(url, download_path):
    # 确保下载路径存在
    os.makedirs(download_path, exist_ok=True)
    
    # 设置ZIP文件的完整路径
    zip_file_path = os.path.join(download_path, 'annotations_trainval2014.zip')
    
    # 下载ZIP文件
    response = requests.get(url)
    response.raise_for_status()  # 检查请求是否成功

    with open(zip_file_path, 'wb') as f:
        f.write(response.content)

    # 解压缩ZIP文件
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)

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

# print(len(coco_captions))

# # 打印前几个标题
# for item in list(coco_captions.items())[:5]:
#     # print(f'Image ID: {img_id}, Captions: {captions}')
#     print(item)

model_name = 'yolo11l.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 确保使用GPU
model = YOLO(model_name).to(device)  # 将模型移动到GPU
raw_result = {}
object_dict = set()


image_folder = '/deepfreeze/junda/datasets/COCO2014/val2014'
# coco_captions = '/deepfreeze/junda/datasets/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json'
questions = coco_captions
image_paths = [os.path.join(image_folder, f"""COCO_val2014_{questions[i]['image_id']:012d}.jpg""") for i in range(len(questions))]

qs = [os.path.join(image_folder, f"""{questions[i]['caption']}""") for i in range(len(questions))]
print("question number:", len(questions), "diffent image:", len(set(image_paths)))


image_paths=list(set(image_paths))
batch_size = 128  # 批处理大小

# 遍历图像路径并进行批处理推理
for batch_start in range(0, len(image_paths), batch_size):
    
    print(batch_start, "/", len(image_paths), "with step", batch_size, end='\r')
    
    if batch_start + batch_size<=len(image_paths):
        batch_paths = image_paths[batch_start: batch_start + batch_size]
    else:
        batch_paths = image_paths[batch_start: len(image_paths)]
    
    # 进行推理
    with torch.no_grad():
        results = model.predict(batch_paths, device=device, verbose=False, batch=batch_size, conf=0.5, half=True)

    for i, image_path in enumerate(batch_paths):
        image_name = os.path.basename(image_path)
        raw_result[image_name] = dict()
        
        # 处理每个结果
        for item in results[i]:
            
            cls_list = [int(cls.item()) for cls in item.boxes.cls.cpu()]
            conf_list = [float(conf.item()) for conf in item.boxes.conf.cpu()]
            for (cls, conf) in zip(cls_list, conf_list):
                object_dict.add(cls)  # 添加检测到的类到集合
                if cls not in raw_result[image_name]:
                    raw_result[image_name][cls] = [conf]
                else:
                    raw_result[image_name][cls].append(conf)


torch.save(object_dict, f'result/coco_feature/coco_caption_object_dict_full_{model_name}.pth')
torch.save(raw_result, f'result/coco_feature/coco_caption_raw_result_full_{model_name}.pth')
# print(raw_result)
print("COCO Caption validation set yolo done!")
print()
print()

