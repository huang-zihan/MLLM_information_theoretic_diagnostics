import os
import torch
import numpy as np
from datasets import load_dataset
from ultralytics import YOLO
import cv2
import random

# 加载数据
object_dict = None#torch.load('object_dict.pth')  # 物体名称与编号的映射
visual_prompt=False


# 初始化数据集
dataset = []

def softmax(lst):
    # 计算所有元素的指数之和
    sum_exp = sum(lst)
    
    # 计算Softmax值
    softmax_lst = [x / sum_exp for x in lst]
    
    return softmax_lst

train_dataset = load_dataset("moranyanuka/OpenCHAIR")['test']

model_name = 'yolo11l.pt'  # 使用高版本模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 确保使用GPU
model = YOLO(model_name).to(device)  # 加载高版本模型

if not visual_prompt:
    # 创建输出文件夹
    output_dir = 'result/chair_feature/img/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
else:
    output_dir = '/home/shared/rohan/vd-llm/yolo_output/'

# 遍历原始结果
missing_obj_index=[]
description_list=[item['text'] for item in train_dataset]
for i, item in enumerate(train_dataset):
    print(i, "/", len(train_dataset), end='\r')

    if not visual_prompt:
        # 将图像保存到文件
        image = item['image'].convert('RGB')
        image_path = os.path.join(output_dir, f'image_{i}.jpg')
        cv2.imwrite(image_path, np.array(image)[:, :, ::-1])  # 保存为BGR格式
    else:
        image_path = os.path.join(output_dir, f'annotated_image_{i}.jpg')
        
    # 进行推理
    with torch.no_grad():
        results = model.predict(image_path, device=device, verbose=False)  # 使用新的推理方式
        if object_dict is None:
            object_dict=results[0].names
            # 获取物体的数量
            num_classes = len(object_dict)


    # 处理结果
    detections = results[0]  # 获取检测结果
    annotation = [0] * len(object_dict)
    
    for box in detections.boxes:  # 遍历每个检测框
        cls = int(box.cls.item())  # 获取类别
        conf = float(box.conf.item())  # 获取置信度
        annotation[cls] += conf
    
    if all(x == 0 for x in annotation):
        annotation = []
        missing_obj_index.append(i)
    else:
        annotation = softmax(annotation)

    assert item['text']==description_list[i]
    question="Is there "+item['text'].lower()
    # 构建数据项
    data_item = {
        'id': i,
        'image': item['image'],
        'annotation': annotation,
        'question': question,
        'true_caption': True
    }
    
    # 添加到数据集中
    dataset.append(data_item)
    
    random_index = random.randint(0, len(description_list) - 1)
    while random_index == i:
        random_index = random.randint(0, len(description_list) - 1)
    question = "Is there "+description_list[random_index].lower()
    # 添加一个负的问题样例进去
    data_item = {
        'id': i,
        'image': item['image'],
        'annotation': annotation,
        'question': question,
        'true_caption': False
    }
    dataset.append(data_item)


# 保存结果
if not visual_prompt:
    torch.save(dataset, f"result/chair_feature/chair_ce_test_data_{model_name}.pth")
else:
    torch.save(dataset, f"result/chair_feature/vp_chair_ce_test_data_{model_name}.pth")
print("CHAIR yolo done!")
print(missing_obj_index, "with length", len(missing_obj_index))
print()
print()