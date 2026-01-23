import os
import torch
import numpy as np
from datasets import load_dataset
from ultralytics import YOLO
import cv2

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

train_dataset = load_dataset("HuggingFaceM4/A-OKVQA", split="validation", cache_dir='/deepfreeze/junda/datasets/')
model_name = 'yolo11l.pt'  # 使用高版本模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 确保使用GPU
model = YOLO(model_name).to(device)  # 加载高版本模型

if not visual_prompt:
    # 创建输出文件夹
    output_dir = 'result/aokvqa_feature/img/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
else:
    output_dir = '/home/shared/rohan/vd-llm/yolo_output/'

# correct answer
for i, item in enumerate(train_dataset):
    
    print(item)
    exit()
    
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
            # print(results[0])
            object_dict=results[0].names
            # 获取物体的数量
            num_classes = len(object_dict)
            # print(object_dict, num_classes)
            # exit()

    # 处理结果
    detections = results[0]  # 获取检测结果
    annotation = [0] * len(object_dict)
    
    for box in detections.boxes:  # 遍历每个检测框
        cls = int(box.cls.item())  # 获取类别
        conf = float(box.conf.item())  # 获取置信度
        annotation[cls] += conf

    # print(annotation)
    if sum(annotation)!=0:
        annotation = softmax(annotation)
    else:
        annotation = []
    
    # 构建数据项
    data_item = {
        'id': i,
        'annotation': annotation
    }
    
    # 添加到数据集中
    dataset.append(data_item)
    # if i==10:
    #     break

# print(dataset)
# 保存结果
if not visual_prompt:
    torch.save(dataset, f"result/aokvqa_feature/aokvqa_ce_test_data_{model_name}.pth")
else:
    torch.save(dataset, f"result/aokvqa_feature/vp_aokvqa_ce_test_data_{model_name}.pth")
print("aokvqa yolo done!")
print()
print()

# # 清理临时文件
# import shutil
# shutil.rmtree(output_dir)  # 删除临时文件夹

# import torch
# import numpy as np
# import math
# from datasets import load_dataset


# # 加载数据
# object_dict = torch.load('object_dict.pth')  # 物体名称与编号的映射

# # 获取物体的数量
# num_classes = len(object_dict)

# # 初始化数据集
# dataset = []

# def softmax(lst):
#     # # 减去最大值，避免数值溢出
#     # max_val = max(lst)
#     # exp_lst = [math.exp(x - max_val) for x in lst]
    
#     # 计算所有元素的指数之和
#     sum_exp = sum(lst)
    
#     # 计算Softmax值
#     softmax_lst = [x / sum_exp for x in lst]
    
#     return softmax_lst


# train_dataset = load_dataset("lmms-lab/POPE", split="test") #"default"
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 可以选择其他模型如 'yolov5m', 'yolov5l', 'yolov5x'


# # 遍历原始结果
# for i, item in enumerate(train_dataset):
#     print(i, "/", len(train_dataset), end='\r')

#     # print(train_dataset)
#     image = item['image'].convert('RGB')
#     results = model(image)
#     detections = results.pandas().xyxy[0]
    
#     annotation=[0]*len(object_dict)
#     for index, row in detections.iterrows():
#         if row['name'] not in object_dict:
#             continue
#         annotation[object_dict[row['name']]]+=row['confidence']
        
#     annotation=softmax(annotation)

#     # 构建数据项
#     data_item = {
#         'id': item['id'],
#         'annotation': annotation
#     }
    
#     # 添加到数据集中
#     dataset.append(data_item)

# torch.save(dataset, "pope_ce_test_data.pth")
