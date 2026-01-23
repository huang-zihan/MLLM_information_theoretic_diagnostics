import torch
from PIL import Image
import os
import numpy as np
from ultralytics import YOLO

# 加载预训练的YOLOv5模型
model_name = 'yolo11l.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 确保使用GPU
model = YOLO(model_name).to(device)  # 将模型移动到GPU

# 图像文件夹路径
image_folder = '/deepfreeze/junda/datasets/COCO2014/train2014'

# 创建保存结果的文件夹
os.makedirs('detection_results_batch', exist_ok=True)

raw_result = {}
object_dict = set()

# 获取所有图像文件路径
image_paths = [os.path.join(image_folder, image_name) for image_name in os.listdir(image_folder)]
batch_size = 256  # 批处理大小

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
        print(results[0])
        exit()
    
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
        
    # print()
    # print(len(raw_result))
    # print(raw_result)
    # print()
    # if batch_start==256:
    #     break

# 保存结果
torch.save(object_dict, f'result/coco_feature/object_dict_full_{model_name}.pth')
torch.save(raw_result, f'result/coco_feature/raw_result_full_{model_name}.pth')

print("MSCOCO yolo done!")
print()
print()

# import torch
# from PIL import Image
# import os

# # 加载预训练的YOLOv5模型
# model_name='yolo11l.pt'
# from ultralytics import YOLO
# # Load a model
# model = YOLO(model_name)  # pretrained YOLO11n model

# # 图像文件夹路径
# image_folder = '/deepfreeze/junda/datasets/COCO2014/train2014' # in total 82783

# # 创建保存结果的文件夹
# os.makedirs('detection_results_batch', exist_ok=True)

# raw_result={}
# object_dict={}

# obj_index=0

# object_dict=set()

# # 遍历文件夹中的所有图像
# for i, image_name in enumerate(os.listdir(image_folder)):
    
#     print(i, "/", len(os.listdir(image_folder)), end='\r')
#     # if i==10000:
#     #     break
    
#     image_path = os.path.join(image_folder, image_name)
    
#     # 加载图像
#     image = Image.open(image_path)
    
#     # 进行推理
#     with torch.no_grad():
#         results = model.predict(image_path, verbose=False, batch=1, conf=0.5, half=True) # , device=device

#     raw_result[image_name]=dict()
#     for item in results:
#         cls_list = [int(cls.item()) for cls in item.boxes.cls.cpu()]
#         conf_list = [float(conf.item()) for conf in item.boxes.conf.cpu()]
#         for (cls, conf) in zip(cls_list, conf_list):
#             if cls not in object_dict:
#                 object_dict.add(cls)
#             if cls not in raw_result[image_name]:
#                 raw_result[image_name][cls]=[conf]
#             else:
#                 raw_result[image_name][cls].append(conf)

# torch.save(object_dict, f'result/coco_feature/object_dict_full_{model_name}.pth')
# torch.save(raw_result, f'result/coco_feature/raw_result_full{model_name}.pth')