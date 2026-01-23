# import torch
# from PIL import Image
# import os

# # 加载预训练的YOLOv5模型
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 可以选择其他模型如 'yolov5m', 'yolov5l', 'yolov5x'
# # 图像文件夹路径
# image_folder = '/deepfreeze/junda/datasets/COCO2014/train2014'

# # 创建保存结果的文件夹
# os.makedirs('detection_results_batch', exist_ok=True)

# object_dict={}
# index=0

# # 遍历文件夹中的所有图像
# for i, image_name in enumerate(os.listdir(image_folder)):
    
#     print(i, "/", 2000, end='\r')
#     if i==2000:
#         break
    
    
#     image_path = os.path.join(image_folder, image_name)
    
#     # 加载图像
#     image = Image.open(image_path)
    
#     # 进行推理
#     results = model(image)
    
#     # 获取检测结果的详细信息
#     detections = results.pandas().xyxy[0]
    
#     # 打印检测到的物体种类
#     for j, row in detections.iterrows():
#         object_dict
#         if row['name'] not in object_dict:
#             object_dict[row['name']]=index
#             index+=1

# print(object_dict)
# torch.save(object_dict, 'object_dict.pth')

