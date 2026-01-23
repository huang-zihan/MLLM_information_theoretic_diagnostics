import torch
from PIL import Image
import os

# 加载预训练的YOLOv5模型
model_name='yolov5s'
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 可以选择其他模型如 'yolov5m', 'yolov5l', 'yolov5x'
# from ultralytics import YOLO
# # Load a model
# model = YOLO(model_name)  # pretrained YOLO11n model
# # exit()

# 图像文件夹路径
image_folder = '/deepfreeze/junda/datasets/COCO2014/train2014' # in total 82783

# 创建保存结果的文件夹
os.makedirs('detection_results_batch', exist_ok=True)

raw_result={}
object_dict={}

obj_index=0
# 遍历文件夹中的所有图像
for i, image_name in enumerate(os.listdir(image_folder)):
    
    print(i, "/", len(os.listdir(image_folder)), end='\r')
    # if i==10000:
    #     break
    
    image_path = os.path.join(image_folder, image_name)
    
    # 加载图像
    image = Image.open(image_path)
    
    # 进行推理
    results = model(image)
    print(results)
    # 获取检测结果的详细信息
    detections = results.pandas().xyxy[0]
    
    raw_result[image_name]=dict()
    for index, row in detections.iterrows():
        
        if row['name'] not in object_dict:
            object_dict[row['name']]=obj_index
            obj_index+=1
        
        if object_dict[row['name']] not in raw_result[image_name]:
            raw_result[image_name][object_dict[row['name']]]=[row['confidence']]
        else:
            raw_result[image_name][object_dict[row['name']]].append(row['confidence'])

    #     print(f"Image: {image_name}, Detected {row['name']} with confidence {row['confidence']:.2f}")
    
    # 保存带有检测框的图像
    # results.save(save_dir=os.path.join('detection_results_batch', image_name))
    
torch.save(object_dict, f'object_dict_full_.pth')
torch.save(raw_result, f'raw_result_full.pth')
