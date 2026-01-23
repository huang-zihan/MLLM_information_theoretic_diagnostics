import torch
from PIL import Image
import os
import numpy as np
from ultralytics import YOLO
import json

# 加载预训练的YOLOv5模型
model_name = 'yolo11l.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 确保使用GPU
model = YOLO(model_name).to(device)  # 将模型移动到GPU

raw_result = {}
object_dict = set()

hal_type_dict={
    "Spatial Relationship Hallucination":0,
    "Objective Hallucination":1,
    "Attributive Hallucination":2,
    "Event Hallucination":3,
}

####################### example for utilizing hal annotation data#######################
import json
file_path = 'hal_eval/in_domain_evaluation.json'
# 加载 JSON 文件
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# # 遍历 JSON 数据并打印信息
# for i, item in enumerate(data):
#     image = item['image']
#     caption = item['caption']
#     hal_captions = item['hal_caption']
    
#     print(f"Image: {image}")
#     print(f"Caption: {caption}")
    
#     # 遍历并打印 hallucinations
#     for hal in hal_captions:
#         hal_type = hal['type']
#         hal_caption = hal['caption']
#         print(f"  {hal_type}: {hal_caption}")

#     print("\n")  # 换行以便于阅读
# exit()

############################################################################################

# 图像文件夹路径
image_folder = '/deepfreeze/junda/datasets/COCO2014/'
    # questions_path = '/deepfreeze/junda/datasets/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json'

# # 加载问题
# with open(questions_path, 'r') as f:
#     questions = json.load(f)

# 获取所有图像文件路径
# if training_set:
#     questions = questions['questions']
#     image_paths = [os.path.join(image_folder, f"""COCO_train2014_{questions[i]['image_id']:012d}.jpg""") for i in range(len(questions))]
# else:
# questions = questions['questions']
image_paths = []
qs=[]

for item in data:
    hal_captions = item['hal_caption']
    # print(item['image'])
    image_paths.append(os.path.join(image_folder, item['image']))
    for hal in hal_captions:
        qs.append(hal['caption'])

print("question number:", len(qs), "diffent image:", len(set(image_paths)))

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

# 保存结果
torch.save(object_dict, f'result/coco_feature/hal_object_dict_full_{model_name}.pth')
torch.save(raw_result, f'result/coco_feature/hal_raw_result_full_{model_name}.pth')
    # print(raw_result)
print("HAL MSCOCO validation set yolo done!")
print()
print()
