import torch
from PIL import Image
import os
import numpy as np
from ultralytics import YOLO
import json

# Load pre-trained YOLO model
model_name = 'yolo11l.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Ensure using GPU
model = YOLO(model_name).to(device)  # Move model to GPU

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
# Load JSON file
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

############################################################################################

# Image folder path
image_folder = '/deepfreeze/junda/datasets/COCO2014/'

image_paths = []
qs=[]

for item in data:
    hal_captions = item['hal_caption']
    image_paths.append(os.path.join(image_folder, item['image']))
    for hal in hal_captions:
        qs.append(hal['caption'])

print("question number:", len(qs), "different image:", len(set(image_paths)))

image_paths=list(set(image_paths))
batch_size = 128  # Batch size

# Iterate over image paths and perform batch inference
for batch_start in range(0, len(image_paths), batch_size):
    
    print(batch_start, "/", len(image_paths), "with step", batch_size, end='\r')
    
    if batch_start + batch_size <= len(image_paths):
        batch_paths = image_paths[batch_start: batch_start + batch_size]
    else:
        batch_paths = image_paths[batch_start: len(image_paths)]
    
    # Perform inference
    with torch.no_grad():
        results = model.predict(batch_paths, device=device, verbose=False, batch=batch_size, conf=0.5, half=True)

    for i, image_path in enumerate(batch_paths):
        image_name = os.path.basename(image_path)
        raw_result[image_name] = dict()
        
        # Process each result
        for item in results[i]:
            
            cls_list = [int(cls.item()) for cls in item.boxes.cls.cpu()]
            conf_list = [float(conf.item()) for conf in item.boxes.conf.cpu()]
            for (cls, conf) in zip(cls_list, conf_list):
                object_dict.add(cls)  # Add detected class to set
                if cls not in raw_result[image_name]:
                    raw_result[image_name][cls] = [conf]
                else:
                    raw_result[image_name][cls].append(conf)

# Save results
torch.save(object_dict, f'result/coco_feature/hal_object_dict_full_{model_name}.pth')
torch.save(raw_result, f'result/coco_feature/hal_raw_result_full_{model_name}.pth')
    # print(raw_result)
print("HAL MSCOCO validation set yolo done!")
print()
print()