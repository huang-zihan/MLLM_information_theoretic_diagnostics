import torch
from PIL import Image
import os
import numpy as np
from ultralytics import YOLO

# Load pre-trained YOLOv5 model
model_name = 'yolo11l.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Ensure using GPU
model = YOLO(model_name).to(device)  # Move model to GPU

# Image folder path
image_folder = '/deepfreeze/junda/datasets/COCO2014/train2014'

# Create folder to save results
os.makedirs('detection_results_batch', exist_ok=True)

raw_result = {}
object_dict = set()

# Get all image file paths
image_paths = [os.path.join(image_folder, image_name) for image_name in os.listdir(image_folder)]
batch_size = 256  # Batch size

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
        print(results[0])
        exit()
    
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
torch.save(object_dict, f'result/coco_feature/object_dict_full_{model_name}.pth')
torch.save(raw_result, f'result/coco_feature/raw_result_full_{model_name}.pth')

print("MSCOCO yolo done!")
print()
print()