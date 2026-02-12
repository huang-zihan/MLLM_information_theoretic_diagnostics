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
    # Ensure the download path exists
    os.makedirs(download_path, exist_ok=True)
    
    # Set the full path for the ZIP file
    zip_file_path = os.path.join(download_path, 'annotations_trainval2014.zip')
    
    # Download the ZIP file
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful

    with open(zip_file_path, 'wb') as f:
        f.write(response.content)

    # Extract the ZIP file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)

    return os.path.join(download_path, 'annotations', 'captions_val2014.json')

def load_coco_captions(length=10000):
    # URL for COCO2014 training annotations
    url = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
    download_path = '/deepfreeze/junda/datasets/COCO2014_caption'
    
    # Download and get the annotation file path
    annotation_file = download_coco_captions(url, download_path)

    # Read the JSON file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Create a dictionary to store image IDs and their captions
    coco_captions = []
    
    # Iterate over annotations, extract image IDs and captions
    for i, annotation in enumerate(annotations['annotations']):

        if i == length:
            break
        
        image_id = annotation['image_id']
        caption = annotation['caption']
        
        # if image_id not in coco_captions:
        #     coco_captions[image_id] = []
        
        coco_captions.append(
            {"image_id": image_id, "caption": caption}
        )
        
    return coco_captions

# Example usage
coco_captions = load_coco_captions(length=10000)

model_name = 'yolo11l.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Ensure using GPU
model = YOLO(model_name).to(device)  # Move model to GPU
raw_result = {}
object_dict = set()


image_folder = '/deepfreeze/junda/datasets/COCO2014/val2014'
questions = coco_captions
image_paths = [os.path.join(image_folder, f"""COCO_val2014_{questions[i]['image_id']:012d}.jpg""") for i in range(len(questions))]

qs = [os.path.join(image_folder, f"""{questions[i]['caption']}""") for i in range(len(questions))]
print("question number:", len(questions), "different image:", len(set(image_paths)))


image_paths = list(set(image_paths))
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


torch.save(object_dict, f'result/coco_feature/coco_caption_object_dict_full_{model_name}.pth')
torch.save(raw_result, f'result/coco_feature/coco_caption_raw_result_full_{model_name}.pth')

print("COCO Caption validation set yolo done!")
print()
print()