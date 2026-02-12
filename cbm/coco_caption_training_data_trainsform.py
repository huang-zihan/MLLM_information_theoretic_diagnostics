import os
import zipfile
import requests
import json

import torch
import numpy as np
import math
import json
import os

def download_coco_captions(url, download_path):
    # (Function implementation omitted for brevity; assume it downloads and returns the path)
    return os.path.join(download_path, 'annotations', 'captions_val2014.json')

def load_coco_captions(length=10000):
    # URL for COCO2014 training annotations
    url = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
    download_path = '/deepfreeze/junda/datasets/COCO2014_caption'
    
    # Download and get the annotation file path
    annotation_file = download_coco_captions(url, download_path)

    # Read JSON file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Create a list to store image IDs and their captions
    coco_captions = []
    
    # Iterate over annotations, extract image ID and caption
    for i, annotation in enumerate(annotations['annotations']):
        if i == length:
            break
        
        image_id = annotation['image_id']
        caption = annotation['caption']
        
        coco_captions.append(
            {"image_id": image_id, "caption": caption}
        )
        
    return coco_captions

# Example usage
coco_captions = load_coco_captions(length=10000)

model_name = 'yolo11l.pt'

# Dictionary mapping object names to indices
object_dict = torch.load(f'result/coco_feature/coco_caption_object_dict_full_{model_name}.pth')
# Dictionary mapping image names to object scores
raw_result = torch.load(f'result/coco_feature/coco_caption_raw_result_full_{model_name}.pth')

# Number of object classes
num_classes = 80

# Initialize dataset
dataset = []

def softmax(lst):
    # Compute sum of exponentials of all elements
    sum_exp = sum(lst)
    
    # Compute softmax values
    softmax_lst = [x / sum_exp for x in lst]
    
    return softmax_lst

# Iterate over raw results
image_folder = '/deepfreeze/junda/datasets/COCO2014/val2014'

questions = coco_captions
image_paths = [os.path.join(image_folder, f"""COCO_val2014_{questions[i]['image_id']:012d}.jpg""") for i in range(len(questions))]

qs = [os.path.join(image_folder, f"""{questions[i]['caption']}""") for i in range(len(questions))]
print("question number:", len(questions), "different image:", len(set(image_paths)))

for i, item in enumerate(questions):
    image_name = f"""COCO_val2014_{item['image_id']:012d}.jpg"""
    scores = raw_result[image_name]
    print(i, "/", len(questions), end='\r')
    
    if len(scores) == 0:
        # No objects detected
        data_item = {
            'question': item['caption'],
            'image': image_name,
            'annotation': None
        }
        dataset.append(data_item)
        continue
    
    # Initialize annotation list with zeros
    annotation = [0] * num_classes
    
    # Fill annotation list
    for obj_index, score in scores.items():
        annotation[obj_index] = sum(score)
    annotation = softmax(annotation)
    
    # Build data item
    data_item = {
        'question': item['caption'],
        'image': image_name,
        'annotation': annotation
    }
    
    # Add to dataset
    dataset.append(data_item)

# Save the dataset
torch.save(dataset, f"coco_caption_ce_data_full_v1.pth")