import torch
import numpy as np
import math

model_name = 'yolo11l.pt'

# Load data
object_dict = torch.load(f'result/coco_feature/object_dict_full_{model_name}.pth')  # Mapping of object names to indices
raw_result = torch.load(f'result/coco_feature/raw_result_full_{model_name}.pth')  # Mapping of images to object scores

# Get number of objects
num_classes = 80 #len(object_dict)

# Initialize dataset
dataset = []

def softmax(lst):
    # Compute the sum of exponentials of all elements
    sum_exp = sum(lst)
    
    # Compute Softmax values
    softmax_lst = [x / sum_exp for x in lst]
    
    return softmax_lst

# Iterate over original results
for i, (image_name, scores) in enumerate(raw_result.items()):
    
    print(i, "/", len(raw_result), end='\r')
    
    if len(scores)==0:
        # continue
        data_item = {
            'image': image_name,
            'annotation': None
        }
        dataset.append(data_item)
        continue
    
    # Create annotation list, initialize with zeros
    annotation = [0] * num_classes
    
    # Fill annotation list
    for obj_index, score in scores.items():
        annotation[obj_index] = sum(score)
    annotation = softmax(annotation)
    
    # Build data item
    data_item = {
        'image': image_name,
        'annotation': annotation
    }
    
    # Add to dataset
    dataset.append(data_item)

torch.save(dataset, f"ce_data_full_v1.pth")