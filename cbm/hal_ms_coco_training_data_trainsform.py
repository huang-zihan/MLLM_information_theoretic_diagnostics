import torch
import numpy as np
import math
import json
import os

model_name = 'yolo11l.pt'

# Generate MSCOCO training
training_set=True


object_dict = torch.load(f'result/coco_feature/hal_object_dict_full_{model_name}.pth')  # Mapping of object names to indices
raw_result = torch.load(f'result/coco_feature/hal_raw_result_full_{model_name}.pth')  # Mapping of images to object scores

# Get number of objects
num_classes = 80

def softmax(lst):
    # Compute the sum of exponentials of all elements
    sum_exp = sum(lst)
    
    # Compute Softmax values
    softmax_lst = [x / sum_exp for x in lst]
    
    return softmax_lst


# Iterate over original results
image_folder = '/deepfreeze/junda/datasets/COCO2014/val2014'

file_path = 'hal_eval/in_domain_evaluation.json'
# Load JSON file
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
hal_questions=[]
questions=[]
# image_paths=[]
for item in data:
    hal_captions = item['hal_caption']
    # image_paths.append()
    for hal in hal_captions:
        hal_questions.append(
            {
                'image_name': os.path.basename(item['image']),
                'question': hal['caption'],
                'hal_type': hal['type']   
            }
        )
        
    questions.append(
        {
            'image_name': os.path.basename(item['image']),
            'question': item['caption'],
        }
    )


### Hallucinated results
dataset = []

for i, item in enumerate(hal_questions):

    scores=raw_result[item['image_name']]
    print(i, "/", len(hal_questions), end='\r')
    
    if len(scores)==0:
        # continue
        data_item = {
            'question': item['question'],
            'image': item['image_name'],
            'hal_type': item['hal_type'],
            'annotation': None
        }
        dataset.append(data_item)
        continue
    
    # Create annotation list, initialize with zeros
    annotation = [0] * num_classes
    
    # Fill annotation list
    for obj_index, score in scores.items():
        annotation[obj_index] = sum(score)
    annotation=softmax(annotation)
    
    # Build data item
    data_item = {
        'question': item['question'],
        'image': item['image_name'],
        'hal_type': item['hal_type'],
        'annotation': annotation
    }
    
    # Add to dataset
    dataset.append(data_item)

# print(dataset[0])
torch.save(dataset, f"wrong_hal_ce_data_full_v1.pth")


### Correct results
# Initialize dataset
dataset = []

for i, item in enumerate(questions):

    scores=raw_result[item['image_name']]
    print(i, "/", len(questions), end='\r')
    
    if len(scores)==0:
        # continue
        data_item = {
            'question': item['question'],
            'image': item['image_name'],
            'annotation': None
        }
        dataset.append(data_item)
        continue
    
    # Create annotation list, initialize with zeros
    annotation = [0] * num_classes
    
    # Fill annotation list
    for obj_index, score in scores.items():
        annotation[obj_index] = sum(score)
    annotation=softmax(annotation)
    
    # Build data item
    data_item = {
        'question': item['question'],
        'image': item['image_name'],
        'annotation': annotation
    }
    
    # Add to dataset
    dataset.append(data_item)

# print(dataset[0])
torch.save(dataset, f"correct_hal_ce_data_full_v1.pth")