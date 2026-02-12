import torch
import numpy as np
import math
import json
import os

model_name = 'yolo11l.pt'

# Generate MSCOCO training
training_set=False

if training_set:
    object_dict = torch.load(f'result/coco_feature/object_dict_full_{model_name}.pth')  # Mapping of object names to indices
    raw_result = torch.load(f'result/coco_feature/raw_result_full_{model_name}.pth')  # Mapping of images to object scores
else:
    object_dict = torch.load(f'result/coco_feature/vqa_object_dict_full_{model_name}.pth')  # Mapping of object names to indices
    raw_result = torch.load(f'result/coco_feature/vqa_raw_result_full_{model_name}.pth')  # Mapping of images to object scores

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
if training_set:
    image_folder = '/deepfreeze/junda/datasets/COCO2014/train2014'
    questions_path = '/deepfreeze/junda/datasets/VQAv2/v2_OpenEnded_mscoco_train2014_questions.json'
else:
    image_folder = '/deepfreeze/junda/datasets/COCO2014/val2014'
    questions_path = '/deepfreeze/junda/datasets/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json'
    annotations_path = '/deepfreeze/junda/datasets/VQAv2/v2_mscoco_val2014_annotations.json'


with open(questions_path, 'r') as f:
    questions = json.load(f)

with open(annotations_path, 'r') as f:
    annotations = json.load(f)

if training_set:
    questions = questions['questions'] #[:10000]
    annotations = annotations['annotations'] #[:10000]
    image_paths = [os.path.join(image_folder, f"""COCO_train2014_{questions[i]['image_id']:012d}.jpg""") for i in range(len(questions))]
else:
    questions = questions['questions'][:10000]
    annotations = annotations['annotations'][:10000]
    image_paths = [os.path.join(image_folder, f"""COCO_val2014_{questions[i]['image_id']:012d}.jpg""") for i in range(len(questions))]

qs = [os.path.join(image_folder, f"""{questions[i]['question']}""") for i in range(len(questions))]

for i, item in enumerate(questions):
    if training_set:
        image_name=f"""COCO_train2014_{item['image_id']:012d}.jpg"""
    else:
        image_name=f"""COCO_val2014_{item['image_id']:012d}.jpg"""
    scores=raw_result[image_name]
    print(i, "/", len(questions), end='\r')
    assert annotations[i]['question_id']==item['question_id']
    if len(scores)==0:
        # continue
        data_item = {
            'question': item['question'],
            'response': annotations[i]['answers'][0]['answer'],
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
    annotation=softmax(annotation)
    
    # Build data item
    data_item = {
        'question': item['question'],
        'response': annotations[i]['answers'][0]['answer'],
        'image': image_name,
        'annotation': annotation
    }
    
    # Add to dataset
    dataset.append(data_item)

if training_set:
    torch.save(dataset, f"vqa_ce_data_full_v1_train.pth")
else:
    torch.save(dataset, f"vqa_ce_data_full_v1.pth")