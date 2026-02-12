import os
import torch
import numpy as np
from datasets import load_dataset
from ultralytics import YOLO
import cv2

# Load data
object_dict = None
visual_prompt=False


# Initialize dataset
dataset = []

def softmax(lst):
    # Compute the sum of exponentials of all elements
    sum_exp = sum(lst)
    
    # Compute Softmax values
    softmax_lst = [x / sum_exp for x in lst]
    
    return softmax_lst

train_dataset = load_dataset("HuggingFaceM4/A-OKVQA", split="validation", cache_dir='/deepfreeze/junda/datasets/')
model_name = 'yolo11l.pt'  # Use higher version model
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Ensure using GPU
model = YOLO(model_name).to(device)  # Load higher version model

if not visual_prompt:
    # Create output directory
    output_dir = 'result/aokvqa_feature/img/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
else:
    output_dir = '/home/shared/rohan/vd-llm/yolo_output/'

# correct answer
for i, item in enumerate(train_dataset):
    
    print(item)
    exit()
    
    print(i, "/", len(train_dataset), end='\r')
    
    if not visual_prompt:
        # Save image to file
        image = item['image'].convert('RGB')
        image_path = os.path.join(output_dir, f'image_{i}.jpg')
        cv2.imwrite(image_path, np.array(image)[:, :, ::-1])  # Save in BGR format
    else:
        image_path = os.path.join(output_dir, f'annotated_image_{i}.jpg')
        
    # Perform inference
    with torch.no_grad():
        results = model.predict(image_path, device=device, verbose=False)  # Use new inference method
        if object_dict is None:
            # print(results[0])
            object_dict=results[0].names
            # Get number of objects
            num_classes = len(object_dict)


    # Process results
    detections = results[0]  # Get detection results
    annotation = [0] * len(object_dict)
    
    for box in detections.boxes:  # Iterate over each detection box
        cls = int(box.cls.item())  # Get class
        conf = float(box.conf.item())  # Get confidence
        annotation[cls] += conf

    # print(annotation)
    if sum(annotation)!=0:
        annotation = softmax(annotation)
    else:
        annotation = []
    
    # Build data item
    data_item = {
        'id': i,
        'annotation': annotation
    }
    
    # Add to dataset
    dataset.append(data_item)

# Save results
if not visual_prompt:
    torch.save(dataset, f"result/aokvqa_feature/aokvqa_ce_test_data_{model_name}.pth")
else:
    torch.save(dataset, f"result/aokvqa_feature/vp_aokvqa_ce_test_data_{model_name}.pth")
print("aokvqa yolo done!")
print()
print()