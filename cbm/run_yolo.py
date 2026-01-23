import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from datasets import load_dataset


def runYOLO(images, model_path='yolo11l.pt', device='cuda'):
    model = YOLO(model_path).to(device)
    output_dir = 'yolo_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    image_paths = []
    
    for i, image in enumerate(images):
        image_name = f"image_{i}.jpg"
        image_path = os.path.join(output_dir, image_name)
        arr = np.array(image)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.shape[-1] == 1:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(image_path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        image_paths.append(image_path)            
    # https://docs.ultralytics.com/usage/cfg/#predict-settings
    # https://docs.ultralytics.com/modes/predict/#masks
    annotated_images = []
    detection_results = {}
    batch_size = 128
    for batch_start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[batch_start: batch_start + batch_size]
        with torch.no_grad():
            results = model.predict(batch_paths, device=device, verbose=False, batch=batch_size, conf=0.5, half=True)
        
        for i, (image_path, result) in enumerate(zip(batch_paths, results)):
            image_name = os.path.basename(image_path)
            arr = cv2.imread(image_path)
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

            boxes = []
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.data.tolist()

            detections_dict = {}

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2, conf, cls = box
                class_label = model.names[int(cls)]
                detections_dict[idx + 1] = class_label
                pos = (int(x1), max(int(y1) - 10, 0))
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                text = str(idx+1)
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                x, y = int(x1), max(int(y1) - 10, text_height + baseline)

                padding = 2
                rect_top_left = (x, y - text_height - padding)
                rect_bottom_right = (x + text_width + 2 * padding, y + padding)

                cv2.rectangle(arr, rect_top_left, rect_bottom_right, (0, 0, 0), cv2.FILLED)
                cv2.rectangle(arr, rect_top_left, rect_bottom_right, (0, 0, 0), thickness)
                cv2.putText(arr, text, (x + padding, y - padding), font, font_scale, (255, 255, 255), thickness)

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                # cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # font_scale = 0.5
                # thickness = 1
                # text = str(idx + 1)
                # (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_x = x1
                text_y = max(y1 - 10, text_height)
                cv2.rectangle(arr, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), (0, 0, 0), cv2.FILLED)
                cv2.putText(arr, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            detection_results[f"annotated_{image_name}"] = detections_dict

            annotated_image_path = os.path.join(output_dir, f"annotated_{image_name}")
            cv2.imwrite(annotated_image_path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

            annotated_image = Image.fromarray(arr)
            annotated_images.append(annotated_image)

    results_path = os.path.join(output_dir, "detection_results.json")
    with open(results_path, "w") as f:
        json.dump(detection_results, f, indent=4)

    return annotated_images, detection_results


def main():
    dataset = load_dataset("lmms-lab/POPE", split="test") 
    images = dataset['image']
    annotated_images, detection_results = runYOLO(images)

    print("Processing complete.")

if __name__ == "__main__":
    main()