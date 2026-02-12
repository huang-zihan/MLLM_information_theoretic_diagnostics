# cd transformers
# git checkout v4.37.2

temperature=0.7

### Yolo 11 dictionary:
# names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

####################################################
CUDA_VISIBLE_DEVICES=0 python run-pope-cbm.py \
        --model-path "liuhaotian/llava-v1.5-7b" --question "" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# --question ""
# --question "origin"
# --question "visual_prompt"
# --question "visual_prompt-1new"
# --question "visual_prompt-2"
# --question "popenew"


## For researching the influence of textual prompts
# --question "indom"
# --question "outdom"
# --question "amb_desc"
# --question "insuff_att"
# --question "insuff_att-2"

############################################

CUDA_VISIBLE_DEVICES=6 python run-pope-cbm.py \
        --model-path "Qwen/Qwen2.5-VL-3B-Instruct" --question "" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# --question ""
# --question "amb_desc-intervene"
# --question "amb_desc-intervene_3"
# --question "visual_prompt"
# --question "visual_prompt-1"
# --question "visual_prompt-2"


