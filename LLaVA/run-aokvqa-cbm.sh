# cd transformers
# git checkout v4.37.2

temperature=0.7

### Yolo 11 dictionary:
# names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# CUDA_VISIBLE_DEVICES=1 python run-aokvqa-cbm-v1.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "correct" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"
# # [795, 875, 976, 1074]

# CUDA_VISIBLE_DEVICES=1 python run-aokvqa-cbm-v1.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "wrong" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"
# # [2385, 2386, 2387, 2625, 2626, 2627, 2928, 2929, 2930, 3222, 3223, 3224]


# CUDA_VISIBLE_DEVICES=5 python run-aokvqa-cbm-v1.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "newcorrect" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"
# # [795, 875, 976, 1074]

# CUDA_VISIBLE_DEVICES=5 python run-aokvqa-cbm-v1.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "newwrong" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"
# # [2385, 2386, 2387, 2625, 2626, 2627, 2928, 2929, 2930, 3222, 3223, 3224]

# CUDA_VISIBLE_DEVICES=0 python run-aokvqa-cbm-v1.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "origincorrect" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"
# # [795, 875, 976, 1074]

# CUDA_VISIBLE_DEVICES=0 python run-aokvqa-cbm-v1.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "originwrong" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"
# # [2385, 2386, 2387, 2625, 2626, 2627, 2928, 2929, 2930, 3222, 3223, 3224]


# CUDA_VISIBLE_DEVICES=7 python run-aokvqa-cbm-v1.py \
#         --model-path "Qwen/Qwen2.5-VL-3B-Instruct" --question "origincorrect" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"
# # [795, 875, 976, 1074]

CUDA_VISIBLE_DEVICES=7 python run-aokvqa-cbm-v1.py \
        --model-path "Qwen/Qwen2.5-VL-3B-Instruct" --question "originwrong" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"
# [2385, 2386, 2387, 2625, 2626, 2627, 2928, 2929, 2930, 3222, 3223, 3224]


