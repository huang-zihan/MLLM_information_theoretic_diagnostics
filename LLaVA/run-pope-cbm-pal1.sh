# cd transformers
# git checkout v4.37.2

temperature=0.7

### Yolo 11 dictionary:
# names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# CUDA_VISIBLE_DEVICES=2,5 python run-pope-cbm.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=1 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "visual_prompt" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "visual_prompt-1" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "visual_prompt-2new" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# [2, 244, 360, 408, 502, 508, 788, 984, 1000, 1032, 1176, 1188, 1218, 1594, 1616, 1628, 1658, 1758, 1892, 1898, 2096, 2174, 2364, 2576, 2700, 2778, 3002, 3244, 3360, 3408, 3502, 3508, 3788, 3984, 4000, 4032, 4176, 4188, 4218, 4594, 4616, 4628, 4658, 4758, 4892, 4898, 5096, 5174, 5364, 5576, 5700, 5778, 6002, 6244, 6360, 6408, 6502, 6508, 6788, 6984, 7000, 7032, 7176, 7188, 7218, 7594, 7616, 7628, 7658, 7758, 7892, 7898, 8096, 8174, 8364, 8576, 8700, 8778]

# CUDA_VISIBLE_DEVICES=3 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "popenew" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"



################################################################
# CUDA_VISIBLE_DEVICES=4 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "person" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "bus" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "car" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "airplane" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "bird" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "dog" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "elephant" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "fork" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "cell phone" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "tv" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"
################################################################



# CUDA_VISIBLE_DEVICES=5,7 python run-pope-cbm.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "global" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"





# CUDA_VISIBLE_DEVICES=4 python run-pope-cbm.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "dog" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4 python run-pope-cbm.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "chair" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4 python run-pope-cbm.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "keyboard" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4 python run-pope-cbm.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "backpack" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# # keyboard
# car

# transformers              4.37.2                   pypi_0    pypi


# CUDA_VISIBLE_DEVICES=6 python run-pope-cbm-v1label.py \
#         --model-path "Qwen/Qwen2.5-VL-3B-Instruct" --question "amb_desc-intervene" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"


CUDA_VISIBLE_DEVICES=6 python run-pope-cbm-v1label.py \
        --model-path "Qwen/Qwen2.5-VL-3B-Instruct" --question "amb_desc-intervene_3" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

