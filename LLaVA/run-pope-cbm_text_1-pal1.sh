# cd transformers
# git checkout v4.37.2

temperature=0.7

### Yolo 11 dictionary:
# names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# CUDA_VISIBLE_DEVICES=7 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "insuff_att" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=7 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "amb_desc" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"



THRESHOLD=17000
temperature=0.7
# CUDA_VISIBLE_DEVICES=6 python run-pope-cbm-v1label.py \
# --model-path "liuhaotian/llava-v1.5-7b" --question "insuff_att-1" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

CUDA_VISIBLE_DEVICES=3 python run-pope-cbm-v1label.py \
--model-path "liuhaotian/llava-v1.5-7b" --question "amb_desc-2" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"


# while true; do
#     for i in {1..1}; do
#         # 获取每张 GPU 的空闲显存
#         free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $i)

#         echo "GPU $i 当前空闲显存: ${free_memory} MB"

#         # 判断空闲显存是否大于阈值
#         if [ "$free_memory" -gt "$THRESHOLD" ]; then
#             echo "找到合适的 GPU: $i，启动任务..."
#             CUDA_VISIBLE_DEVICES=1 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "mislead" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"
#             exit 0  # 启动任务后退出脚本
#         fi
#     done

#     # 暂停一段时间后再检查，避免频繁查询
#     sleep 30
# done

# CUDA_VISIBLE_DEVICES=1 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "mislead" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"


####
########Insufficient Context Attention （add irrelavant information to reduce the attention on visual info）
# Generating Fluent but Inaccurate Responses： Based on this image, describe a scene, but ensure your description sounds very fluent and natural. You can overlook inconsistent details and focus solely on smooth expression.
# Focusing on Partial Information： This image contains various elements. Try to answer based only on the specific part you see at the moment, ignoring other potentially important visual information. For instance, if you only see a person's hand, describe their hand without mentioning the rest of their body or the environment.
# concentrating on partial summarizing tokens of the generated content ： Based on the top section of this image, provide a brief summary and description while ignoring the content in the bottom and other areas.
########Capability Misalignment (ask model to do very strange tasks on image VQA)

# Ambiguous Environment Description: In this image of a forest, there's a mysterious creature hiding. What might it look like? Please describe its features and behavior in detail.
# Beyond Common Knowledge Inquiry:If the child and the animals in this picture could communicate, what strange stories might they discuss about seasonal changes?
# Time and Place Mixing: "Imagine this image of a modern city park is set in a future world. What activities might people be doing differently?"
# Complex Hypothetical Scenario: "Envision this picture depicting a gathering in the rain. How might participants use their surroundings to create fun?"
# Inconsistent Information: "In this spring flower field photo, what elements that don’t belong to spring might appear in the background? Describe these elements and their possible significance."

####


# CUDA_VISIBLE_DEVICES=7 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "irrelevant-2" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=7 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "irrelevant-3" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=7 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "irrelevant-4" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=7 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "irrelevant-5" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=1 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "python code" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=1 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "email" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=1 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "aircraft" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=1 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "operating system" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=1 python run-pope-cbm-v1label.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "alien" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

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