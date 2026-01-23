# cd transformers
# git checkout v4.37.2

temperature=0.7

CUDA_VISIBLE_DEVICES=5 python run-pope.py \
        --model-path "liuhaotian/llava-v1.5-7b" --question "" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4 python run-pope.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "global" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4 python run-pope.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "dog" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4 python run-pope.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "chair" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4 python run-pope.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "keyboard" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4 python run-pope.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "backpack" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"



# CUDA_VISIBLE_DEVICES=1,2 python run-pope.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "backpack" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"


# CUDA_VISIBLE_DEVICES=4,5 python run-pope-cross.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "car" --cross_question "dining table" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4,5 python run-pope-cross.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "car" --cross_question "car" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4,5 python run-pope-cross.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "dining table" --cross_question "car" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# CUDA_VISIBLE_DEVICES=4,5 python run-pope-cross.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "dining table" --cross_question "dining table" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"


# CUDA_VISIBLE_DEVICES=0,1 python run-pope-counterfact.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "apple" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"



# CUDA_VISIBLE_DEVICES=6,7 python run-pope.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"


# CUDA_VISIBLE_DEVICES=2,3 python run-pope.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "keyboard" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"


# CUDA_VISIBLE_DEVICES=2,3 python run-pope-global.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question-file "car" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# # keyboard
# car

# transformers              4.37.2                   pypi_0    pypi