# cd transformers
# git checkout v4.37.2

temperature=0.7

# tmux 1
# CUDA_VISIBLE_DEVICES=3 python hal_datagen-v1.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "correct" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" | tee vaq_train_gen_log-1.txt

# CUDA_VISIBLE_DEVICES=2 python hal_datagen-v1.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "wrong" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" | tee vaq_train_gen_log-1.txt

# miss object index

# CUDA_VISIBLE_DEVICES=7 python hal_datagen-v1.py \
#         --model-path "Qwen/Qwen2.5-VL-3B-Instruct" --question "correctorigin" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"  #| tee vaq_train_gen_log-1.txt


# CUDA_VISIBLE_DEVICES=6 python hal_datagen-v1.py \
#         --model-path "Qwen/Qwen2.5-VL-3B-Instruct" --question "wrongvisual_prompt" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" # | tee vaq_train_gen_log-1.txt

# CUDA_VISIBLE_DEVICES=6 python hal_datagen-v1.py \
#         --model-path "Qwen/Qwen2.5-VL-3B-Instruct" --question "wrongvisual_prompt-1" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" # | tee vaq_train_gen_log-1.txt


CUDA_VISIBLE_DEVICES=6 python hal_datagen-v1.py \
        --model-path "Qwen/Qwen2.5-VL-3B-Instruct" --question "wrongvisual_prompt-2" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" # | tee vaq_train_gen_log-1.txt

CUDA_VISIBLE_DEVICES=6 python hal_datagen-v1.py \
        --model-path "Qwen/Qwen2.5-VL-3B-Instruct" --question "correctvisual_prompt" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" # | tee vaq_train_gen_log-1.txt

CUDA_VISIBLE_DEVICES=6 python hal_datagen-v1.py \
        --model-path "Qwen/Qwen2.5-VL-3B-Instruct" --question "correctvisual_prompt-1" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" # | tee vaq_train_gen_log-1.txt

CUDA_VISIBLE_DEVICES=6 python hal_datagen-v1.py \
        --model-path "Qwen/Qwen2.5-VL-3B-Instruct" --question "correctvisual_prompt-2" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" # | tee vaq_train_gen_log-1.txt



