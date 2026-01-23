# cd transformers
# git checkout v4.37.2

temperature=0.7

# tmux 4
# CUDA_VISIBLE_DEVICES=7 python vqa_datagen-v1.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" | tee vaq_train_gen_log-1.txt

# CUDA_VISIBLE_DEVICES=3 python vqa_datagen-v1.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "question_only_" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" | tee vaq_train_gen_log-1.txt

# CUDA_VISIBLE_DEVICES=3 python vqa_datagen-v1.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "fixed_question_" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" | tee vaq_train_gen_log-1.txt

# CUDA_VISIBLE_DEVICES=2 python vqa_datagen-v1.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "new" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" | tee vaq_train_gen_log-1.txt

# CUDA_VISIBLE_DEVICES=7 python vqa_datagen-v1.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "origin" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" | tee vaq_train_gen_log-1.txt

# CUDA_VISIBLE_DEVICES=7 python vqa_datagen-v1.py \
#         --model-path "Qwen/Qwen2.5-VL-3B-Instruct" --question "origin" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

CUDA_VISIBLE_DEVICES=6 python vqa_datagen-v1-intervene.py \
        --model-path "Qwen/Qwen2.5-VL-3B-Instruct" --question "origin_intervene_5" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"


# miss object index
# [68, 69, 70, 270, 271, 272, 273, 928, 929, 930, 1240, 1241, 1242, 1502, 1503, 1504, 3823, 3824, 3825, 3826, 3827, 3828, 3829, 3830, 3831, 3912, 3913, 3914, 3915, 3920, 3921, 3922, 3923, 3924, 4171, 4172, 4173, 4174, 4175, 4176, 4177, 5159, 5160, 5161, 5162, 5163, 5164, 5165, 5166, 5167, 5168, 5169, 5170, 5171, 5172, 5173, 5174, 5175, 5176, 5177, 5178, 5179, 5180, 5181, 5182, 5183, 5944, 5945, 5946, 5947, 6007, 6008, 6009, 6803, 6804, 6805, 6806, 6807, 6808, 6809, 6810, 6811, 6812, 7022, 7023, 7024, 7025, 7026, 7027, 7028, 7029, 7030, 7031, 7032, 7033, 7034, 7229, 7230, 7231, 7609, 7610, 7611, 7771, 7772, 7773, 7774, 8054, 8055, 8056, 8270, 8271, 8272, 8505, 8506, 8507, 8508, 8509, 8931, 8932, 8933, 8934, 8935, 9600, 9601, 9602, 9603, 9604, 9605, 9606, 9607, 9608, 9609, 9863, 9864, 9865, 9866, 9867, 9868, 9869, 9870, 9871, 9872, 9873, 9992, 9993, 9994, 9995, 9996, 9997, 9998, 9999]
# [68, 69, 70, 270, 271, 272, 273, 928, 929, 930, 1240, 1241, 1242, 1502, 1503, 1504, 3823, 3824, 3825, 3826, 3827, 3828, 3829, 3830, 3831, 3912, 3913, 3914, 3915, 3920, 3921, 3922, 3923, 3924, 4171, 4172, 4173, 4174, 4175, 4176, 4177, 5159, 5160, 5161, 5162, 5163, 5164, 5165, 5166, 5167, 5168, 5169, 5170, 5171, 5172, 5173, 5174, 5175, 5176, 5177, 5178, 5179, 5180, 5181, 5182, 5183, 5944, 5945, 5946, 5947, 6007, 6008, 6009, 6803, 6804, 6805, 6806, 6807, 6808, 6809, 6810, 6811, 6812, 7022, 7023, 7024, 7025, 7026, 7027, 7028, 7029, 7030, 7031, 7032, 7033, 7034, 7229, 7230, 7231, 7609, 7610, 7611, 7771, 7772, 7773, 7774, 8054, 8055, 8056, 8270, 8271, 8272, 8505, 8506, 8507, 8508, 8509, 8931, 8932, 8933, 8934, 8935, 9600, 9601, 9602, 9603, 9604, 9605, 9606, 9607, 9608, 9609, 9863, 9864, 9865, 9866, 9867, 9868, 9869, 9870, 9871, 9872, 9873, 9992, 9993, 9994, 9995, 9996, 9997, 9998, 9999]