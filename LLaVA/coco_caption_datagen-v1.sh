# cd transformers
# git checkout v4.37.2
THRESHOLD=20000
# THRESHOLD=8000
temperature=0.7

# while true; do
#     for i in {0..7}; do
#         # 获取每张 GPU 的空闲显存
#         free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $i)

#         echo "GPU $i 当前空闲显存: ${free_memory} MB"

#         # 判断空闲显存是否大于阈值
#         if [ "$free_memory" -gt "$THRESHOLD" ]; then
#             echo "找到合适的 GPU: $i，启动任务..."
#             CUDA_VISIBLE_DEVICES=${i} python coco_caption_datagen-v1.py \
#                 --model-path "liuhaotian/llava-v1.5-7b" \
#                 --question "" \
#                 --answers-file "$answer_path" \
#                 --image-folder "$image_folder" \
#                 --temperature "$temperature" | tee vaq_train_gen_log-1.txt
#             exit 0  # 启动任务后退出脚本
#         fi
#     done

#     # 暂停一段时间后再检查，避免频繁查询
#     sleep 60
# done

# # tmux 4
# CUDA_VISIBLE_DEVICES=7 python coco_caption_datagen-v1.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "question_only_" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" | tee vaq_train_gen_log-1.txt

# CUDA_VISIBLE_DEVICES=3 python coco_caption_datagen-v1.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "fixed_question_" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" | tee vaq_train_gen_log-1.txt

# CUDA_VISIBLE_DEVICES=2 python coco_caption_datagen-v1.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "new" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" | tee vaq_train_gen_log-1.txt

# CUDA_VISIBLE_DEVICES=6 python coco_caption_datagen-v1.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "origin" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" | tee vaq_train_gen_log-1.txt

CUDA_VISIBLE_DEVICES=6 python coco_caption_datagen-v1.py \
        --model-path "Qwen/Qwen2.5-VL-3B-Instruct" --question "origin" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" # | tee vaq_train_gen_log-1.txt


# miss object index
# [73, 98, 188, 190, 232, 235, 258, 267, 275, 287, 297, 309, 316, 338, 365, 423, 470, 480, 483, 657, 1256, 1284, 1323, 1453, 1484, 1654, 1701, 1717, 1726, 1729, 1794, 1796, 1815, 1830, 1871, 1891, 1942, 1954, 2004, 2017, 2037, 2038, 2082, 2096, 2144, 2207, 2282, 2373, 2477, 2502, 2529, 2631, 2778, 2850, 2912, 2941, 2952, 2953, 3013, 3022, 3049, 3067, 3069, 3081, 3185, 3188, 3196, 3202, 3212, 3242, 3296, 3309, 3344, 3355, 3494, 3611, 3687, 3699, 3797, 3824, 3887, 3892, 3999, 4003, 4081, 4819, 4855, 4862, 4980, 5001, 5029, 5096, 5112, 5189, 5280, 5309, 5393, 5523, 5537, 5618, 5669, 5784, 5815, 5854, 5992, 6059, 6063, 6101, 6125, 6312, 6456, 6590, 6597, 6611, 6632, 6666, 6720, 6728, 6731, 6763, 7493, 7523, 7605, 7675, 7694, 7717, 7773, 7778, 7892, 7895, 8206, 8521, 8533, 8689, 8707, 8759, 8802, 8821, 8825, 8921, 8951, 9020, 9021, 9073, 9112, 9134, 9139, 9147, 9251, 9257, 9320, 9363, 9430, 9431, 9440, 9477, 9484, 9490, 9518, 9538, 9541, 9564, 9603, 9606, 9689, 9721, 9740, 9754, 9864, 9932, 9974, 9992]
# [73, 98, 188, 190, 232, 235, 258, 267, 275, 287, 297, 309, 316, 338, 365, 423, 470, 480, 483, 657, 1256, 1284, 1323, 1453, 1484, 1654, 1701, 1717, 1726, 1729, 1794, 1796, 1815, 1830, 1871, 1891, 1942, 1954, 2004, 2017, 2037, 2038, 2082, 2096, 2144, 2207, 2282, 2373, 2477, 2502, 2529, 2631, 2778, 2850, 2912, 2941, 2952, 2953, 3013, 3022, 3049, 3067, 3069, 3081, 3185, 3188, 3196, 3202, 3212, 3242, 3296, 3309, 3344, 3355, 3494, 3611, 3687, 3699, 3797, 3824, 3887, 3892, 3999, 4003, 4081, 4819, 4855, 4862, 4980, 5001, 5029, 5096, 5112, 5189, 5280, 5309, 5393, 5523, 5537, 5618, 5669, 5784, 5815, 5854, 5992, 6059, 6063, 6101, 6125, 6312, 6456, 6590, 6597, 6611, 6632, 6666, 6720, 6728, 6731, 6763, 7493, 7523, 7605, 7675, 7694, 7717, 7773, 7778, 7892, 7895, 8206, 8521, 8533, 8689, 8707, 8759, 8802, 8821, 8825, 8921, 8951, 9020, 9021, 9073, 9112, 9134, 9139, 9147, 9251, 9257, 9320, 9363, 9430, 9431, 9440, 9477, 9484, 9490, 9518, 9538, 9541, 9564, 9603, 9606, 9689, 9721, 9740, 9754, 9864, 9932, 9974, 9992]