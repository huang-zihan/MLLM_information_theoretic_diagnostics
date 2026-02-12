# cd transformers
# git checkout v4.37.2

temperature=0.7

CUDA_VISIBLE_DEVICES=0 python ce_datagen.py \
        --model-path "liuhaotian/llava-v1.5-7b" --question "" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

CUDA_VISIBLE_DEVICES=0 python ce_datagen.py \
        --model-path "Qwen/Qwen2.5-VL-3B-Instruct" --question "" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

