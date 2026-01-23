# cd transformers
# git checkout v4.37.2

temperature=0.7

# CUDA_VISIBLE_DEVICES=4,5 python run-pope.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question "car" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

CUDA_VISIBLE_DEVICES=0,1 python run-pope.py \
        --model-path "liuhaotian/llava-v1.5-7b" --question "" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"


CUDA_VISIBLE_DEVICES=0,1 python run-pope-global.py \
        --model-path "liuhaotian/llava-v1.5-7b" --question-file "" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# keyboard
# car

# transformers              4.37.2                   pypi_0    pypi