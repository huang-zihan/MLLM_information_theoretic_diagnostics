# cd transformers
# git checkout v4.37.2

temperature=0.7

CUDA_VISIBLE_DEVICES=2,7 python cifar.py \
        --model-path "liuhaotian/llava-v1.5-7b" --dataset cifar10 --question-file "$question_path" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

CUDA_VISIBLE_DEVICES=2,7 python cifar.py \
        --model-path "liuhaotian/llava-v1.5-7b" --dataset cifar100 --question-file "$question_path" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"


# CUDA_VISIBLE_DEVICES=1,2 python cifar100.py \
#         --model-path "liuhaotian/llava-v1.5-7b" --question-file "$question_path" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# transformers              4.37.2                   pypi_0    pypi