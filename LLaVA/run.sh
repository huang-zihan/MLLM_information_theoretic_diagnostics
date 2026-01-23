# cd transformers
# git checkout v4.37.2

temperature=0.7

CUDA_VISIBLE_DEVICES=0,1 python run.py \
        --model-path "liuhaotian/llava-v1.5-7b" --question-file "$question_path" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"

# transformers              4.37.2                   pypi_0    pypi