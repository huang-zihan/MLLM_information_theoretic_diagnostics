import argparse
import torch
torch._C._cuda_init()
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from datasets import load_dataset

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


ce_data_dir="../cbm/data/train/"


class MSCOCODataset(Dataset):
    def __init__(self, annotations_file, image_dir, image_processor, model, model_path=None):
        self.image_dir = image_dir
        self.image_processor = image_processor
        self.model = model
        self.model_path = model_path
        self.missing_index=[]
        
        # 读取 annotation 文件
        self.dataset = torch.load(annotations_file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # item = self.dataset.items()[idx]
        # print(idx)
        item = self.dataset[idx]
        question = "List all items on this image."  # 固定问题
        image_path = os.path.join(self.image_dir, item['image'])
        image = Image.open(image_path).convert('RGB')
        label=item['annotation']
        if label is None:
            label=[]
            self.missing_index.append(idx)
        image_sizes = image.size

        # 将 PIL 图像转换为张量
        if "qwen" in self.model_path.lower():        
            image_tensor = image_path
        else:            
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]
            
        return {
            'image': image_tensor,
            'question': question,
            'label': label,
            'image_sizes': image_sizes,
            'image_source': item['image']  # 或者其他相关信息
        }
        # return {
        #     'label': label
        # }
        
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    if "qwen" in model_path.lower():
        image_processor = AutoProcessor.from_pretrained(model_path)
    

    # if args.lora_ckpt != "":
    #     model.load_adapter(args.lora_ckpt)
    #     print(f"adapter {args.lora_ckpt} loaded")
    
    # if args.pretrain_vison != "":
    #     non_lora_state_dict = torch.load(args.pretrain_vison)
    #     new_state_dict = {}
        
    #     if args.pretrain_mm_mlp_adapter and args.lora_ckpt:
    #         prefix="base_model.model.model.vision_tower."
    #         for key, value in non_lora_state_dict.items():
    #             new_key = key.replace(prefix, '')
    #             new_state_dict[new_key] = value
    #         model.get_model().vision_tower.load_state_dict(new_state_dict) # , strict=False
    #     else:
    #         prefix="model.vision_tower."
    #         for key, value in non_lora_state_dict.items():
    #             # print(key)
    #             new_key = key.replace(prefix, '')
    #             new_state_dict[new_key] = value
    #         model.get_model().vision_tower.load_state_dict(new_state_dict)
            
    #     print(f"vision {args.pretrain_vison} loaded")
        
    
    # if args.pretrain_mm_mlp_adapter != "":
        
    #     non_lora_state_dict = torch.load(args.pretrain_mm_mlp_adapter)
    #     new_state_dict = {}
        
    #     if args.lora_ckpt:
    #         prefix="base_model.model.model.mm_projector."
    #         for key, value in non_lora_state_dict.items():
    #             new_key = key.replace(prefix, '')
    #             new_state_dict[new_key] = value

    #         model.get_model().mm_projector.load_state_dict(new_state_dict) # , strict=False
    #     else:            
    #         prefix="base_model.model.model.mm_projector."
            
    #         for key, value in non_lora_state_dict.items():
    #             new_key = key.replace(prefix, '')
    #             new_state_dict[new_key] = value
    #         model.get_model().mm_projector.load_state_dict(new_state_dict)
        
    #     print(f"mmp {args.pretrain_mm_mlp_adapter} loaded")

    item_dict = {'snowboard': 0, 'backpack': 1, 'person': 2, 'car': 3, 'skis': 4, 'dog': 5, 'truck': 6, 'dining table': 7, 'handbag': 8, 'bicycle': 9, 'motorcycle': 10, 'potted plant': 11, 'vase': 12, 'traffic light': 13, 'bus': 14, 'chair': 15, 'bed': 16, 'book': 17, 'spoon': 18, 'cup': 19, 'fork': 20, 'tv': 21, 'toaster': 22, 'microwave': 23, 'bottle': 24, 'bird': 25, 'boat': 26, 'couch': 27, 'sandwich': 28, 'bowl': 29, 'hot dog': 30, 'frisbee': 31, 'knife': 32, 'cake': 33, 'remote': 34, 'baseball glove': 35, 'sports ball': 36, 'baseball bat': 37, 'bench': 38, 'sink': 39, 'toilet': 40, 'teddy bear': 41, 'bear': 42, 'cat': 43, 'mouse': 44, 'laptop': 45, 'toothbrush': 46, 'cow': 47, 'skateboard': 48, 'surfboard': 49, 'cell phone': 50, 'train': 51, 'clock': 52, 'tennis racket': 53, 'suitcase': 54, 'horse': 55, 'banana': 56, 'wine glass': 57, 'refrigerator': 58, 'carrot': 59, 'broccoli': 60, 'tie': 61, 'scissors': 62, 'sheep': 63, 'airplane': 64, 'stop sign': 65, 'fire hydrant': 66, 'keyboard': 67, 'pizza': 68, 'donut': 69, 'kite': 70, 'parking meter': 71, 'giraffe': 72, 'zebra': 73, 'umbrella': 74, 'orange': 75, 'oven': 76, 'elephant': 77, 'apple': 78}
    
    
    # image_dir = '/deepfreeze/junda/datasets/COCO2014/train2014'
    image_dir = '/deepfreeze/zihan/deepfreeze/junda/datasets/COCO2014/train2014'
    
    # annotations_file = '/home/junda/zihan/cbm/ce_data.pth'  
    # annotations_file = '/home/junda/zihan/cbm/result/coco_feature/ce_data_full.pth'  
    # annotations_file = '/home/junda/zihan/cbm/ce_data_full_v1.pth'  
    annotations_file = '../cbm/ce_data_full_v1.pth'
    
    mscoco_dataset = MSCOCODataset(annotations_file, image_dir, image_processor, model, model_path=model_path)

    # Create data loader for the POPE dataset
    train_loader = torch.utils.data.DataLoader(
        dataset=mscoco_dataset,
        batch_size=1,
        shuffle=False
    )
    if "qwen" in model_path.lower():
        info_save_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
    else:
        info_save_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    info_save_list=[[] for _ in range(len(info_save_index))]
    image_info_save_list=[[] for _ in range(len(info_save_index))]
    label_list=[]
    output_list = []
    question_list = []
    vit_feature_list=[]
    temp=[0]*len(item_dict)
    for i, item in enumerate(train_loader):
        
        label = [float(x[0]) for x in item['label']]  # Assuming 'label' key exists
        # item['image_sizes']=(item['image_sizes'][0][0], item['image_sizes'][1][0])

        print(i, "/", len(train_loader), end="\r")
        
        qs=item['question'][0] #"Please directly answer what is the number digit on the image by a digit."
        label_list.append(label)
        
        if "qwen" in model_path.lower():
            
            image = Image.open(item['image'][0]).convert('RGB')
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"{qs}"}
                    ]
                }
            ]
            
            instruction_token = tokenizer(qs)['input_ids']
            # print("question len", instruction_token, len(instruction_token))
            
            text = image_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, _ = process_vision_info(messages)
            
            inputs = image_processor(
                text=[text],
                images=image_inputs,
                # videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
        else:
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' +  qs
            
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image_tensor=item['image']
            
        
        with torch.inference_mode():
            
            # output = model.generate(
            #     input_ids,
            #     images=image_tensor.unsqueeze(0).half().cuda(),
            #     image_sizes=[item['image_sizes']],
            #     do_sample=True if args.temperature > 0 else False,
            #     temperature=args.temperature,
            #     top_p=args.top_p,
            #     num_beams=args.num_beams,
            #     max_new_tokens=1024,
            #     use_cache=True,
            #     output_hidden_states=True,  # 启用隐藏状态输出
            #     return_dict_in_generate=True  # 确保返回字典格式的输出
            # )
            if "qwen" in model_path.lower():
                output = model.forward(
                    **inputs,
                    output_hidden_states=True,  # 启用隐藏状态输出
                    use_cache=False,
                    return_dict=True
                )

            else:
                output = model.forward(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[item['image_sizes']],
                    output_hidden_states=True,  # 启用隐藏状态输出
                    use_cache=False,
                    return_dict=True
                )
            # print("input_ids", inputs)
            # 获取 hidden states
            hidden_states = output.hidden_states  # hidden_states 是一个元组，包含每一层的隐藏状态;单个的shape为torch.Size([1, 625, 4096])
            # print(hidden_states)
            # print("hidden_states.shape", len(hidden_states), hidden_states[0].shape, len(inputs['input_ids'][0]))
            # for qwen [37,1,419,2048]
            if "qwen" in model_path.lower():
                image_tokens_range = [(0, len(inputs['input_ids'][0]) - len(instruction_token) - 1 - 5)]
            else:
                image_tokens_range = output.image_tokens_range
            # print(image_tokens_range)
            # print("================================================")
            # if i==2:
            #     exit()
            # [33,1,625,4096] layer,bz,seq_len,dim
                        
        for j, index in enumerate(info_save_index):
            info_save_list[j].append(hidden_states[index][0][-1].cpu())
            image_info_save_list[j].append(hidden_states[index][0][image_tokens_range[-1][-1]].cpu())
        torch.cuda.empty_cache()
        # if i==10:
        #     break

    if '_full' in annotations_file:
        # print("called1")
        # print("info_save_list", info_save_list)
        # print("label", label_list)
        torch.save(info_save_list, ce_data_dir+'ce_training_full_v1.pth')
        torch.save(image_info_save_list, ce_data_dir+'ce_training_image_full_v1.pth')
        torch.save(label_list, ce_data_dir+'ce_training_label_full_v1.pth')
    else:
        # print("called2")
        torch.save(info_save_list, ce_data_dir+'ce_training_v1.pth')
        torch.save(image_info_save_list, ce_data_dir+'ce_training_image_v1.pth')
        torch.save(label_list, ce_data_dir+'ce_training_label_v1.pth')
    print("save to", ce_data_dir)
    print(mscoco_dataset.missing_index)
    
    # torch.save(output_list, ce_data_dir+'ce_training_response.pth')  # 新增保存模型输出的代码

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    
    parser.add_argument("--lora_ckpt", type=str, default='')
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default='')
    parser.add_argument("--pretrain_vison", type=str, default='')
    
    
    args = parser.parse_args()

    eval_model(args)