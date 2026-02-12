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

import random

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

ce_data_dir="../cbm/hal/"

# QUESTION_ONLY=False
# FIXED_QUESTION=True
QUESTION_ONLY=False
FIXED_QUESTION=False

class MSCOCODataset(Dataset):
    def __init__(self, annotations_file, image_dir, image_processor, model, visual_prompt=False, correct_refer=1, model_path=None):
        self.image_dir = image_dir
        self.image_processor = image_processor
        self.model = model
        self.model_path = model_path
        self.missing_index=[]
        
        if visual_prompt:
            json_file = '/deepfreeze/zihan/deepfreeze/junda/datasets/COCO2014/yolo_output_hal/detection_results.json'
            with open(json_file, 'r') as file:
                self.visual_annotation=json.load(file)

        # 读取 annotation 文件
        self.dataset = torch.load(annotations_file)
        
        self.visual_prompt=visual_prompt
        self.correct_refer=correct_refer
        self.missing_digit_list=[]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # item = self.dataset.items()[idx]
        item = self.dataset[idx]
        # hal_type = item['hal_type']
        
        if self.visual_prompt:
            #############################################################
            refer_digit_list=[]
            not_refer_digit_list=[]

            if len(self.visual_annotation[f"annotated_{item['image']}"].items())<=1:
                self.missing_digit_list.append(idx)
                
            for label_id, obj_name in self.visual_annotation[f"annotated_{item['image']}"].items():
                refer_digit_list.append((label_id,obj_name))

            label_ids = list(self.visual_annotation[f"annotated_{item['image']}"].keys())
            obj_names = list(self.visual_annotation[f"annotated_{item['image']}"].values())
            random.shuffle(obj_names)
            for i in range(len(label_ids)):
                not_refer_digit_list.append((label_ids[i], obj_names[i]))
            if self.correct_refer==1:
                question=item['question']+" on the image? "+f" Plase refer to objects near annotated digis in the following list {refer_digit_list} on the image.\n"
            elif self.correct_refer==0:
                question=item['question']+" on the image? "+f" Plase refer to objects near annotated digis on the image as is in the following list {not_refer_digit_list}.\n"
            else:
                question=item['question']+" on the image?\n"
            # image = Image.open(visual_image_dir+f"annotated_image_{idx}.jpg").convert('RGB')
            image_path = os.path.join(self.image_dir, f"annotated_{item['image']}")
        else:
            question = item['question']  # 固定问题
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
        # image_tensor = []
        return {
            'image': image_tensor,
            'question': question,
            'label': label,
            # 'hal_type': hal_type,
            'image_sizes': image_sizes,
            'image_source': item['image']
        }
        
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
    
    if "visual" in args.question:
        image_dir = '/deepfreeze/zihan/deepfreeze/junda/datasets/COCO2014/yolo_output_hal'
    else:    
        image_dir = '/deepfreeze/zihan/deepfreeze/junda/datasets/COCO2014/val2014'
    
    if "visual_prompt-1" in args.question:
        correct_refer=1 # correct refer
    elif "visual_prompt-2" in args.question:
        correct_refer=0 # wrong refer
    else:
        correct_refer=2 # no refer
    
    if "correct" in args.question:
        annotations_file = f'../cbm/hal/correct_hal_ce_data_full_v1.pth'  
    elif "wrong" in args.question:
        annotations_file = f'../cbm/hal/wrong_hal_ce_data_full_v1.pth'  

    visual="visual" in args.question
    mscoco_dataset = MSCOCODataset(annotations_file, image_dir, image_processor, model, visual_prompt=visual, correct_refer=correct_refer, model_path=model_path)

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
    avg_info_save_list=[[] for _ in range(len(info_save_index))]
    image_info_save_list=[[] for _ in range(len(info_save_index))]
    label_list=[]
    output_list = []
    question_list = []
    vit_feature_list=[]
    for i, item in enumerate(train_loader):
        label = [float(x[0]) for x in item['label']]  # Assuming 'label' key exists
        # item['image_sizes']=(item['image_sizes'][0][0], item['image_sizes'][1][0])

        print(i, "/", len(train_loader), end="\r")
        
        if "origin" in args.question or "visual" in args.question:
            qs = "Is there " + " " + item['question'][0] + "?\n"
            # qs = item['question'][0] + "\n"
        elif not FIXED_QUESTION:
            qs = " Describe the scene in the image, focusing on key elements such as people, objects, actions etc. \n" #"Please directly answer what is the number digit on the image by a digit."
        else:
            qs = " List all items on this image. \n"
        
        label_list.append(label)
        
        if "qwen" in model_path.lower():
            image = Image.open(item['image'][0]).convert('RGB')
            
            if not QUESTION_ONLY and not FIXED_QUESTION and "origin" not in args.question and "visual" not in args.question :
                prompt = qs + " " + item['question'][0]
            else: 
                prompt = qs
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"{prompt}"}
                    ]
                }
            ]
            
            instruction_token = tokenizer(prompt)['input_ids']
            
            text = image_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, _ = process_vision_info(messages)
            
            inputs = image_processor(
                text=[text],
                images=image_inputs,
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
            
            if not QUESTION_ONLY and not FIXED_QUESTION and "origin" not in args.question and "visual" not in args.question :
                prompt += " " + item['question'][0]

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image_tensor=item['image']
        
        question_list.append(prompt)
        
        with torch.inference_mode():
            
            if "qwen" in model_path.lower():
                output = model.forward(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True
                )
                image_tokens_range = [(0, len(inputs['input_ids'][0]) - len(instruction_token) - 1 - 5)]
            else:
                output = model.forward(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[item['image_sizes']],
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True
                )
                image_tokens_range=output.image_tokens_range

            hidden_states = output.hidden_states
            # [33,1,625,4096] layer,bz,seq_len,dim
                        
        for j, index in enumerate(info_save_index):
            info_save_list[j].append(hidden_states[index][0][-1].cpu())
            avg_info_save_list[j].append(hidden_states[index][0][image_tokens_range[-1][-1]+1:].cpu().mean(dim=0))
            image_info_save_list[j].append(hidden_states[index][0][image_tokens_range[-1][-1]].cpu())

    prefix="" if not QUESTION_ONLY else "question_only_"
    prefix="" if not FIXED_QUESTION else "fixed_question_"

    if '_full' in annotations_file:
        
        torch.save(info_save_list, ce_data_dir+f'{prefix}{args.question}_hal_ce_val_full_v1.pth')
        torch.save(avg_info_save_list, ce_data_dir+f'{prefix}avg_{args.question}_info_save_list_hal_ce_val_full_v1.pth')

        torch.save(image_info_save_list, ce_data_dir+f'{prefix}{args.question}_hal_ce_val_image_full_v1.pth')
        torch.save(label_list, ce_data_dir+f'{prefix}{args.question}_hal_ce_val_label_full_v1.pth')
        torch.save(question_list, ce_data_dir+f'{prefix}{args.question}_hal_ce_val_question_full_v1.pth')
    else:
        torch.save(info_save_list, ce_data_dir+f'{prefix}{args.question}_hal_ce_val_v1.pth')
        torch.save(avg_info_save_list, ce_data_dir+f'{prefix}avg_{args.question}_hal_ce_val_v1.pth')

        
        torch.save(image_info_save_list, ce_data_dir+f'{prefix}{args.question}_hal_ce_val_image_v1.pth')
        torch.save(label_list, ce_data_dir+f'{prefix}{args.question}_hal_ce_val_label_v1.pth')
        torch.save(question_list, ce_data_dir+f'{prefix}{args.question}_hal_ce_val_question_v1.pth')
        # pass
    
    print("save to", ce_data_dir)
    
    if "visual" not in args.question:    
        print(mscoco_dataset.missing_index)
        # torch.save(mscoco_dataset.missing_index, ce_data_dir+f'{args.question}_hal_missing_index_v1.pth')
    else:
        print(mscoco_dataset.missing_digit_list)
    
    
    
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