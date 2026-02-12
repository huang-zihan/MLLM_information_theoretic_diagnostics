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

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from torchvision import transforms

QUESTION_ONLY=False
FIXED_QUESTION=False

class AOKVQADataset(Dataset):
    def __init__(self, image_processor, model, cbm_path='../cbm/', visual_prompt=False, atype="", dataset=None, model_path=None):
        self.image_processor = image_processor
        self.model=model
        self.model_path = model_path
        self.dataset = dataset
        if visual_prompt:
            self.annotation=torch.load(cbm_path+'result/aokvqa_feature/vp_aokvqa_ce_test_data_yolo11l.pt.pth')
        else:
            self.annotation=torch.load(cbm_path+'result/aokvqa_feature/aokvqa_ce_test_data_yolo11l.pt.pth')
        self.missing_index=[]
        self.atype=atype
        
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # item = self.annotation[idx]
        
        question=item['question']+"\n"
        
        image = item['image'].convert('RGB')
        
        label = self.annotation[idx]['annotation']
        image_sizes=image.size

        # 将PIL图像转换为张量
        if "qwen" in self.model_path.lower():        
            image_tensor = image_tensor = self.to_tensor(item['image'])
        else:            
            image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        # image_tensor = []     
        
        if label==[]:
            
            if "correct" in self.atype:
                self.missing_index.append(idx)
            elif "wrong" in self.atype:
                for index in range(3*idx, 3*idx+3):
                    self.missing_index.append(index)
        
        assert len(item['choices'])==4
        
        correct_index=item['correct_choice_idx']
        if "correct" in self.atype:
            response=[item['choices'][correct_index]]
        elif "wrong" in self.atype:
            response=[item['choices'][j] for j in range(len(item['choices'])) if j!=correct_index]    
        return {'image': image_tensor, 'question':question, 'label': label, 'image_sizes':image_sizes, 'response': response}


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
    #         # prefix="1111111111111"
    #         for key, value in non_lora_state_dict.items():
    #             new_key = key.replace(prefix, '')
    #             new_state_dict[new_key] = value
    #             # if "mm_projector" in key:
    #             #     print("!!!", key, new_key)
    #             print(key)
    #         model.get_model().mm_projector.load_state_dict(new_state_dict) # , strict=False
    #     else:
    #         # prefix="model.mm_projector."
            
    #         prefix="base_model.model.model.mm_projector."
            
    #         for key, value in non_lora_state_dict.items():
    #             new_key = key.replace(prefix, '')
    #             new_state_dict[new_key] = value
    #         model.get_model().mm_projector.load_state_dict(new_state_dict)
        
    #     print(f"mmp {args.pretrain_mm_mlp_adapter} loaded")
    

    import torch
    from torchvision import datasets, transforms

    from datasets import load_dataset
    
    if "visual_prompt" in args.question:
        visual_prompt=True
    else:
        visual_prompt=False
    
    train_dataset = load_dataset("HuggingFaceM4/A-OKVQA", split="validation", cache_dir='/deepfreeze/zihan/datasets/')
    aokvqa_dataset = AOKVQADataset(image_processor, model, visual_prompt=visual_prompt, atype=args.question, dataset=train_dataset, model_path=model_path)

    train_loader = torch.utils.data.DataLoader(
        dataset=aokvqa_dataset,
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
        label = item['label']  # Assuming 'label' key exists
        
        item['image_sizes']=(item['image_sizes'][0][0], item['image_sizes'][1][0])
        print(i, "/", len(train_loader), end="\r")
        image_tensor=item['image']
        
        for response in item['response']:
            
            if "origin" in args.question:
                qs = "Based on this image. " + item['question'][0] + " " + response[0] + "\n"
            elif not FIXED_QUESTION:
                qs = item['question'][0]
            else:
                qs = " List all items on this image. \n"
                    
            label_list.append(label)
            
            if "qwen" in model_path.lower():
                
                toplt = transforms.ToPILImage() #Image.open(item['image'][0]).convert('RGB')
                image = toplt(item['image'][0])
                
                if not QUESTION_ONLY and not FIXED_QUESTION and not "origin" in args.question:
                    prompt = qs + " "+response[0]+" "
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
                
                if not QUESTION_ONLY and not FIXED_QUESTION and not "origin" in args.question:
                    prompt += " "+response[0]+" "
                
                # continue
                
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            question_list.append(prompt)

            with torch.inference_mode():
                
                if "qwen" not in model_path.lower():
                    vit_features = model.get_model().vision_tower(image_tensor.half().cuda())
                    vit_feature_list.append(torch.mean(vit_features[0], dim=0))

                    output = model.forward(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        image_sizes=[item['image_sizes']],
                        output_hidden_states=True,  # 启用隐藏状态输出
                        use_cache=False,
                        return_dict=True
                    )
                    image_tokens_range=output.image_tokens_range
                else:
                    output = model.forward(
                        **inputs,
                        output_hidden_states=True,  # 启用隐藏状态输出
                        use_cache=False,
                        return_dict=True
                    )
                    
                    if "qwen" in model_path.lower():
                        image_tokens_range = [(0, len(inputs['input_ids'][0]) - len(instruction_token) - 1 - 5)]
                    else:
                        image_tokens_range = output.image_tokens_range
                    
                # 获取 hidden states
                hidden_states = output.hidden_states  # hidden_states 是一个元组，包含每一层的隐藏状态;单个的shape为torch.Size([1, 625, 4096])
                
                # [33,1,625,4096] layer,bz,seq_len,dim
            
            # print(instruction_token)
            # print("input_ids", inputs)
            # print("hidden_states.shape", len(hidden_states), hidden_states[0].shape, len(inputs['input_ids'][0]))
            # print()
            # print("================================================")
            # if i==2:
            #     exit()
            
            for j, index in enumerate(info_save_index):
                info_save_list[j].append(hidden_states[index][0][-1].cpu())
                avg_info_save_list[j].append(hidden_states[index][0][image_tokens_range[-1][-1]+1:].cpu().mean(dim=0))
                image_info_save_list[j].append(hidden_states[index][0][image_tokens_range[-1][-1]].cpu())
            torch.cuda.empty_cache()
        
        # if i==10:
        #     break
        

    cbm_path="../cbm/aokvqa/"
    
    prefix="" if not QUESTION_ONLY else "question_only_"
    prefix="" if not FIXED_QUESTION else "fixed_question_"
    
    torch.save(info_save_list, cbm_path+f'{prefix}{args.question}_aokvqa_info_probe_list_v1.pth')
    torch.save(avg_info_save_list, cbm_path+f'{prefix}avg_{args.question}_aokvqa_info_probe_list_v1.pth')
    
    torch.save(image_info_save_list, cbm_path+f'{prefix}{args.question}_image_aokvqa_info_probe_list_v1.pth')
    torch.save(label_list, cbm_path+f'{prefix}{args.question}_aokvqa_label_list_v1.pth')
    torch.save(question_list, cbm_path+f'{prefix}{args.question}_aokvqa_question_list_v1.pth')

    print("save to", cbm_path)
    print(aokvqa_dataset.missing_index)
    
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