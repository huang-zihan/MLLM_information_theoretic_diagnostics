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
 
class CHAIRDataset(Dataset):
    def __init__(self, image_processor, model, cbm_path='/home/junda/zihan/cbm/', visual_prompt=False):
        self.image_processor = image_processor
        self.model=model
        if visual_prompt:
            self.annotation=torch.load(cbm_path+'result/chair_feature/vp_chair_ce_test_data_yolo11l.pt.pth')
            # with open(json_file, 'r') as file:
            #     self.visual_annotation=json.load(file)
            
        else:
            self.annotation=torch.load(cbm_path+'result/chair_feature/chair_ce_test_data_yolo11l.pt.pth')
        self.missing_index=[]

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        item = self.annotation[idx]
        question=item['question']
        image = item['image'].convert('RGB')
        
        label = item['annotation']  # 确保这里的'key'与你的数据集一致
        image_sizes=image.size

        # 将PIL图像转换为张量
        # image_tensor=None
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        # return {'label': label}
        # image_tensor=[]
        if label==[]:
            self.missing_index.append(idx)
        
        return {'image': image_tensor, 'question':question, 'label': label, 'image_sizes':image_sizes}


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

    
    if args.lora_ckpt != "":
        model.load_adapter(args.lora_ckpt)
        print(f"adapter {args.lora_ckpt} loaded")
    
    if args.pretrain_vison != "":
        non_lora_state_dict = torch.load(args.pretrain_vison)
        new_state_dict = {}
        
        if args.pretrain_mm_mlp_adapter and args.lora_ckpt:
            prefix="base_model.model.model.vision_tower."
            for key, value in non_lora_state_dict.items():
                new_key = key.replace(prefix, '')
                new_state_dict[new_key] = value
            model.get_model().vision_tower.load_state_dict(new_state_dict) # , strict=False
        else:
            prefix="model.vision_tower."
            for key, value in non_lora_state_dict.items():
                # print(key)
                new_key = key.replace(prefix, '')
                new_state_dict[new_key] = value
            model.get_model().vision_tower.load_state_dict(new_state_dict)
            
        print(f"vision {args.pretrain_vison} loaded")
        
    
    if args.pretrain_mm_mlp_adapter != "":
        
        non_lora_state_dict = torch.load(args.pretrain_mm_mlp_adapter)
        new_state_dict = {}
        
        if args.lora_ckpt:
            prefix="base_model.model.model.mm_projector."
            # prefix="1111111111111"
            for key, value in non_lora_state_dict.items():
                new_key = key.replace(prefix, '')
                new_state_dict[new_key] = value
                # if "mm_projector" in key:
                #     print("!!!", key, new_key)
                print(key)
            model.get_model().mm_projector.load_state_dict(new_state_dict) # , strict=False
        else:
            # prefix="model.mm_projector."
            
            prefix="base_model.model.model.mm_projector."
            
            for key, value in non_lora_state_dict.items():
                new_key = key.replace(prefix, '')
                new_state_dict[new_key] = value
            model.get_model().mm_projector.load_state_dict(new_state_dict)
        
        print(f"mmp {args.pretrain_mm_mlp_adapter} loaded")
    

    import torch
    from torchvision import datasets, transforms

    from datasets import load_dataset
    
    if "visual_prompt" in args.question:
        visual_prompt=True
    else:
        visual_prompt=False
        
    chair_dataset = CHAIRDataset(image_processor, model, visual_prompt=visual_prompt)

    # Create data loader for the CHAIR dataset
    train_loader = torch.utils.data.DataLoader(
        dataset=chair_dataset,
        batch_size=1,
        shuffle=False
    )

    info_save_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
    info_save_list=[[] for _ in range(len(info_save_index))]
    image_info_save_list=[[] for _ in range(len(info_save_index))]
    label_list=[]
    output_list = []
    question_list = []
    vit_feature_list=[]
    for i, item in enumerate(train_loader):
        
        # label = item['label']  # Assuming 'label' key exists
        # label_list.append(label)
        # item['image_sizes']=(item['image_sizes'][0][0], item['image_sizes'][1][0])

        print(i, "/", len(train_loader), end="\r")
        
        # qs=item['question'][0]
        # question_list.append(qs)
        
        # if model.config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' +  qs
        
        # conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()

        # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # image_tensor=item['image']

        # with torch.inference_mode():
            
        #     vit_features = model.get_model().vision_tower(image_tensor.half().cuda())
        #     vit_feature_list.append(torch.mean(vit_features[0], dim=0))
            
        #     # output = model.generate(
        #     #     input_ids,
        #     #     images=image_tensor.unsqueeze(0).half().cuda(),
        #     #     image_sizes=[item['image_sizes']],
        #     #     do_sample=True if args.temperature > 0 else False,
        #     #     temperature=args.temperature,
        #     #     top_p=args.top_p,
        #     #     num_beams=args.num_beams,
        #     #     max_new_tokens=1024,
        #     #     use_cache=True,
        #     #     output_hidden_states=True,  # 启用隐藏状态输出
        #     #     return_dict_in_generate=True  # 确保返回字典格式的输出
        #     # )
        #     output = model.forward(
        #         input_ids,
        #         images=image_tensor.unsqueeze(0).half().cuda(),
        #         image_sizes=[item['image_sizes']],
        #         output_hidden_states=True,  # 启用隐藏状态输出
        #         use_cache=False,
        #         return_dict=True
        #     )

        #     # 获取 hidden states
        #     hidden_states = output.hidden_states  # hidden_states 是一个元组，包含每一层的隐藏状态;单个的shape为torch.Size([1, 625, 4096])
        #     image_tokens_range=output.image_tokens_range
        #     # [33,1,625,4096] layer,bz,seq_len,dim
            
        # for j, index in enumerate(info_save_index):
        #     info_save_list[j].append(hidden_states[index][0][-1].cpu())
        #     image_info_save_list[j].append(hidden_states[index][0][image_tokens_range[-1][-1]].cpu())
        # torch.cuda.empty_cache()
        
        # ### for calling generate
        # # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # # print(model.info_probe_list[0])
        # # print(model.image_info_probe_list[0])
        # # for j, index in enumerate(info_save_index):
        # #     info_save_list[j].append(hidden_states[-1][index].squeeze())
        # #     image_info_save_list[j].append(model.image_info_probe_list[index].squeeze()) 

        # # for j, index in enumerate(info_save_index):
        # #     info_save_list[j].append(hidden_states[index][0][-1].squeeze().cpu())
        # #     image_info_save_list[j].append(hidden_states[index][0][image_tokens_range[-1][-1]].squeeze())
        # # torch.cuda.empty_cache()
        # # if i==10:
        # #     break
        
    
    cbm_path="../cbm/chair/"
    print(chair_dataset.missing_index)
    # if args.question=="":
    #     torch.save(info_save_list, cbm_path+'chair_info_probe_list_v1.pth')
    #     torch.save(image_info_save_list, cbm_path+'image_chair_info_probe_list_v1.pth')
    #     torch.save(label_list, cbm_path+'chair_label_list_v1.pth')
    #     # torch.save(output_list, cbm_path+'pope_output_list.pth')  # 新增保存模型输出的代码
    #     # torch.save(question_list, cbm_path+'pope_question_list.pth')  # 新增保存模型输出的代码
    #     # torch.save(vit_feature_list, cbm_path+'pope_vit_feature_list.pth')  # 新增保存模型输出的代码
    # # elif args.question=="visual_prompt":
    # #     torch.save(info_save_list, cbm_path+'vp_pope_info_probe_list_v1.pth')
    # #     torch.save(image_info_save_list, cbm_path+'vp_image_pope_info_probe_list_v1.pth')
    # #     torch.save(label_list, cbm_path+'vp_pope_label_list_v1.pth')
    # else:
    #     torch.save(info_save_list, cbm_path+f'chair_info_probe_list{args.question}_v1.pth')
    #     torch.save(image_info_save_list, cbm_path+f'image_chair_info_probe_list{args.question}_v1.pth')
        
    #     torch.save(label_list, cbm_path+'chair_label_list_v1.pth')
    #     # print(question_list)
    #     # print(label_list[0])
    #     # print(label_list[0]['annotation'], len(label_list[0]['annotation']))
    #     # torch.save(output_list, cbm_path+f'pope_output_list{args.question}.pth')  # 新增保存模型输出的代码
    #     # torch.save(question_list, cbm_path+f'pope_question_list{args.question}.pth')  # 新增保存模型输出的代码
    #     # torch.save(vit_feature_list, cbm_path+f'pope_vit_feature_list{args.question}.pth')  # 新增保存模型输出的代码

    
if __name__ == "__main__":
    
    # label_list = torch.load("../cbm/pope/"+'pope_label_list_v1.pth')
    # print(label_list[0], len(label_list[0]))
    # exit()
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