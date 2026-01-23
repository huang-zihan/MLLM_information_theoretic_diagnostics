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


class POPEDataset(Dataset):
    def __init__(self, dataset, image_processor, model):
        self.dataset = dataset
        self.image_processor = image_processor
        self.model=model

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question=item['question']
        image = item['image'].convert('RGB')
        label = 1 if item['answer']=='yes' else 0  # 确保这里的'key'与你的数据集一致

        image_sizes=image.size

        # 将PIL图像转换为张量
        image_tensor = process_images([image], self.image_processor, self.model.config)[0]
        return {'image': image_tensor, 'question':question, 'label': label, 'image_sizes':image_sizes, 'image_source':item['image_source']}

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
    
    # print(model)
    # exit()
    # # 获取LORA适配器的参数(打印可训练lora参数使用)
    # lora_params = [param for name, param in model.named_parameters() if 'lora' in name]
    # print(lora_params)
    # # 计算LORA适配器的总参数量
    # total_lora_params = sum(param.numel() for param in lora_params)
    # print(f'Total LORA adapter parameters: {total_lora_params}')
    # exit()


    item_dict = {'snowboard': 0, 'backpack': 1, 'person': 2, 'car': 3, 'skis': 4, 'dog': 5, 'truck': 6, 'dining table': 7, 'handbag': 8, 'bicycle': 9, 'motorcycle': 10, 'potted plant': 11, 'vase': 12, 'traffic light': 13, 'bus': 14, 'chair': 15, 'bed': 16, 'book': 17, 'spoon': 18, 'cup': 19, 'fork': 20, 'tv': 21, 'toaster': 22, 'microwave': 23, 'bottle': 24, 'bird': 25, 'boat': 26, 'couch': 27, 'sandwich': 28, 'bowl': 29, 'hot dog': 30, 'frisbee': 31, 'knife': 32, 'cake': 33, 'remote': 34, 'baseball glove': 35, 'sports ball': 36, 'baseball bat': 37, 'bench': 38, 'sink': 39, 'toilet': 40, 'teddy bear': 41, 'bear': 42, 'cat': 43, 'mouse': 44, 'laptop': 45, 'toothbrush': 46, 'cow': 47, 'skateboard': 48, 'surfboard': 49, 'cell phone': 50, 'train': 51, 'clock': 52, 'tennis racket': 53, 'suitcase': 54, 'horse': 55, 'banana': 56, 'wine glass': 57, 'refrigerator': 58, 'carrot': 59, 'broccoli': 60, 'tie': 61, 'scissors': 62, 'sheep': 63, 'airplane': 64, 'stop sign': 65, 'fire hydrant': 66, 'keyboard': 67, 'pizza': 68, 'donut': 69, 'kite': 70, 'parking meter': 71, 'giraffe': 72, 'zebra': 73, 'umbrella': 74, 'orange': 75, 'oven': 76, 'elephant': 77, 'apple': 78}
    
    import torch
    from torchvision import datasets, transforms

    from datasets import load_dataset
    
    
    pope_meta=torch.load('pope_meta_data.pth')
    item_to_image=pope_meta['item_to_image']
    image_to_item=pope_meta['image_to_item']
    # # print(item_to_image)
    # # zebra apple
    
    # # 计算集合长度并排序
    # sorted_items = sorted(item_to_image.items(), key=lambda x: len(x[1]), reverse=True)
    # # 获取前N个键及其对应的集合
    # longest_items = sorted_items[:10]
    # 转换为字典格式，方便查看
    # longest_items_dict = dict(longest_items)
    # for key, val in longest_items_dict.items():
    #     print(key, len(val))
    
    # shortest_items = sorted_items[-10:]
    # # 转换为字典格式，方便查看
    # shortest_items_dict = dict(shortest_items)
    # for key, val in shortest_items_dict.items():
    #     print(key, len(val))
    # print(longest_items_dict)
    # car 189
    # dining table 183
    # zebra 3
    # apple 4
    delect_list=item_to_image[args.question]
    indices_to_remove = []
    for index in delect_list:
        indices_to_remove.extend(range(index, index + 6))
    # print(delect_list)
    # print(indices_to_remove)
    # exit()
    
    train_dataset = load_dataset("lmms-lab/POPE", split="test") #"default"
    train_dataset = [item for idx, item in enumerate(train_dataset) if idx not in indices_to_remove]
    # print(len(train_dataset))
    pope_dataset = POPEDataset(train_dataset, image_processor, model)
    # print(len(pope_dataset))
    # exit()
    
    

    # Create data loader for the POPE dataset
    train_loader = torch.utils.data.DataLoader(
        dataset=pope_dataset,
        batch_size=1,
        shuffle=False
    )
    
    # info_save_index=[0, 6, 12, 18, 24, 30]
    # info_save_index=[0, 12, 24, 27, 30, 31]
    info_save_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    info_save_list=[[] for _ in range(len(info_save_index))]
    label_list=[]
    output_list = []
    question_list = []
    vit_feature_list=[]
    temp=[0]*len(item_dict)
    for i, item in enumerate(train_loader):
        
        label = item['label']  # Assuming 'label' key exists
        # print(item['image_source'])
        item['image_sizes']=(item['image_sizes'][0][0], item['image_sizes'][1][0])

        print(i, "/", len(train_dataset), end="\r")
        
        if args.question=="":
            qs=item['question'][0] #"Please directly answer what is the number digit on the image by a digit."
        else:
            if args.question[0] in ['a', 'e', 'i', 'o', 'u']:
                qs="Is there an " +args.question+ " in the image?"
            else:
                qs="Is there a " +args.question+ " in the image?"
        ###########
        # if label==1:
        #     if "Is there a " in qs:
        #         start = qs.find("Is there a ") + len("Is there a ")
        #     elif "Is there an " in qs:
        #         start = qs.find("Is there an ") + len("Is there an ")
        #     else:
        #         print(qs)
        #         print('not find')
                
        #     end = qs.find(" in the image?")
        #     if start != -1 and end != -1:
        #         object_name = qs[start:end].strip()
        #         temp[item_dict[object_name]]=1
        ###########
        
        question_list.append(qs)
        
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
            
            vit_features = model.get_model().vision_tower(image_tensor.half().cuda())
            vit_feature_list.append(torch.flatten(vit_features))
            
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[item['image_sizes']],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # print(i, qs, outputs)
        
        for j, index in enumerate(info_save_index):
            info_save_list[j].append(model.info_probe_list[index].squeeze())
        output_list.append(outputs)
        
        
        # if (i+1)%6==0:
        #     for _ in range(6):
        #         label_list.append(temp)
        #     print(temp)
        #     temp=[0]*len(item_dict)
        
        # if args.question=="" and i==11:
        #     break
            
    #     # break  # 只查看第一个批次
    if args.question=="":
        torch.save(info_save_list, 'pope_info_probe_list.pth')
        # torch.save(label_list, 'pope_label_list.pth')
        torch.save(output_list, 'pope_output_list.pth')  # 新增保存模型输出的代码
        torch.save(question_list, 'pope_question_list.pth')  # 新增保存模型输出的代码
        # torch.save(vit_feature_list, 'pope_vit_feature_list.pth')  # 新增保存模型输出的代码
    else:
        torch.save(info_save_list, f'pope_info_probe_list{args.question}.pth')
        # torch.save(label_list, 'pope_label_list.pth')
        torch.save(output_list, f'pope_output_list{args.question}.pth')  # 新增保存模型输出的代码
        torch.save(question_list, f'pope_question_list{args.question}.pth')  # 新增保存模型输出的代码
        # torch.save(vit_feature_list, f'pope_vit_feature_list{args.question}.pth')  # 新增保存模型输出的代码

    # print(label_list)
    
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