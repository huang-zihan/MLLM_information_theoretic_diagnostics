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
    
    
    # # 获取LORA适配器的参数(打印可训练lora参数使用)
    # lora_params = [param for name, param in model.named_parameters() if 'lora' in name]
    # print(lora_params)
    # # 计算LORA适配器的总参数量
    # total_lora_params = sum(param.numel() for param in lora_params)
    # print(f'Total LORA adapter parameters: {total_lora_params}')
    # exit()

    import torch
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    # 定义图像预处理
    # transform = transforms.Compose([
    #     transforms.ToTensor(),  # 将图像转换为Tensor
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 标准化
    # ])

    # 加载CIFAR100训练数据集
    train_dataset_cifar100 = datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        # transform=transform,
    )

    # 加载CIFAR100测试数据集
    test_dataset_cifar100 = datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        # transform=transform,
    )

    # 创建CIFAR100数据加载器
    train_loader_cifar100 = torch.utils.data.DataLoader(
        dataset=train_dataset_cifar100,
        batch_size=64,
        shuffle=True,
    )

    test_loader_cifar100 = torch.utils.data.DataLoader(
        dataset=test_dataset_cifar100,
        batch_size=64,
        shuffle=False,
    )

    # 信息保存索引和列表
    info_save_index = [0, 6, 12, 18, 24, 30]
    info_save_list = [[] for _ in range(len(info_save_index))]
    label_list = []

    output_list = []

    # 对CIFAR100数据集进行处理
    for i, (image, label) in enumerate(train_dataset_cifar100):
        print(i, "/", len(train_dataset_cifar100), end="\r")
        
        qs = "Please directly answer what is the object on the image by the object."

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image_tensor = process_images([image], image_processor, model.config)[0]  # [image]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        for j, index in enumerate(info_save_index):
            info_save_list[j].append(model.info_probe_list[index].squeeze())
        label_list.append(label)
        output_list.append(outputs)

        # if i == 100:  # 只处理前100个样本
        #     break

    # 保存信息探测列表和标签列表
    torch.save(info_save_list, 'info_probe_list_cifar100.pth')
    torch.save(label_list, 'label_list_cifar100.pth')
    torch.save(output_list, 'cifar100_output_list.pth')  # 新增保存模型输出的代码

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
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