ce_data_dir="./pope/"

import torch

def tensors_equal(tensor1, tensor2):
    # 检查两个张量是否相等
    if isinstance(tensor1, list) and isinstance(tensor2, list):
        if len(tensor1) != len(tensor2):
            return False
        for sub_tensor1, sub_tensor2 in zip(tensor1, tensor2):
            if not tensors_equal(sub_tensor1, sub_tensor2):
                return False
        return True
    else:
        return torch.equal(tensor1, tensor2)

# 加载数据
input_datas = torch.load(ce_data_dir + 'pope_info_probe_listvisual_prompt_v1.pth')
input_datas_1 = torch.load(ce_data_dir + 'pope_info_probe_list_v1.pth')

# 比较两个列表
are_equal = tensors_equal(input_datas, input_datas_1)

if are_equal:
    print("两个列表完全相等。")
else:
    print("两个列表不完全相等。")