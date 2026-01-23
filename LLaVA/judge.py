# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset, random_split


# input_datas = torch.load('cifar100_output_list.pth')
# labels = torch.load('label_list_cifar100.pth')

# print(input_datas[:10])
# print(labels[:10])


# import pickle

# def load_cifar10_meta(file_path):
#     with open(file_path, 'rb') as f:
#         meta = pickle.load(f, encoding='bytes')
#     classes = meta[b'label_names']
#     return [class_name.decode('utf-8') for class_name in classes]

# cifar10_meta_path = './data/cifar-10-batches-py/batches.meta'
# cifar10_classes = load_cifar10_meta(cifar10_meta_path)

# def load_cifar100_meta(file_path):
#     with open(file_path, 'rb') as f:
#         meta = pickle.load(f, encoding='bytes')
#     fine_classes = meta[b'fine_label_names']
#     coarse_classes = meta[b'coarse_label_names']
#     return [fine.decode('utf-8') for fine in fine_classes], [coarse.decode('utf-8') for coarse in coarse_classes]

# cifar100_meta_path = './data/cifar-100-python/meta'
# cifar100_classes, cifar100_coarse_classes = load_cifar100_meta(cifar100_meta_path)

import torch
import pickle
import random
import numpy as np

# 假设 input_datas 是句子，labels 是数字标签
input_datas = torch.load('cifar100_output_list.pth')
labels = torch.load('label_list_cifar100.pth')

# 确保 input_datas 和 labels 是同样大小的列表
assert len(input_datas) == len(labels), "Input data and labels must have the same length."

# 加载 CIFAR-100 类别名称
def load_cifar100_meta(file_path):
    with open(file_path, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
    fine_classes = meta[b'fine_label_names']
    return [fine.decode('utf-8') for fine in fine_classes]

cifar100_meta_path = './data/cifar-100-python/meta'
cifar100_classes = load_cifar100_meta(cifar100_meta_path)

# 初始化正确和错误的索引列表
correct_indices = []
incorrect_indices = []

# 遍历每个句子和对应的标签
for i, (sentence, label) in enumerate(zip(input_datas, labels)):
    # 将数字标签转换为字符串标签
    label_str = cifar100_classes[label]
    
    # 检查句子是否包含标签
    if label_str.lower() in sentence.lower():
        correct_indices.append(i)  # 句子正确描述标签
    else:
        incorrect_indices.append(i)  # 句子未正确描述标签

# 输出结果
# print("Correct indices:", correct_indices)
# print("Incorrect indices:", incorrect_indices)

print(len(correct_indices), len(incorrect_indices))
sampled_incorrect_indices = random.sample(incorrect_indices, 16000)
remaining_indices = correct_indices + list(set(incorrect_indices) - set(sampled_incorrect_indices))
sampled_remaining_indices = random.sample(remaining_indices, 4000)

print(len(sampled_incorrect_indices), len(sampled_remaining_indices))

# sampled_incorrect_indices=np.array(sampled_incorrect_indices)
# sampled_remaining_indices=np.array(sampled_remaining_indices)

labels = torch.load('label_list_cifar100.pth')
# # 如果 labels 是一个列表，则转换为张量
# if isinstance(labels, list):
#     labels = torch.tensor(labels)
    
# labels=labels.cpu().numpy()

input_datas = torch.load('info_probe_list_cifar100.pth')
# # 如果 labels 是一个列表，则转换为张量
# if isinstance(input_datas, list):
#     input_datas = np.array(input_datas)
#     input_datas = torch.from_numpy(input_datas)
# input_datas=input_datas.cpu().numpy()
# # print(input_datas[2,4,6])

info_save_index=[0, 6, 12, 18, 24, 30]
training_input=[]
for index in range(len(info_save_index)):
    training_input.append([input_datas[index][i] for i in sampled_incorrect_indices])

training_label=[labels[i] for i in sampled_incorrect_indices]

test_input=[]
for index in range(len(info_save_index)):
    test_input.append([input_datas[index][i] for i in sampled_remaining_indices])

test_label=[labels[i] for i in sampled_remaining_indices]

# 保存到新的 .pth 文件
torch.save(training_input, "info_probe_list_cifar100_incorrect.pth")
torch.save(training_label, "label_list_cifar100_incorrect.pth")
torch.save(test_input, "info_probe_list_cifar100_incorrect_test.pth")
torch.save(test_label, "label_list_cifar100_incorrect_test.pth")
