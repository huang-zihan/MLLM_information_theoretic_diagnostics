import matplotlib.pyplot as plt
import numpy as np
import torch

# 设置字体为 Times New Roman
from matplotlib.font_manager import FontProperties
plt.rcParams['font.family'] = 'serif'
legend_font_comb = FontProperties(fname='/home/junda/zihan/font/TimesNewRoman.ttf', size=24)
legend_font_sep = FontProperties(fname='/home/junda/zihan/font/TimesNewRoman.ttf', size=22)
xylabel_comb = FontProperties(fname='/home/junda/zihan/font/TimesNewRoman.ttf', size=26)
xylabel_font = FontProperties(fname='/home/junda/zihan/font/TimesNewRoman.ttf', size=24)
title_font = FontProperties(fname='/home/junda/zihan/font/TimesNewRoman.ttf', size=24)

def custom_moving_average(data, window_size=2):
    n = len(data)
    ma = np.empty(n)

    for i in range(n):
        left_start = max(0, i - window_size)
        left_end = i + 1
        right_start = i + 1
        right_end = min(n, i + window_size + 1)

        left_values = data[left_start:left_end]
        right_values = data[right_start:right_end]
        all_values = np.concatenate((left_values, right_values))
        ma[i] = np.mean(all_values)

    return ma

datasets = ['VQA', 'COCO_caption', 'AOKVQA', 'HAL']
datasets_title = {
    'VQA': 'VQA',
    'COCO_caption': 'COCO-caption',
    'AOKVQA':'AOKVQA',
    'HAL': 'HAL-Eval'
}

QUESTION_ONLY=False
FIXED_QUESTION=False

origin=True
avg_token=False

if QUESTION_ONLY:
    prefix="question_only_"
elif origin:
    if not avg_token:
        prefix="origin"
    else:
        prefix="originavg_"
else:
    prefix="" if not FIXED_QUESTION else "fixed_question_"

data_results = {}

for dataset in datasets:
    if dataset in ['HAL', 'AOKVQA']:
        for file_type in ['correct_', 'wrong_']:
            file_path = f"{prefix}{file_type}{dataset.lower()}_scores_data.pth"
            print("loaded from:", file_path)
            loaded_data = torch.load(file_path)
            data_results[file_type+dataset] = {
                'scores': loaded_data['scores'],
                'base_scores': loaded_data['base_scores'],
                'result': [score-base for base, score in zip(loaded_data['base_scores'], loaded_data['scores'])]
            }
    else:
        file_path = f'{prefix}{dataset.lower()}_scores_data.pth'
        print("loaded from:", file_path)
        loaded_data = torch.load(file_path)
        data_results[dataset] = {
            'scores': loaded_data['scores'],
            'base_scores': loaded_data['base_scores'],
            'result': [score-base for base, score in zip(loaded_data['base_scores'], loaded_data['scores'])]
        }

for i, dataset in enumerate(data_results.keys()):
    data_results[dataset]['scores']=custom_moving_average(data_results[dataset]['scores'], window_size=2)
    data_results[dataset]['base_scores']=custom_moving_average(data_results[dataset]['base_scores'], window_size=2)

# 创建 2x2 子图
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 绘制每种数据的 base_scores 和 scores
for i, dataset in enumerate(datasets):
    ax = axs[i // 2, i % 2]
    ax.tick_params(axis='both', labelsize=18)
    
    
    if dataset in ['HAL', 'AOKVQA']:
        if QUESTION_ONLY:
            x_values = np.arange(1, len(data_results["wrong_"+dataset]) + 1)  # 从1开始的 X 值
            ax.plot(x_values, data_results["wrong_"+dataset]['scores'], marker='o', color='b', label='Task Scores')
            ax.plot(x_values, data_results["wrong_"+dataset]['base_scores'], marker='x', color='g', label='Image Scores')
        else:
            x_values = np.arange(1, len(data_results["correct_"+dataset]['scores']) + 1)  # 从1开始的 X 值
            ax.plot(x_values, data_results["correct_"+dataset]['scores'], marker='o', color='b', label='Correct Answer Text, Task Scores')
            ax.plot(x_values, data_results["correct_"+dataset]['base_scores'], marker='x', color='g', label='Correct Answer Text, Image Scores')
            ax.plot(x_values, data_results["wrong_"+dataset]['scores'], marker='o', color='r', label='Wrong Answer Text, Task Scores')
            ax.plot(x_values, data_results["wrong_"+dataset]['base_scores'], marker='x', color='orange', label='Wrong Answer Text, Image Scores')

        ax.set_title(datasets_title[dataset], fontproperties=title_font)
        ax.set_xlabel('Layer Index', fontproperties=xylabel_font)
        ax.set_ylabel(r'$\mathcal{F}^{(l)}_T$', fontproperties=xylabel_font)
        ax.grid()
    
    else:
        x_values = np.arange(1, len(data_results[dataset]['scores']) + 1)  # 从1开始的 X 值
        ax.plot(x_values, data_results[dataset]['scores'], marker='o', color='b', label=r'$I(X^{(l)}_{T}; X^{(L)}_{T})$')
        ax.plot(x_values, data_results[dataset]['base_scores'], marker='x', color='g', label=r'$I(X^{(l)}_{0}; X^{(L)}_{0})$')
        ax.set_title(datasets_title[dataset], fontproperties=title_font)
        ax.set_xlabel('Layer Index', fontproperties=xylabel_font)
        ax.set_ylabel(r'$\mathcal{F}^{(l)}_T$', fontproperties=xylabel_font)
        ax.grid()

# 添加共享 legend
handles, labels = axs[1, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', prop=legend_font_sep, ncol=2)

# 调整布局
plt.tight_layout(rect=[0, 0.03, 1, 0.9])
plt.savefig(f'{prefix}scores_base_scores_comparison.pdf', format='pdf')
plt.show()

# 画出所有数据集的 score - base_score
if QUESTION_ONLY or FIXED_QUESTION:
    plt.figure(figsize=(10, 6))
else:    
    plt.figure(figsize=(8, 8))

for dataset in datasets:
    if dataset in ['HAL', 'AOKVQA']:
        if QUESTION_ONLY:
            ma_difference = [a-b for a, b in zip(data_results["wrong_"+dataset]['scores'], data_results["wrong_"+dataset]['base_scores'])]
        else:
            ma_difference = [a-b for a, b in zip(data_results["wrong_"+dataset]['scores'], data_results["wrong_"+dataset]['base_scores'])]
        shift_value = ma_difference[0]
        ma_difference = ma_difference - shift_value
    else:
        data_results[dataset]['result'] = [a-b for a, b in zip(data_results[dataset]['scores'], data_results[dataset]['base_scores'])]
        ma_difference = data_results[dataset]['result']
        shift_value = ma_difference[0]
        ma_difference = ma_difference - shift_value
    x_values = np.arange(1, len(ma_difference) + 1)  # 从1开始的 X 值
    plt.plot(x_values, ma_difference, linestyle='--', label=f'{datasets_title[dataset]}', linewidth=2)

plt.tick_params(axis='both', labelsize=24)
plt.xlabel('Layer Index', fontproperties=xylabel_comb)
plt.ylabel(r'$\mathcal{F}^{(l)}_T$', fontproperties=xylabel_comb)
plt.grid()
plt.axhline(0, color='black', linestyle='--', label='Zero Line')
plt.legend(prop=legend_font_comb)
plt.tight_layout()
plt.savefig(f'{prefix}combined_score_base_score_comparison.pdf', format='pdf')
print("save to", f"{prefix}combined_score_base_score_comparison.pdf")
plt.close()

# import matplotlib.pyplot as plt
# import numpy as np
# import torch

# # 设置字体为 Times New Roman
# from matplotlib.font_manager import FontProperties
# plt.rcParams['font.family'] = 'serif'
# legend_font_comb = FontProperties(fname='/home/junda/zihan/font/TimesNewRoman.ttf', size=24)
# legend_font_sep = FontProperties(fname='/home/junda/zihan/font/TimesNewRoman.ttf', size=22)
# xylabel_comb = FontProperties(fname='/home/junda/zihan/font/TimesNewRoman.ttf', size=26)
# xylabel_font = FontProperties(fname='/home/junda/zihan/font/TimesNewRoman.ttf', size=24)
# title_font = FontProperties(fname='/home/junda/zihan/font/TimesNewRoman.ttf', size=24)
# # legend_font_comb = FontProperties(fname='/home/junda/zihan/font/TimesNewRoman.ttf', size=24)
# # legend_font_sep = FontProperties(fname='/home/junda/zihan/font/TimesNewRoman.ttf', size=30)
# # xylabel_comb = FontProperties(fname='/home/junda/zihan/font/TimesNewRoman.ttf', size=32)
# # xylabel_font = FontProperties(fname='/home/junda/zihan/font/TimesNewRoman.ttf', size=28)
# # title_font = FontProperties(fname='/home/junda/zihan/font/TimesNewRoman.ttf', size=32)



# def custom_moving_average(data, window_size=2):
#     n = len(data)
#     ma = np.empty(n)  # 创建一个空数组用于存储移动平均值

#     for i in range(n):
#         # 获取左边和右边的索引
#         left_start = max(0, i - window_size)
#         left_end = i + 1
#         right_start = i + 1
#         right_end = min(n, i + window_size + 1)

#         # 获取左右值
#         left_values = data[left_start:left_end]
#         right_values = data[right_start:right_end]

#         # 计算平均值
#         all_values = np.concatenate((left_values, right_values))
#         ma[i] = np.mean(all_values)

#     return ma


# # # 加载 scores 和 base_scores
# # datasets = ['HAL', 'POPE', 'VQA', 'CHAIR']
# # datasets = ['HAL', 'COCO_caption', 'VQA', 'CHAIR']
# # datasets = ['HAL', 'COCO_caption', 'VQA', 'CHAIR', 'POPE']

# # datasets = ['HAL', 'COCO_caption', 'VQA', 'AOKVQA']
# datasets = ['VQA', 'COCO_caption', 'AOKVQA', 'HAL']
# datasets_title = {
#     'VQA': 'VQA',
#     'COCO_caption': 'COCO-caption',
#     'AOKVQA':'AOKVQA',
#     'HAL': 'HAL-Eval'
# }
# # datasets = ['correct_HAL', 'wrong_HAL', 'COCO_caption', 'VQA', 'correct_AOKVQA', 'wrong_AOKVQA']


# QUESTION_ONLY=False
# FIXED_QUESTION=False

# origin=True
# avg_token=False

# if QUESTION_ONLY:
#     prefix="question_only_"
# elif origin:
#     if not avg_token:
#         prefix="origin"
#     else:
#         prefix="originavg_"
# else:
#     prefix="" if not FIXED_QUESTION else "fixed_question_"


# data_results = {}

# for dataset in datasets:
#     if dataset in ['HAL', 'AOKVQA']:
#         # file_path_correct = f'correct_'
#         # file_path_wrong = f'wrong_{dataset.lower()}_scores_data.pth'
        
#         for file_type in ['correct_', 'wrong_']:
#             file_path = f"{prefix}{file_type}{dataset.lower()}_scores_data.pth"
#             print("loaded from:", file_path)
#             loaded_data = torch.load(file_path)
#             data_results[file_type+dataset] = {
#                 'scores': loaded_data['scores'],
#                 'base_scores': loaded_data['base_scores'],
#                 'result': [score-base for base, score in zip(loaded_data['base_scores'], loaded_data['scores'])]
#             }
#     else:
#         file_path = f'{prefix}{dataset.lower()}_scores_data.pth'
#         print("loaded from:", file_path)
#         loaded_data = torch.load(file_path)
#         data_results[dataset] = {
#             'scores': loaded_data['scores'],
#             'base_scores': loaded_data['base_scores'],
#             'result': [score-base for base, score in zip(loaded_data['base_scores'], loaded_data['scores'])]
#         }
    
# for i, dataset in enumerate(data_results.keys()):
#     data_results[dataset]['scores']=custom_moving_average(data_results[dataset]['scores'], window_size=2)
#     data_results[dataset]['base_scores']=custom_moving_average(data_results[dataset]['base_scores'], window_size=2)
    
# # 创建 2x2 子图
# fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# # 绘制每种数据的 base_scores 和 scores
# for i, dataset in enumerate(datasets):
#     ax = axs[i // 2, i % 2]  # 获取相应的子图
#     ax.tick_params(axis='both', labelsize=18)
    
#     if dataset in ['HAL', 'AOKVQA']:
#         if QUESTION_ONLY:
#             ax.plot(data_results["wrong_"+dataset]['scores'], marker='o', color='b', label='Task Scores')
#             ax.plot(data_results["wrong_"+dataset]['base_scores'], marker='x', color='g', label='Image Scores')
#         else:
#             ax.plot(data_results["correct_"+dataset]['scores'], marker='o', color='b', label='Correct Answer Text, Task Scores')
#             ax.plot(data_results["correct_"+dataset]['base_scores'], marker='x', color='g', label='Correct Answer Text, Image Scores')
#             ax.plot(data_results["wrong_"+dataset]['scores'], marker='o', color='r', label='Wrong Answer Text, Task Scores')
#             ax.plot(data_results["wrong_"+dataset]['base_scores'], marker='x', color='orange', label='Wrong Answer Text, Image Scores')
#             # ax.plot(data_results["wrong_"+dataset]['scores'], marker='o', color='b', label=r'$I(X^{(l)}_{T}; X^{(L)}_{T})$')
#             # ax.plot(data_results["wrong_"+dataset]['base_scores'], marker='x', color='g', label=r'$I(X^{(l)}_{0}; X^{(L)}_{0})$')

#         ax.set_title(datasets_title[dataset], fontproperties=title_font)
#         ax.set_xlabel('Layer Index', fontproperties=xylabel_font)
#         ax.set_ylabel(r'$\mathcal{F}^{(l)}_T$', fontproperties=xylabel_font)
#         ax.grid()
    
#     else:
#         ax.plot(data_results[dataset]['scores'], marker='o', color='b', label=r'$I(X^{(l)}_{T}; X^{(L)}_{T})$')
#         ax.plot(data_results[dataset]['base_scores'], marker='x', color='g', label=r'$I(X^{(l)}_{0}; X^{(L)}_{0})$')
#         ax.set_title(datasets_title[dataset], fontproperties=title_font)
#         ax.set_xlabel('Layer Index', fontproperties=xylabel_font)
#         ax.set_ylabel(r'$\mathcal{F}^{(l)}_T$', fontproperties=xylabel_font)
#         ax.grid()

# # 添加共享 legend
# handles, labels = axs[1, 1].get_legend_handles_labels()  # 获取第一个子图的 legend
# fig.legend(handles, labels, loc='upper center', prop=legend_font_sep, ncol=2)

# # 调整布局
# plt.tight_layout(rect=[0, 0.03, 1, 0.9])  # 保持标题的空间
# plt.savefig(f'{prefix}scores_base_scores_comparison.pdf', format='pdf')
# plt.show()

# # 画出所有数据集的 score - base_score
# if QUESTION_ONLY or FIXED_QUESTION:
#     plt.figure(figsize=(10, 6))
# else:    
#     plt.figure(figsize=(8, 8))  # 调整为更扁平的尺寸

# for dataset in datasets:
    
#     if dataset in ['HAL', 'AOKVQA']:
#         if QUESTION_ONLY:
#             ma_difference = [a-b for a, b in zip(data_results["wrong_"+dataset]['scores'], data_results["wrong_"+dataset]['base_scores'])]
#         else:
#             ma_difference = [a-b for a, b in zip(data_results["wrong_"+dataset]['scores'], data_results["wrong_"+dataset]['base_scores'])]
#         shift_value = ma_difference[0]
#         ma_difference = ma_difference - shift_value
#     else:
#         # plt.plot(data_results[dataset]['result'], marker='o', label=f'{dataset}')
#         data_results[dataset]['result'] = [a-b for a, b in zip(data_results[dataset]['scores'], data_results[dataset]['base_scores'])]
#         ma_difference = data_results[dataset]['result'] #custom_moving_average(data_results[dataset]['result'], window_size=2)
#         ##### shift
#         shift_value = ma_difference[0]
#         ma_difference = ma_difference - shift_value
#     #####
#     plt.plot(ma_difference, linestyle='--', label=f'{datasets_title[dataset]}', linewidth=2)


# # plt.title('Score - Base Score Comparison Across Datasets')
# plt.tick_params(axis='both', labelsize=24)
# plt.xlabel('Layer Index', fontproperties=xylabel_comb)
# plt.ylabel(r'$\mathcal{F}^{(l)}_T$', fontproperties=xylabel_comb)
# plt.grid()
# plt.axhline(0, color='black', linestyle='--', label='Zero Line')
# plt.legend(prop=legend_font_comb)
# plt.tight_layout()
# plt.savefig(f'{prefix}combined_score_base_score_comparison.pdf', format='pdf')
# print("save to", f"{prefix}combined_score_base_score_comparison.pdf")
# plt.close()