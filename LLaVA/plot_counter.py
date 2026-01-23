import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


mask=False
classify_type_list=[('counter', 'dog'), ('counter', 'backpack'), ('counter', 'chair'), ('counter', 'keyboard')]
for classify_type in classify_type_list:
    
    print("processing:", classify_type)
    
    # 读取文件
    results = {}

    info_list=f"pope_result_{classify_type[1]}.pth"
    pic_dir=f'pope_acc_result_{classify_type[1]}_{classify_type[0]}.pdf'


    item_dict = {'snowboard': 0, 'backpack': 1, 'person': 2, 'car': 3, 'skis': 4, 'dog': 5, 'truck': 6, 'dining table': 7, 'handbag': 8, 'bicycle': 9, 'motorcycle': 10, 'potted plant': 11, 'vase': 12, 'traffic light': 13, 'bus': 14, 'chair': 15, 'bed': 16, 'book': 17, 'spoon': 18, 'cup': 19, 'fork': 20, 'tv': 21, 'toaster': 22, 'microwave': 23, 'bottle': 24, 'bird': 25, 'boat': 26, 'couch': 27, 'sandwich': 28, 'bowl': 29, 'hot dog': 30, 'frisbee': 31, 'knife': 32, 'cake': 33, 'remote': 34, 'baseball glove': 35, 'sports ball': 36, 'baseball bat': 37, 'bench': 38, 'sink': 39, 'toilet': 40, 'teddy bear': 41, 'bear': 42, 'cat': 43, 'mouse': 44, 'laptop': 45, 'toothbrush': 46, 'cow': 47, 'skateboard': 48, 'surfboard': 49, 'cell phone': 50, 'train': 51, 'clock': 52, 'tennis racket': 53, 'suitcase': 54, 'horse': 55, 'banana': 56, 'wine glass': 57, 'refrigerator': 58, 'carrot': 59, 'broccoli': 60, 'tie': 61, 'scissors': 62, 'sheep': 63, 'airplane': 64, 'stop sign': 65, 'fire hydrant': 66, 'keyboard': 67, 'pizza': 68, 'donut': 69, 'kite': 70, 'parking meter': 71, 'giraffe': 72, 'zebra': 73, 'umbrella': 74, 'orange': 75, 'oven': 76, 'elephant': 77, 'apple': 78}

    datas=torch.load(info_list)

    ####
    train_dataset = load_dataset("lmms-lab/POPE", split="test")
    # 划分训练集和测试集
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    ####

    # print(datas[0]['classifiy_res'][0])
    # print(datas[0]['gt'])
    print(len(datas[0]['classifiy_res']))
    print(len(datas[0]['gt']))

    pope_meta=torch.load('pope_meta_data.pth')
    item_to_image=pope_meta['item_to_image']
    image_to_item=pope_meta['image_to_item']

    delect_list=item_to_image[classify_type[1]]
    indices_to_remove = []
    print(delect_list)
    for index in delect_list:
        if 6*index<train_size:
            continue
        index=6*index-train_size
        indices_to_remove.extend(range(6*index, 6*index + 6))
    print(indices_to_remove)
    info_list_index=[0, 12, 24, 27, 30, 31]
    # info_list_index=[0, 12]

    for epoch in range(len(datas[info_list_index[0]]['classifiy_res'])):
        # if epoch == 5:
        #     break
        print(epoch,"/",len(datas[info_list_index[0]]['classifiy_res']), end='\r')
        for index in info_list_index:
            if index not in results:
                results[index] = {'epochs': [], 'accuracies': []}
            
            predict=datas[index]['classifiy_res'][epoch]
            gt=datas[index]['gt']
            
            predict = [x.cpu() for idx, x in enumerate(predict) if not idx in indices_to_remove]
            gt=[x.cpu() for idx, x in enumerate(gt) if not idx in indices_to_remove]
            target_index=item_dict[classify_type[1]]
            
            # correct_predictions = sum(1 for p, g in zip(predict, gt) if p[target_index] == g[target_index])
            # total_predictions = len(predict)
            # # 计算准确率
            # accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            # print(correct_predictions, total_predictions, accuracy)
            f1=[f1_score(predict[i], gt[i]) for i in range(len(predict))]
            accuracy=float(sum(f1))/float(len(f1))
            # accuracy=f1_score(predict, gt)

            # print(accuracy)
            
            
            results[index]['epochs'].append(epoch)
            results[index]['accuracies'].append(accuracy)

    # exit()

    # with open(written_file, "r") as f:
    #     next(f)  # 跳过标题行
    #     for line in f:
    #         index, epoch, loss, accuracy = line.strip().split('\t')
    #         index = int(index)
    #         epoch = int(epoch)
    #         accuracy = float(accuracy)

    #         if index not in results:
    #             results[index] = {'epochs': [], 'accuracies': []}
            
    #         results[index]['epochs'].append(epoch)
    #         results[index]['accuracies'].append(accuracy)

    # plot_index=[0,12,24,31]
    plot_index=info_list_index
    # 绘制准确性图像
    plt.figure(figsize=(6, 4))
    for index, metrics in results.items():
        if index not in plot_index:
            continue
        plt.plot(metrics['epochs'], metrics['accuracies'], label=f'Index {index}')

    plt.title('Model Accuracy per Epoch')
    plt.xlabel('Epoch')
    if 'pope' in classify_type or 'cross' in classify_type:
        plt.ylabel('F1')
    else:
        plt.ylabel('Accuracy (%)')
    # plt.xticks(range(1, max(metrics['epochs']) + 1))
    # if classify_type=='cifar10':
    #     if mask:
    #         plt.ylim(60, 90)
    #     else:
    #         plt.ylim(85, 100)
    # plt.ylim(85, 100)
    # elif classify_type=='cifar100':
    #     if mask:
    #         plt.ylim(40, 65)
    #     else:
    #         plt.ylim(60, 90)
        
    plt.legend()
    plt.grid()
    plt.tight_layout()
    # plt.show()

    # 保存图像到文件
    # plt.savefig(pic_dir)
    plt.savefig(pic_dir, format='pdf')
    plt.close()  # 关闭图像以释放内存


