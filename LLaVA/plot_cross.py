import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


mask=False
# ('cross', 'dog', 'backpack')
classify_type_list=[('cross', 'chair', 'keyboard'), ('cross', 'chair', 'dog'), ('cross', 'chair', 'backpack'), ('cross', 'dog', 'keyboard'), ('cross', 'backpack', 'keyboard')]

# ('counter', 'backpack'), ('counter', 'chair'), ('counter', 'keyboard')

for classify_type in classify_type_list:
    
    print("processing:", classify_type)
    
    # 读取文件
    results_pos = {}
    results_neg = {}

    info_list=f"pope_result_{classify_type[1]}.pth"
    pic_dir=f'pope_acc_result_{classify_type[1]}_{classify_type[2]}_{classify_type[0]}.pdf'


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

    item1_list=item_to_image[classify_type[1]]
    item2_list=item_to_image[classify_type[2]]
    indices_item1 = []
    indices_item2 = []
    for index in item1_list:
        if 6*index<train_size:
            continue
        index=6*index-train_size
        indices_item1.extend(range(index, index + 6))
    
    for index in item2_list:
        if 6*index<train_size:
            continue
        index=6*index-train_size
        indices_item2.extend(range(index, index + 6))
    
    print(len(indices_item1), len(indices_item2))
    expect_len=min(len(indices_item1), len(indices_item2))
    indices_item1=indices_item1[:expect_len]
    indices_item2=indices_item2[:expect_len]

    # candidate=[i for i in range(test_size) if i not in indices_pos]
    # indices_neg=candidate[:len(indices_pos)]
    

    info_list_index=[0, 12, 24, 27, 30, 31]
    # info_list_index=[0, 12]
    print(indices_item1)
    print(indices_item2)
    print(len(indices_item1))
    print(len(indices_item2))
    for epoch in range(len(datas[info_list_index[0]]['classifiy_res'])):
        # if epoch == 5:
        #     break
        print(epoch,"/",len(datas[info_list_index[0]]['classifiy_res']), end='\r')
        for index in info_list_index:
            if index not in results_pos:
                results_pos[index] = {'epochs': [], 'accuracies': []}
            
            if index not in results_neg:
                results_neg[index] = {'epochs': [], 'accuracies': []}
            
            predict=datas[index]['classifiy_res'][epoch]
            gt=datas[index]['gt']
            
            pos_predict = [x.cpu() for idx, x in enumerate(predict) if idx in indices_item1]
            pos_gt=[x.cpu() for idx, x in enumerate(gt) if idx in indices_item1]
            
            neg_predict = [x.cpu() for idx, x in enumerate(predict) if idx in indices_item2]
            neg_gt=[x.cpu() for idx, x in enumerate(gt) if idx in indices_item2]
            
            # target_index=item_dict[classify_type[1]]
            
            # correct_predictions = sum(1 for p, g in zip(predict, gt) if p[target_index] == g[target_index])
            # total_predictions = len(predict)
            # # 计算准确率
            # accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            # print(correct_predictions, total_predictions, accuracy)
            pos_f1=[f1_score(pos_predict[i], pos_gt[i]) for i in range(len(pos_predict))]
            pos_accuracy=float(sum(pos_f1))/float(len(pos_f1))
            
            neg_f1=[f1_score(neg_predict[i], neg_gt[i]) for i in range(len(neg_predict))]
            neg_accuracy=float(sum(neg_f1))/float(len(neg_f1))
            # accuracy=f1_score(predict, gt)

            # print(accuracy)
            
            results_pos[index]['epochs'].append(epoch)
            results_pos[index]['accuracies'].append(pos_accuracy)
            
            results_neg[index]['epochs'].append(epoch)
            results_neg[index]['accuracies'].append(neg_accuracy)


    plot_index=info_list_index
    # 绘制pos准确性图像
    plt.figure(figsize=(6, 4))
    for index, metrics in results_pos.items():
        if index not in plot_index:
            continue
        plt.plot(metrics['epochs'], metrics['accuracies'], label=f'Index {index}')

    plt.title('Model Accuracy per Epoch')
    plt.xlabel('Epoch')
    if 'pope' in classify_type or 'cross' in classify_type:
        plt.ylabel('F1')
    else:
        plt.ylabel('Accuracy (%)')
        
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("pos_"+pic_dir, format='pdf')
    print("pos_"+pic_dir)
    plt.close()  # 关闭图像以释放内存
    
    
    # 绘制neg准确性图像
    plt.figure(figsize=(6, 4))
    for index, metrics in results_neg.items():
        if index not in plot_index:
            continue
        plt.plot(metrics['epochs'], metrics['accuracies'], label=f'Index {index}')

    plt.title('Model Accuracy per Epoch')
    plt.xlabel('Epoch')
    if 'pope' in classify_type or 'cross' in classify_type:
        plt.ylabel('F1')
    else:
        plt.ylabel('Accuracy (%)')
        
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("neg_"+pic_dir, format='pdf')
    print("neg_"+pic_dir)
    plt.close()  # 关闭图像以释放内存


