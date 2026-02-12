import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn.functional as F
import matplotlib.pyplot as plt

is_qwen=True

if is_qwen:
    info_save_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]
else:
    info_save_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]    
use_avg_prior=False
text_margin=False
seperate_layer=False
avg_token=False
origin=True

QUESTION_ONLY=False
FIXED_QUESTION=False
if QUESTION_ONLY:
    prefix="question_only_"
elif origin:
    if not avg_token:
        prefix="origin"
    else:
        prefix="originavg_"
elif FIXED_QUESTION:
    prefix="fixed_question_"
elif avg_token:
    prefix="newavg_"
else:
    prefix=""

ce_data_dir="../cbm/coco_caption/"

# if FIXED_QUESTION or QUESTION_ONLY:
labels = torch.load(ce_data_dir+f'{prefix}coco_caption_ce_val_label_full_v1.pth')
print("load from:", ce_data_dir+f'{prefix}coco_caption_ce_val_label_full_v1.pth')
# else:
# labels = torch.load(ce_data_dir+f'coco_caption_ce_val_label_full_v1.pth')
    # print("load from:", ce_data_dir+f'coco_caption_ce_val_label_full_v1.pth')
input_datas = torch.load(ce_data_dir+f'{prefix}coco_caption_ce_val_full_v1.pth')
print(ce_data_dir+f'{prefix}coco_caption_ce_val_full_v1.pth')
print()

missing_index = [73, 98, 188, 190, 232, 235, 258, 267, 275, 287, 297, 309, 316, 338, 365, 423, 470, 480, 483, 657, 1256, 1284, 1323, 1453, 1484, 1654, 1701, 1717, 1726, 1729, 1794, 1796, 1815, 1830, 1871, 1891, 1942, 1954, 2004, 2017, 2037, 2038, 2082, 2096, 2144, 2207, 2282, 2373, 2477, 2502, 2529, 2631, 2778, 2850, 2912, 2941, 2952, 2953, 3013, 3022, 3049, 3067, 3069, 3081, 3185, 3188, 3196, 3202, 3212, 3242, 3296, 3309, 3344, 3355, 3494, 3611, 3687, 3699, 3797, 3824, 3887, 3892, 3999, 4003, 4081, 4819, 4855, 4862, 4980, 5001, 5029, 5096, 5112, 5189, 5280, 5309, 5393, 5523, 5537, 5618, 5669, 5784, 5815, 5854, 5992, 6059, 6063, 6101, 6125, 6312, 6456, 6590, 6597, 6611, 6632, 6666, 6720, 6728, 6731, 6763, 7493, 7523, 7605, 7675, 7694, 7717, 7773, 7778, 7892, 7895, 8206, 8521, 8533, 8689, 8707, 8759, 8802, 8821, 8825, 8921, 8951, 9020, 9021, 9073, 9112, 9134, 9139, 9147, 9251, 9257, 9320, 9363, 9430, 9431, 9440, 9477, 9484, 9490, 9518, 9538, 9541, 9564, 9603, 9606, 9689, 9721, 9740, 9754, 9864, 9932, 9974, 9992]
labels = [x for i, x in enumerate(labels) if i not in missing_index]

if use_avg_prior:
    sum_tensor = None
    num_tensors = len(labels)
    for tensor in labels:
        tensor=torch.tensor(tensor)
        if sum_tensor is None:
            sum_tensor = tensor
        else:
            sum_tensor += tensor
    average_tensor = [float(x) for x in list(sum_tensor / num_tensors)]
    labels = [average_tensor for _ in range(len(labels))]

for i in range(len(input_datas)):
    input_datas[i]=[x for i, x in enumerate(input_datas[i]) if i not in missing_index]

# margin_num_list=[3,5,10,20,30,40,50,100]
margin_num_list=[3] # ,20,30,40,50,100
# epoch=25
epoch=19
# epoch=49

if is_qwen:
    prefix="qwen_"+prefix
    
for text_margin in [True, False]:

    for margin_num in margin_num_list:
            
        if text_margin:
            written_file=f'result/{prefix}coco_caption_all_kls_text_margin_{margin_num}_trainepoch{epoch}_v1'
            pic_file=f'result/{prefix}coco_caption_line_chart_text_margin_{margin_num}_trainepoch{epoch}_v1'
        else:
            written_file=f'result/{prefix}coco_caption_all_kls_merge_trainepoch{epoch}_v1'
            pic_file=f'result/{prefix}coco_caption_line_chart_merge_trainepoch{epoch}_v1'
            if seperate_layer:
                pic_file=f'result/{prefix}coco_caption_line_chart_seperatelayer_trainepoch{epoch}_v1'

        if use_avg_prior:
            written_file+="_avg"

        written_file+=".txt"
        pic_file+=".pdf"

        if text_margin:
            index_to_cluster=torch.load(f'margin/coco_caption_index_to_cluster-{margin_num}.pth', weights_only=False)
            seperate_kl=[[] for _ in range(margin_num)]

        vit=False
        one_item=False

        class CE(nn.Module):
            def __init__(self):
                super(CE, self).__init__()

                self.bn = nn.BatchNorm1d(2048 if is_qwen else 4096)
        
                if is_qwen:
                    self.fc1 = nn.Linear(2048, 512)
                else:
                    self.fc1 = nn.Linear(4096, 512)
                self.fc2 = nn.Linear(512, 128)
                self.fc3 = nn.Linear(128, len(labels[0]))

            def forward(self, x):
                x = self.bn(x)
                
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x


        if vit==True:
            written_file="vit_"+written_file
            info_save_index=[0]

        results={}
        inference_result={}
        all_kls=[]
        
        if seperate_layer == False:
            model = CE()
            model.load_state_dict(torch.load(f'./result/ckpt/ce_model_merge_epoch{epoch}_v1.pth'))
            model.to('cuda:0')
            
        for i, index in enumerate(info_save_index):
            print(f"-----------------index {index}-----------------")
            
            if seperate_layer:

                model = CE()
                model.load_state_dict((torch.load(f'./result/ckpt/ce_model_{i}_epoch{epoch}_v1.pth')))
                # model.load_state_dict((torch.load(f'./result/ce_model_{i}.pth')))
                model.to('cuda:0')

            if not vit:
                input_data=input_datas[i]

            if vit:
                data_tensor = torch.stack(vit_feature_list)
            else:
                data_tensor = torch.stack(input_data)  # (N, 1024)

            labels_tensor = torch.tensor(labels, device='cuda:0')

            test_dataset = TensorDataset(data_tensor, labels_tensor)
            print("test_dataset", len(test_dataset))
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

            all_losses=[]
            all_f1=[]
            all_accuracies = []
            classifiy_res_index=[]
            gt_res_index=[]

            cnt=0
            with torch.no_grad():
                total_kl=[]
                for inputs, target in test_loader:
                    inputs = inputs.float().to('cuda:0')
                    predicted = model(inputs)

                    for j in range(len(target)):
                        if torch.all(target[j] == 0):
                            cnt+=1
                            continue

                        if text_margin:
                            seperate_kl[index_to_cluster[j]].append(F.kl_div(torch.log_softmax(predicted[j], dim=0), target[j].float()))
                        else:
                            total_kl.append(F.kl_div(torch.log_softmax(predicted[j], dim=0), target[j].float()))

            if text_margin:
                averages = []
                for sublist in seperate_kl:
                    if sublist:
                        avg = sum(sublist) / len(sublist)
                        averages.append(avg)

                if averages:
                    overall_avg = sum(averages) / len(averages)
                else:
                    overall_avg = None
                all_kls.append(overall_avg)
                print(f'KL: {overall_avg}')
            else:
                print("total_kl length", len(total_kl))
                print(f'KL: {sum(total_kl)/len(total_kl)}')
                all_kls.append(sum(total_kl)/len(total_kl))

        all_kls=[float(x.cpu()) for x in all_kls]
        
        with open(written_file, 'w') as f:
            for item in all_kls:
                f.write(f"{item}\n")
        print("written to:", written_file)