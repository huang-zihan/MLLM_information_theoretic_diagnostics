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
seperate_layer=False

text_margin=False
correct_answer=False

QUESTION_ONLY=False
FIXED_QUESTION=False

avg_token=False
origin=True

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

# epoch=49
# epoch=25
epoch=19

ce_data_dir="../cbm/aokvqa/"

for text_margin in [True, False]:

    for correct_answer in [True, False]: #True, False

        if correct_answer:
            # if QUESTION_ONLY or FIXED_QUESTION:
            labels = torch.load(ce_data_dir+f'{prefix}correct_aokvqa_label_list_v1.pth')
            print("load from:", ce_data_dir+f'{prefix}correct_aokvqa_label_list_v1.pth')
            # else:
            #     labels = torch.load(ce_data_dir+f'correct_aokvqa_label_list_v1.pth')

            input_datas = torch.load(ce_data_dir+f'{prefix}correct_aokvqa_info_probe_list_v1.pth')
            print(ce_data_dir+f'{prefix}correct_aokvqa_info_probe_list_v1.pth')
            missing_index = [795, 875, 976, 1074]
        else:
            # if QUESTION_ONLY or FIXED_QUESTION:
            labels = torch.load(ce_data_dir+f'{prefix}wrong_aokvqa_label_list_v1.pth')
            print("load from:", ce_data_dir+f'{prefix}wrong_aokvqa_label_list_v1.pth')
            # else:
            #     labels = torch.load(ce_data_dir+f'wrong_aokvqa_label_list_v1.pth')

            input_datas = torch.load(ce_data_dir+f'{prefix}wrong_aokvqa_info_probe_list_v1.pth')
            print(ce_data_dir+f'{prefix}wrong_aokvqa_info_probe_list_v1.pth')
            
            missing_index = [2385, 2386, 2387, 2625, 2626, 2627, 2928, 2929, 2930, 3222, 3223, 3224]

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

        for margin_num in margin_num_list:
            
            temp_prefix = prefix + ("correct_" if correct_answer else "wrong_")
            # prefix = "correct_" if correct_answer else "wrong_"
            if is_qwen:
                temp_prefix ="qwen_"+temp_prefix
            
            if text_margin:
                written_file=f'{ce_data_dir}/{temp_prefix}aokvqa_all_kls_text_margin_{margin_num}_trainepoch{epoch}_v1'
                pic_file=f'{ce_data_dir}/{temp_prefix}aokvqa_line_chart_text_margin_{margin_num}_trainepoch{epoch}_v1'
            else:
                written_file=f'{ce_data_dir}/{temp_prefix}aokvqa_all_kls_merge_trainepoch{epoch}_v1'
                pic_file=f'{ce_data_dir}/{temp_prefix}aokvqa_line_chart_merge_trainepoch{epoch}_v1'
                
                if seperate_layer:
                    pic_file=f'{ce_data_dir}/{temp_prefix}aokvqa_line_chart_seperatelayer_trainepoch{epoch}_v1'

            if use_avg_prior:
                written_file+="_avg"

            written_file+=".txt"
            pic_file+=".pdf"

            if text_margin:
                index_to_cluster=torch.load(f'margin/{("correct_" if correct_answer else "wrong_")}aokvqa_index_to_cluster-{margin_num}.pth', weights_only=False)
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
            # print("info_save_index length:", len(info_save_index))
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

            print("written to", written_file)
            with open(written_file, 'w') as f:
                for item in all_kls:
                    f.write(f"{item}\n")
            print("written to", written_file)
