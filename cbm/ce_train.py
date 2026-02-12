import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn.functional as F
import random

# info_save_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]    
# For Qwen
info_save_index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]    

ce_data_dir="../cbm/data/train/"

# Train model
num_epochs = 50

input_datas = torch.load(ce_data_dir+'ce_training_full_v1.pth')
coco_missing_obj_list=[31, 61, 149, 201, 209, 281, 481, 486, 514, 553, 579, 585, 678, 733, 852, 864, 1009, 1033, 1053, 1131, 1183, 1331, 1334, 1382, 1425, 1457, 1467, 1491, 1512, 1603, 1802, 1848, 1893, 2140, 2157, 2212, 2224, 2307, 2310, 2328, 2366, 2676, 2876, 2880, 2969, 3052, 3105, 3162, 3177, 3341, 3421, 3654, 3663, 3684, 3695, 3723, 3863, 4023, 4097, 4101, 4114, 4307, 4490, 4544, 4588, 4672, 4713, 4736, 4754, 4797, 4918, 4972, 4979, 5007, 5041, 5095, 5109, 5148, 5191, 5256, 5334, 5384, 5456, 5464, 5495, 5614, 5668, 5814, 5820, 5927, 5973, 6021, 6389, 6451, 6453, 6502, 6506, 6521, 6558, 6625, 6739, 7068, 7070, 7203, 7278, 7325, 7333, 7340, 7622, 7699, 7750, 7880, 7922, 7977, 8029, 8055, 8078, 8200, 8223, 8304, 8354, 8357, 8407, 8418, 8465, 8495, 8523, 8751, 8793, 8880, 9107, 9109, 9160, 9222, 9310, 9315, 9390, 9572, 9582, 9640, 9681, 9742, 9886, 9895, 9917, 9997, 10043, 10128, 10177, 10184, 10221, 10234, 10399, 10475, 10528, 10552, 10729, 10743, 10763, 10766, 10862, 10863, 10912, 10972, 10995, 11064, 11075, 11135, 11335, 11361, 11450, 11535, 11545, 11731, 11777, 12069, 12096, 12378, 12382, 12384, 12430, 12476, 12520, 12526, 12555, 12594, 12643, 12736, 12813, 12882, 13010, 13040, 13295, 13367, 13395, 13418, 13484, 13583, 13790, 13854, 13932, 13933, 13985, 13991, 14061, 14113, 14124, 14166, 14400, 14409, 14507, 14510, 14609, 14659, 14685, 14690, 14746, 14780, 14834, 14981, 15041, 15067, 15227, 15382, 15414, 15436, 15438, 15489, 15506, 15549, 15676, 15739, 15849, 15881, 15934, 15971, 15982, 16007, 16052, 16094, 16099, 16130, 16176, 16230, 16333, 16359, 16383, 16439, 16454, 16491, 16506, 16525, 16911, 17016, 17049, 17170, 17224, 17275, 17425, 17429, 17529, 17604, 17709, 17710, 17723, 17778, 17788, 17825, 18002, 18154, 18348, 18366, 18413, 18453, 18524, 18638, 18701, 18739, 18779, 18938, 18964, 19036, 19189, 19247, 19322, 19417, 19478, 19509, 19545, 19550, 19574, 19580, 19724, 19790, 19839, 19905, 19911, 19940, 19976, 20054, 20109, 20112, 20156, 20185, 20202, 20208, 20233, 20340, 20363, 20378, 20443, 20493, 20550, 20581, 20654, 20920, 21123, 21136, 21159, 21223, 21233, 21245, 21266, 21319, 21323, 21547, 21815, 22009, 22091, 22099, 22214, 22572, 22586, 22601, 22662, 22665, 22667, 22679, 22728, 22806, 22817, 22846, 22886, 22981, 22989, 23023, 23093, 23196, 23231, 23338, 23432, 23499, 23617, 23728, 23806, 23912, 23921, 23931, 24092, 24208, 24317, 24384, 24584, 24619, 24702, 24776, 24806, 24913, 24926, 25187, 25199, 25233, 25426, 25642, 25707, 25761, 25788, 25825, 25860, 26034, 26134, 26325, 26391, 26415, 26417, 26445, 26468, 26524, 26581, 26738, 26744, 26851, 26870, 27021, 27041, 27107, 27173, 27271, 27279, 27423, 27460, 27478, 27521, 27544, 27562, 27621, 27660, 27671, 27724, 27761, 27762, 27808, 27812, 27965, 28143, 28205, 28255, 28272, 28340, 28453, 28484, 28613, 28717, 28767, 28822, 28827, 28885, 28889, 28995, 29207, 29248, 29408, 29528, 29640, 29673, 29802, 29834, 29906, 29962, 30016, 30230, 30268, 30312, 30382, 30433, 30573, 30800, 30874, 30975, 31033, 31062, 31213, 31265, 31271, 31318, 31348, 31372, 31414, 31424, 31501, 31517, 31533, 31556, 31657, 31670, 31751, 31779, 31863, 31994, 32033, 32082, 32179, 32219, 32263, 32291, 32295, 32391, 32393, 32397, 32402, 32419, 32437, 32712, 32758, 32769, 32788, 32855, 32885, 32964, 33211, 33274, 33299, 33306, 33308, 33321, 33323, 33351, 33394, 33429, 33530, 33733, 33777, 33778, 33800, 33875, 33986, 34058, 34159, 34167, 34272, 34408, 34437, 34452, 34461, 34470, 34473, 34547, 34615, 34647, 34703, 34726, 34783, 34806, 34838, 34850, 35057, 35097, 35129, 35206, 35411, 35458, 35535, 35580, 35588, 35708, 35726, 35738, 35747, 35764, 35834, 35846, 35914, 35978, 36030, 36037, 36085, 36280, 36377, 36394, 36453, 36463, 36604, 36644, 36657, 36697, 36838, 36878, 36955, 36958, 36988, 36999, 37233, 37358, 37526, 37551, 37599, 37707, 37716, 37717, 37746, 37779, 37799, 37875, 37891, 37933, 37943, 37961, 38035, 38082, 38111, 38193, 38427, 38434, 38445, 38568, 38617, 38658, 38836, 38983, 38998, 39051, 39095, 39134, 39138, 39254, 39267, 39437, 39471, 39510, 39562, 39574, 39613, 39709, 39748, 39782, 39793, 39819, 39831, 39871, 39900, 39949, 39955, 39978, 40014, 40038, 40058, 40095, 40291, 40324, 40332, 40528, 40544, 40576, 40594, 40659, 40676, 40747, 40780, 40817, 40839, 40891, 40892, 40909, 40928, 40993, 41032, 41119, 41293, 41306, 41378, 41424, 41501, 41531, 41675, 41698, 41705, 41796, 41896, 41981, 42096, 42171, 42259, 42425, 42426, 42486, 42490, 42498, 42630, 42701, 42955, 42965, 43007, 43087, 43289, 43441, 43571, 43606, 43674, 43814, 43839, 44050, 44367, 44572, 44661, 44735, 44753, 44837, 44865, 45292, 45373, 45427, 45439, 45674, 45712, 45733, 45930, 46000, 46053, 46139, 46158, 46357, 46411, 46559, 46670, 46782, 46842, 47149, 47247, 47265, 47440, 47543, 47561, 47772, 47899, 48118, 48179, 48275, 48395, 48449, 48768, 48817, 48821, 48822, 49014, 49047, 49117, 49146, 49241, 49302, 49307, 49327, 49360, 49374, 49441, 49536, 49596, 49599, 49608, 49624, 49648, 49674, 49683, 49684, 49723, 50023, 50062, 50180, 50184, 50317, 50442, 50465, 50493, 50514, 50793, 50815, 50930, 51031, 51124, 51142, 51217, 51430, 51455, 51505, 51556, 51565, 51571, 51588, 51668, 51708, 51840, 51910, 52049, 52077, 52082, 52093, 52205, 52254, 52348, 52381, 52418, 52425, 52511, 52525, 52650, 52735, 52754, 52803, 52910, 52911, 52923, 52951, 52960, 52971, 53051, 53123, 53399, 53413, 53474, 53584, 53631, 53683, 53798, 53846, 53905, 53938, 53956, 53967, 54192, 54229, 54354, 54355, 54384, 54485, 54493, 54730, 54737, 54837, 54883, 54978, 55344, 55578, 55626, 55641, 55686, 55744, 55750, 55783, 55791, 55818, 55821, 55871, 55978, 56007, 56038, 56040, 56073, 56090, 56113, 56207, 56404, 56549, 56567, 56736, 56856, 57006, 57034, 57222, 57268, 57343, 57400, 57558, 57608, 57677, 57682, 57783, 57787, 57816, 57843, 57847, 57864, 57898, 57923, 57931, 57967, 58015, 58138, 58234, 58388, 58419, 58436, 58444, 58544, 58553, 58565, 58623, 58810, 58816, 58872, 58888, 58889, 58988, 59002, 59036, 59212, 59293, 59331, 59498, 59837, 60000, 60233, 60268, 60468, 60523, 60600, 60630, 60747, 60818, 60831, 60982, 61019, 61033, 61034, 61052, 61189, 61210, 61261, 61352, 61380, 61481, 61531, 61579, 61642, 61695, 61761, 61762, 61768, 61784, 61992, 62001, 62082, 62091, 62103, 62127, 62149, 62319, 62440, 62576, 62671, 62701, 62733, 62770, 62829, 62929, 62954, 62970, 62995, 63029, 63119, 63174, 63193, 63205, 63324, 63464, 63522, 63539, 63543, 63570, 63721, 63909, 63915, 63949, 63986, 63987, 64389, 64399, 64417, 64418, 64454, 64457, 64480, 64481, 64488, 64492, 64531, 64630, 64637, 64696, 64731, 64865, 64886, 64970, 65047, 65049, 65055, 65161, 65325, 66073, 66079, 66186, 66213, 66215, 66297, 66313, 66395, 66464, 66478, 66611, 66740, 66792, 66833, 66990, 67205, 67376, 67402, 67501, 67509, 67532, 67739, 67778, 68030, 68061, 68101, 68173, 68209, 68216, 68299, 68341, 68355, 68434, 68466, 68551, 68591, 68695, 68766, 68841, 69105, 69125, 69168, 69216, 69222, 69299, 69330, 69415, 69443, 69444, 69621, 69646, 69677, 69703, 69773, 69932, 70485, 70546, 70622, 70757, 70855, 70918, 70932, 70969, 70980, 71003, 71053, 71071, 71096, 71110, 71126, 71136, 71145, 71161, 71510, 71533, 71676, 71757, 71830, 71871, 71893, 71934, 71956, 71967, 72000, 72037, 72122, 72148, 72176, 72187, 72226, 72249, 72260, 72284, 72387, 72397, 72426, 72524, 72742, 72999, 73088, 73128, 73195, 73285, 73321, 73464, 73503, 73528, 73548, 73594, 73622, 73762, 73819, 73995, 74010, 74072, 74128, 74274, 74351, 74439, 74498, 74564, 74674, 74688, 74702, 74841, 74858, 75021, 75121, 75202, 75244, 75299, 75463, 75477, 75499, 75571, 75584, 75648, 75682, 75792, 75933, 75938, 75941, 75954, 76108, 76263, 76293, 76316, 76447, 76469, 76478, 76482, 76489, 76743, 76839, 76924, 77007, 77038, 77078, 77143, 77156, 77229, 77237, 77364, 77406, 77517, 77753, 77896, 77961, 77974, 78014, 78027, 78052, 78180, 78270, 78281, 78285, 78425, 78452, 78460, 78554, 78566, 78583, 78619, 78797, 78875, 78895, 78955, 79120, 79174, 79196, 79204, 79215, 79236, 79254, 79278, 79360, 79485, 79489, 79633, 79654, 79768, 79856, 79948, 80070, 80445, 80486, 80601, 80677, 80781, 80888, 80974, 81243, 81350, 81358, 81446, 81466, 81510, 81537, 81651, 81708, 81744, 81809, 81958, 82005, 82184, 82298, 82351, 82379, 82421, 82730, 82748, 82778]
# # qwen
# coco_missing_obj_list = []
####
for i in range(len(info_save_index)):    
    if i==0:
        input_datas[i]=[input_datas[i][j] for j in range(len(input_datas[i])) if j not in coco_missing_obj_list ]
    else:
        input_datas[i]=[input_datas[i][j] for j in range(len(input_datas[i])) if j not in coco_missing_obj_list ]
####

# if isinstance(input_datas, list):
#     # Convert the list to a tensor, assuming each element in the list has the same shape
#     input_datas = torch.stack([torch.tensor(data) for data in input_datas])
# # input_datas = input_datas[:, :10, :]
merge_label=None
sample_size = len(input_datas[0])
labels = torch.load(ce_data_dir+'ce_training_label_full_v1.pth')
labels = [label for label in labels if label!=[]]
print(f"sample length, input {len(input_datas[0])}, label{len(labels)}")

for i in range(len(input_datas)):
    random_indices = random.sample(range(len(input_datas[0])), sample_size)
    temp=[input_datas[i][x] for x in random_indices]
    input_datas[i]=temp #input_datas[i][random_indices]
    temp=[labels[x] for x in random_indices]
    if i==0:
        merge_label=temp #labels[random_indices]
    else:
        merge_label+=temp #labels[random_indices]
labels=merge_label

written_file=ce_data_dir + "ce_result_merge_v1.txt"


vit=False
one_item=False
is_qwen = True

# Define model
class CE(nn.Module):
    def __init__(self):
        super(CE, self).__init__()

        if is_qwen:
            self.fc1 = nn.Linear(2048, 512)
        else:
            self.fc1 = nn.Linear(4096, 512)  # Assuming each tensor is 4096-dim
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, len(labels[0])) 
        print(len(labels[0]))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = torch.sigmoid(self.fc3(x))
        x = self.fc3(x)
        return x


if vit==True:
    written_file="vit_"+written_file
    info_save_index=[0]

results={}
inference_result={}

############
# print(f"-----------------index {index}-----------------")

# Initialize model, loss function and optimizer
model = CE().to('cuda:0')

# criterion = nn.BCELoss()  # Binary cross entropy loss
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()

# optimizer = optim.Adam(model.parameters(), lr=0.0001)
# For Qwen
optimizer = optim.Adam(model.parameters(), lr=0.00005)

merge_data=None

if not vit:
    for i in range(len(info_save_index)):    
        if i==0:
            merge_data=input_datas[i]
            # merge_label=labels[:]
        else:
            merge_data+=input_datas[i]
            # merge_label+=labels[:]

input_data=merge_data
# labels=merge_label

# Convert data to Tensor
if vit:
    data_tensor = torch.stack(vit_feature_list)
else:
    data_tensor = torch.stack(input_data)  # (N, 1024) assuming each tensor is 1024-dim
labels_tensor = torch.tensor(labels, device='cuda:0')

# Create TensorDataset
dataset = TensorDataset(data_tensor, labels_tensor)

# # Split into train and test sets
# train_size = int(0.9 * len(dataset))
# test_size = len(dataset) - train_size

# # Create indices
# train_indices = list(range(train_size))
# test_indices = list(range(train_size, len(dataset)))

# # Use Subset to create subsets
# train_dataset = Subset(dataset, train_indices)
# test_dataset = Subset(dataset, test_indices)
# Split into train and test sets
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

# Randomize the dataset and split
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True) #64
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# vit_feature_list
all_losses=[]
all_f1=[]
all_accuracies = []
classifiy_res_index=[]
gt_res_index=[]
for epoch in range(num_epochs):
    classifiy_res_epoch=[]
    gt_res_epoch=[]
    
    model.train()  # Set model to training mode
    for inputs, target in train_loader:
        inputs = inputs.float().to('cuda:0')
        if not one_item:
            target = target.float().to('cuda:0')
        else:
            target=target.unsqueeze(1).float()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
    
    # Test the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        total_kl=[]
        for inputs, target in test_loader:
            inputs = inputs.float().to('cuda:0')
            # target = target.int()
            
            predicted = model(inputs)
            
            classifiy_res_epoch+=predicted
            if gt_res_index==[]:
                gt_res_epoch+=target
            
            for j in range(len(target)):
                if torch.all(target[j] == 0):
                    continue
                
                # Compute softmax and log
                # softmax_pred = F.softmax(predicted[j], dim=0)
                # log_softmax_pred = softmax_pred.log()
                log_softmax_pred=torch.log_softmax(predicted[j], dim=0)
                target_float = target[j].float()
                
                
                kl_div = F.kl_div(log_softmax_pred, target_float, reduction='batchmean')
                
                # total_kl.append(F.kl_div(F.softmax(predicted[j]).log(), target[j].float()))
                
                # Debug: check for NaN
                if torch.isnan(kl_div):
                    print(f"NaN detected at index {j}:")
                    # print(f"Softmax result: {softmax_pred}")
                    print(f"Log Softmax result: {log_softmax_pred}")
                    print(f"Target (float): {target_float}")

                total_kl.append(kl_div)

    
    if one_item:
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
    else:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, kl_div: {sum(total_kl)/len(total_kl):.2f}, middle case: {sum(total_kl):.2f}, {len(total_kl):.2f}')
    # Save loss and accuracy for current epoch
    all_losses.append(loss.item())
    if one_item:
        all_accuracies.append(accuracy)
    else:
        all_f1.append(sum(total_kl)/len(total_kl))
    
    classifiy_res_index.append(classifiy_res_epoch)
    if gt_res_index==[]:
        gt_res_index=gt_res_epoch
    
    if (epoch+1)%2==0:
        torch.save(model.state_dict(), f'./result/ckpt/ce_model_merge_epoch{epoch}_v1.pth')
    
    # break
    
# Store results for current index into dictionary
if vit:
    results[0] = {
        'losses': all_losses,
        'accuracies': all_f1,
    }
        
    inference_result[0] = {
        'classifiy_res': classifiy_res_index,
        'gt': gt_res_index
    }
else:
    if one_item:
        results[0] = {
            'losses': all_losses,
            'accuracies': all_accuracies,
        }
        
        inference_result[0] = {
            'classifiy_res': classifiy_res_index,
            'gt': gt_res_index
        }
    else:
        results[0] = {
            'losses': all_losses,
            'accuracies': all_f1,
        }
        
        inference_result[0] = {
            'classifiy_res': classifiy_res_index,
            'gt': gt_res_index
        }
torch.save(model.state_dict(), f'./result/ce_model_merge_v1.pth')
    
with open(written_file, "w") as f:
    if one_item:
        f.write("Index\tEpoch\tLoss\tAccuracy\n")
    else:
        f.write("Index\tEpoch\tLoss\tF1\n")
    for index, metrics in results.items():
        for epoch in range(num_epochs):
            f.write(f"{index}\t{epoch + 1}\t{metrics['losses'][epoch]:.4f}\t{metrics['accuracies'][epoch]:.2f}\n")

print(written_file[:-4]+".pth")
torch.save(inference_result, written_file[:-4]+".pth")

print("Training completed!")