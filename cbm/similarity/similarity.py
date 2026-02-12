import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import random

class SimilarityNetwork(nn.Module):
    def __init__(self):
        super(SimilarityNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
        )

    def forward(self, feature1, feature2):
        feature1 = self.encoder(feature1)
        feature2 = self.encoder(feature2)
        feature1 = torch.tanh(feature1)
        feature2 = torch.tanh(feature2)
        
        logits_feature1 = feature1 @ feature2.t() / feature1[0].shape[-1]
        logits_feature2 = logits_feature1.t()

        # logits_feature1 = feature1 @ feature2.t()
        return logits_feature1, logits_feature2


# index=1
for only_image in [False]:
# only_image=True
    input_datas = torch.load('../data/ce_training_full_v1.pth') if not only_image else torch.load('../data/ce_training_image_full_v1.pth')

    index_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31] 
    select_ratio=0.2
    for i, index in enumerate(index_list):
        rand_index = random.sample(range(len(input_datas[index])), k=int(select_ratio*len(input_datas[index])))
        input_datas[index] = torch.stack(input_datas[index]).to(torch.float)[rand_index]

        if i==0:
            input_L = torch.stack(input_datas[-1])[rand_index].to(torch.float)
        else:
            input_L = torch.cat((input_L, torch.stack(input_datas[-1])[rand_index].to(torch.float)), dim=0)

    input_l = input_datas = torch.stack(input_datas[:32]).to(torch.float).view(-1, 4096)
    
    print(input_l.shape, input_L.shape)
    input_l = input_l.to("cuda:0")
    input_L = input_L.to("cuda:0")

    # 4. Create data loader
    train_dataset = TensorDataset(input_l, input_L)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 5. Training process
    def train_model(model, train_loader, criterion, optimizer, num_epochs=30, save_per=50):
        model.train()
        for epoch in range(num_epochs):
            for batch_input_l, batch_input_L in train_loader:
                optimizer.zero_grad()
                # Calculate similarity
                similarity_scores1, similarity_scores2 = model(batch_input_l, batch_input_L)
                # print(similarity_scores.shape, feature1.shape, feature2.shape) #[32, 32] [32, 64] [bs, dim]

                labels = torch.arange(batch_input_l.size(0), device=batch_input_l.device)

                loss1 = criterion(similarity_scores1, labels)
                loss2 = criterion(similarity_scores2, labels)
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            
            if (epoch + 1) % save_per == 0:
                if not only_image:
                    torch.save(model.state_dict(), f'./similarity_network_full_v1_combine_epoch{epoch}.pth')
                else:
                    torch.save(model.state_dict(), f'./similarity_network_only_image_full_v1_combine_epoch{epoch}.pth')

    # 6. Initialize model, loss function and optimizer
    model = SimilarityNetwork().to("cuda:0")
    criterion = nn.CrossEntropyLoss().to("cuda:0")  # Use cross-entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=400)