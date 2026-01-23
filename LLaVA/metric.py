import numpy as np
import torch
import torch.nn as nn
import torch.distributions as distributions

# 假设我们的特征维度是4096
DIM = 4096

class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        self.fc1 = nn.Linear(DIM, 512)
        self.fc2_mu = nn.Linear(512, DIM)
        self.fc2_logvar = nn.Linear(512, DIM)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc2_mu(h)
        logvar = self.fc2_logvar(h)
        return mu, logvar

def kl_divergence(q_mu, q_logvar, r_mu, r_logvar):
    # 计算KL散度
    kl = (r_logvar - q_logvar) + (torch.exp(q_logvar) + (q_mu - r_mu)**2) / (2 * torch.exp(r_logvar)) - 0.5
    return kl.sum()

def mutual_information_upper_bound(encoder, visual_input, prior_mu, prior_logvar):
    q_mu, q_logvar = encoder(visual_input)
    kl = kl_divergence(q_mu, q_logvar, prior_mu, prior_logvar)
    return kl.mean()

# 条件互信息
def conditional_mutual_information_upper_bound(encoder, visual_input, text_instruction, prior_cond_mu, prior_cond_logvar):
    q_mu, q_logvar = encoder(visual_input)
    kl = kl_divergence(q_mu, q_logvar, prior_cond_mu, prior_cond_logvar)
    return kl.mean()

# 示例用法
encoder = VariationalEncoder()

# 随机生成输入
visual_input = torch.randn(1, DIM)
prior_mu = torch.zeros(DIM)
prior_logvar = torch.zeros(DIM)

# 计算互信息上界
mi_upper_bound = mutual_information_upper_bound(encoder, visual_input, prior_mu, prior_logvar)
print(f'Mutual Information Upper Bound: {mi_upper_bound.item()}')

# 条件互信息上界
text_instruction = "example instruction"  # 示例文本指令
prior_cond_mu = torch.zeros(DIM)
prior_cond_logvar = torch.zeros(DIM)

cmi_upper_bound = conditional_mutual_information_upper_bound(encoder, visual_input, text_instruction, prior_cond_mu, prior_cond_logvar)
print(f'Conditional Mutual Information Upper Bound: {cmi_upper_bound.item()}')