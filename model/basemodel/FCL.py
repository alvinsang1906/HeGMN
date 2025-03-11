import torch
import torch.nn as nn
import torch.nn.functional as F

class FCL(nn.Module):
    r"""四层全连接层"""
    def __init__(self, in_dim):
        super(FCL, self).__init__()
        self._fc1 = nn.Linear(in_dim, in_dim // 2)          # 256 -> 128
        self._fc2 = nn.Linear(in_dim // 2, in_dim // 4)     # 128 -> 64
        self._fc3 = nn.Linear(in_dim // 4, in_dim // 8)     # 64  -> 32
        self._score = nn.Linear(in_dim // 8, 1)             # 32  -> 1
    
    def forward(self, h_1, h_2):
        h = torch.cat([h_1, h_2], dim=1)
        score = F.relu(self._fc1(h))
        score = F.relu(self._fc2(score))
        score = F.relu(self._fc3(score))
        score = F.sigmoid(self._score(score))
        return score.view(-1)
