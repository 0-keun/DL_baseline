import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import gpytorch
import torch.nn.functional as F

class ResidualFFNNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualFFNNBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        residual = x  # 입력 저장 (skip connection)
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out += residual   # Residual connection
        out = F.relu(out)
        return out

class ResidualFFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=64):
        super(ResidualFFNN, self).__init__()
        self.block1 = ResidualFFNNBlock(input_dim, hidden_dim)
        self.block2 = ResidualFFNNBlock(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.fc_out(out)
        return out