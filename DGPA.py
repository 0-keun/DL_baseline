'''
reference paper: Uncertainty aware machine-learning-based surrogate models for particle accelerators: Study at the Fermilab Booster Accelerator Complex
Model: DGPA (Deep Gaussian Process Approximation)
Main algotithm: GP(Gaussian Process), SN(Spectral Normalization), RFF(Random Fourier Features), bi-Lipschitz
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math

from utils.data_processing import normalize_and_save, normalize_std_scaler, load_and_normalize
from utils.models import FFNN_model
from utils.utils import load_json, save_acc_plot, save_loss_plot, name_date, name_time, name_to_dir
import pandas as pd

# --------------------------------------
# 1. Residual Block with Spectral Norm
# --------------------------------------
class ResidualBlockSN(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear = spectral_norm(nn.Linear(dim, dim))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        out = self.dropout(out)
        return x + out  # Residual connection

# --------------------------------------
# 2. Feature Extractor (stacked residual blocks)
# --------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlockSN(hidden_dim) for _ in range(num_layers)])

    def forward(self, x):
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return h  # final hidden representation

# --------------------------------------
# 3. Random Fourier Features Layer
# --------------------------------------
class RFFLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(input_dim, output_dim), requires_grad=False)
        self.b = nn.Parameter(2 * math.pi * torch.rand(output_dim), requires_grad=False)

    def forward(self, h):
        projection = h @ self.W + self.b
        return math.sqrt(2.0 / self.W.shape[1]) * torch.cos(projection)

# --------------------------------------
# 4. Output layer: Predict mean and log_std
# --------------------------------------
class GPOutputLayer(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.mean_layer = nn.Linear(input_dim, output_dim)
        self.log_std_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        std = torch.exp(log_std)
        return mean, std

# --------------------------------------
# 5. 전체 모델 구성
# --------------------------------------
class SNGPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, rff_dim, num_blocks, output_dim=1):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim, num_blocks)
        self.rff = RFFLayer(hidden_dim, rff_dim)
        self.output_layer = GPOutputLayer(rff_dim, output_dim)

    def forward(self, x):
        h = self.feature_extractor(x)
        phi = self.rff(h)
        mean, std = self.output_layer(phi)
        return mean, std  # for regression: mean, std / for classification: logits, uncertainty
    

p = load_json('./params.json')
df = pd.read_csv(p.train_data_dir)

MODEL_DIR = name_to_dir(name='model',time_flag=True)
SAVE_NORMALIZATION_FILE = False

X = df[p.feature_list].values 
y = df[p.output_list].values  

if SAVE_NORMALIZATION_FILE:
    scaler = normalize_and_save(X,time_flag=True)
    X = normalize_std_scaler(X, scaler)
else:
    X = load_and_normalize(X,'./scaler/scaler_250612/mean_213605.npy','./scaler/scaler_250612/scale_213605.npy')

model = SNGPModel(input_dim=len(p.feature_list), hidden_dim=256, rff_dim=256, num_blocks=4)
x = torch.randn(16, len(p.feature_list))  # 배치 사이즈 16, 입력 차원 10
mean, std = model(x)

print("Mean shape:", mean.shape)  # (16, 1)
print("Std shape:", std.shape)    # (16, 1)
