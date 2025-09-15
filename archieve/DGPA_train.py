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
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

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
    
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
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
class DGPAModel(nn.Module):
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

# --------------------------------------
# 6. Loss Function
# --------------------------------------

def gaussian_nll_loss(y_pred_mean, y_pred_std, y_true):
    var = y_pred_std ** 2 + 1e-6  # 수치 안정성 확보
    error_weight = 5
    return torch.mean(0.5 * torch.log(2 * math.pi * var) + 0.5 * (y_true - y_pred_mean) ** 2 / var)

def custom_elbo_like_loss(pred_mean, pred_std, y_true, model, kl_weight=1e-6):
    var = pred_std ** 2 + 1e-6  # 수치 안정성 확보
    log_likelihood = -0.5 * torch.log(2 * math.pi * var) - 0.5 * (y_true - pred_mean) ** 2 / var
    nll = -torch.mean(log_likelihood)

    # KL 유사 항: 모델 파라미터에 대한 prior penalty
    kl_div = 0.0
    for param in model.parameters():
        kl_div += torch.sum(param ** 2)
    kl_div *= kl_weight  # 작은 계수 곱함

    return nll + kl_div

#############################################
##                   Main                  ##
#############################################
if __name__ == "__main__":
    p = load_json('./params.json')
    df = pd.read_csv(p.train_data_dir)

    MODEL_DIR = name_to_dir(name='model',time_flag=True)
    SAVE_NORMALIZATION_FILE = False

    X = df[p.feature_list].values 
    y = df[p.output_list].values

    y_0 = []
    for y_instance in y:
        y_0.append(y_instance[0])

    # 텐서로 변환
    X_train_tensor = torch.tensor(X, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_0, dtype=torch.float32)

    # DataLoader
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)
    # val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=64, shuffle=False)


    if SAVE_NORMALIZATION_FILE:
        scaler = normalize_and_save(X,time_flag=True)
        X = normalize_std_scaler(X, scaler)
    else:
        X = load_and_normalize(X,'./scaler/scaler_250612/mean_213605.npy','./scaler/scaler_250612/scale_213605.npy')

    model = DGPAModel(input_dim=len(p.feature_list), hidden_dim=256, rff_dim=256, num_blocks=5)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)     # 100 에폭마다 lr *= 0.5
    num_epochs = 1000

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred_mean, pred_std = model(x_batch)
            loss = gaussian_nll_loss(pred_mean, pred_std, y_batch)
            # loss = custom_elbo_like_loss(pred_mean, pred_std, y_batch, model, kl_weight=1e-6)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(X)

        # # Validation
        # model.eval()
        # val_loss = 0
        # with torch.no_grad():
        #     for x_val, y_val_ in val_loader:
        #         val_mean, val_std = model(x_val)
        #         loss = gaussian_nll_loss(val_mean, val_std, y_val_)
        #         val_loss += loss.item()
        # val_loss /= len(val_loader)

        print(f"[{epoch+1}/{num_epochs}] Train Loss: {train_loss:.6f}") #| Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), f"{MODEL_DIR}/dgpa_model.pt")

    # x = torch.randn(16, len(p.feature_list))  # 배치 사이즈 16, 입력 차원 10
    # mean, std = model(x)

    # print("Mean shape:", mean.shape)  # (16, 1)
    # print("Std shape:", std.shape)    # (16, 1)
    # print(y)
