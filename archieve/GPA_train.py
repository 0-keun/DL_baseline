import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

from utils.data_processing import normalize_and_save, normalize_std_scaler, load_and_normalize
from utils.utils import load_json, name_to_dir


# --------------------------------------
# 1. RFF Layer
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
# 2. Output layer: Predict mean and log_std
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
# 3. GPA-only Model
# --------------------------------------
class GPAOnlyModel(nn.Module):
    def __init__(self, input_dim, rff_dim, output_dim=1):
        super().__init__()
        self.rff = RFFLayer(input_dim, rff_dim)
        self.output_layer = GPOutputLayer(rff_dim, output_dim)

    def forward(self, x):
        phi = self.rff(x)
        mean, std = self.output_layer(phi)
        return mean, std

# --------------------------------------
# 4. Loss Function
# --------------------------------------
def gaussian_nll_loss(y_pred_mean, y_pred_std, y_true):
    var = y_pred_std ** 2 + 1e-6  # 수치 안정성 확보
    return torch.mean(0.5 * torch.log(2 * math.pi * var) + 0.5 * (y_true - y_pred_mean) ** 2 / var)


# --------------------------------------
# 5. Main Train Script
# --------------------------------------
if __name__ == "__main__":
    # 하이퍼파라미터 및 설정 로드
    p = load_json('./params.json')
    df = pd.read_csv(p.train_data_dir)

    MODEL_DIR = name_to_dir(name='model', time_flag=True)
    SAVE_NORMALIZATION_FILE = False

    # 데이터 로딩
    X = df[p.feature_list].values 
    y = df[p.output_list].values
    y_0 = [y_instance[0] for y_instance in y]  # 단일 출력값 사용

    # 정규화
    if SAVE_NORMALIZATION_FILE:
        scaler = normalize_and_save(X, time_flag=True)
        X = normalize_std_scaler(X, scaler)
    else:
        X = load_and_normalize(X, './scaler/scaler_250612/mean_213605.npy', './scaler/scaler_250612/scale_213605.npy')

    # Tensor 변환
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_0, dtype=torch.float32)

    # DataLoader
    train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)

    # 모델 초기화
    model = GPAOnlyModel(input_dim=len(p.feature_list), rff_dim=256)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 학습
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred_mean, pred_std = model(x_batch)
            loss = gaussian_nll_loss(pred_mean, pred_std, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(X)
        print(f"[{epoch+1}/{num_epochs}] Train Loss: {train_loss:.6f}")

    # 모델 저장
    torch.save(model.state_dict(), f"{MODEL_DIR}/gpa_model.pt")
